"""`torch_tools.default_dtype` puts the process default back, exception or not.

The default dtype is process-wide, and this scope guards code that raises.
`MACECalculator.calculate` runs inside it, in nine places across the two
calculator classes, and so does every converter entry point -- which sets the
default from the source model's parameters before it can refuse a model it does
not accept.

Without a `finally` an exception left the whole process on the scope's dtype, and
nothing reads that back, so the next unrelated tensor was simply built at the
wrong precision. Under pytest the same worker then ran every later test at
float64, which is the kind of contamination that makes a test pass alone and fail
in a suite.
"""

import pytest
import torch

from mace.tools import torch_tools


@pytest.fixture(name="float32_default", autouse=True)
def fixture_float32_default():
    """Leave the process where it was found, whatever these tests do to it."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous)


def test_the_scope_sets_the_dtype_it_was_given():
    with torch_tools.default_dtype("float64"):
        assert torch.get_default_dtype() is torch.float64


def test_a_clean_exit_restores_the_default():
    with torch_tools.default_dtype("float64"):
        pass

    assert torch.get_default_dtype() is torch.float32


def test_an_exception_restores_the_default_too():
    """The bug. A converter that refuses a model raises after setting the default
    from that model, so this is the ordinary path rather than an edge case."""
    with pytest.raises(TypeError):
        with torch_tools.default_dtype("float64"):
            raise TypeError("the converter refuses this model")

    assert torch.get_default_dtype() is torch.float32


def test_the_exception_still_reaches_the_caller():
    """Restoring must not swallow it: `pytest.raises` above would pass on a
    `finally` that returned normally, but a caller needs the error."""
    with pytest.raises(ValueError, match="propagated"):
        with torch_tools.default_dtype("float64"):
            raise ValueError("propagated")


def test_a_torch_dtype_is_accepted_as_well_as_a_string():
    with torch_tools.default_dtype(torch.float64):
        assert torch.get_default_dtype() is torch.float64

    assert torch.get_default_dtype() is torch.float32


def test_nesting_unwinds_to_the_outer_dtype():
    with torch_tools.default_dtype("float64"):
        with torch_tools.default_dtype("float32"):
            assert torch.get_default_dtype() is torch.float32
        assert torch.get_default_dtype() is torch.float64

    assert torch.get_default_dtype() is torch.float32


def test_nesting_unwinds_through_an_exception():
    with torch_tools.default_dtype("float64"):
        with pytest.raises(RuntimeError):
            with torch_tools.default_dtype("float32"):
                raise RuntimeError("inner")
        assert torch.get_default_dtype() is torch.float64

    assert torch.get_default_dtype() is torch.float32


# ---------------------------------------------------------------------------
# The converters, which are library functions and not only CLI entry points
# ---------------------------------------------------------------------------

CONVERTER_MODULES = [
    "mace.cli.convert_e3nn_cueq",
    "mace.cli.convert_cueq_e3nn",
    "mace.cli.convert_e3nn_oeq",
    "mace.cli.convert_oeq_e3nn",
    "mace.cli.convert_e3nn_hybrid",
]


def test_the_decorator_restores_after_a_body_that_raises():
    @torch_tools.restores_default_dtype
    def refuses_the_model():
        torch.set_default_dtype(torch.float64)
        raise TypeError("not a cueq model")

    with pytest.raises(TypeError):
        refuses_the_model()

    assert torch.get_default_dtype() is torch.float32


def test_the_decorator_restores_after_a_body_that_succeeds():
    """The half a `try/except` around the call site would not have covered:
    the converters changed the default on every successful conversion too."""

    @torch_tools.restores_default_dtype
    def converts():
        torch.set_default_dtype(torch.float64)
        return torch.zeros(1)

    built = converts()

    assert built.dtype is torch.float64, "the body must still get its dtype"
    assert torch.get_default_dtype() is torch.float32


def test_the_decorator_keeps_the_wrapped_signature():
    """`run` is called with keywords (`device=`) from five places in mace/."""

    @torch_tools.restores_default_dtype
    def convert(model, device="cpu"):
        """Docstring kept."""
        return model, device

    assert convert("m", device="cuda") == ("m", "cuda")
    assert convert.__name__ == "convert"
    assert convert.__doc__ == "Docstring kept."


@pytest.mark.parametrize("module_name", CONVERTER_MODULES)
def test_every_converter_run_restores_the_default_dtype(module_name):
    """Source-level, because a real conversion needs cueq or oeq installed and
    the point is the entry points that are *not* covered by a GPU job.

    Each `run` reads the dtype off the source model and sets it process-wide so
    the submodules it builds inherit it. That is fine inside the call and wrong
    after it: `MACECalculator(enable_cueq=True)` and `run_train` both call these
    as ordinary functions, so the leak lands in a caller that already chose its
    own dtype.
    """
    import ast  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    import importlib  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    import inspect  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    # The modules import their backend lazily, so parsing beats importing: oeq
    # and cueq are absent from the CPU suite where this test has to run.
    path = importlib.util.find_spec(module_name).origin
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    run = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    decorators = {ast.unparse(d) for d in run.decorator_list}

    assert "restores_default_dtype" in decorators, (
        f"{module_name}.run sets the process default dtype from the source "
        "model; it must put the caller's back"
    )
