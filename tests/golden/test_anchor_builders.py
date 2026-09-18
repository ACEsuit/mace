"""No anchor builder may default its output path.

Each `build_*_anchor.py` writes a checkpoint and a sidecar, and every reference in
this directory was recorded against the committed bytes. `build_anchor` used to
default `model_path` to the committed `MODEL_PATH`, so calling it with no argument
overwrote the anchor with a fresh one: the same recipe, a different file, and no
sign of it until the goldens are compared or the diff is read.

A rebuild is a deliberate act, hence `regenerate.py --i-know-what-i-am-doing`. The
signature is the last place that can insist on it, which is what this checks. It
also runs each builder into a temporary directory, because a required argument is
only an improvement if the builders still work.
"""

import ast
import importlib
import inspect
from pathlib import Path

import pytest
import torch

BUILDERS = [
    "build_mace_anchor",
    "build_dipole_anchor",
    "build_maceles_anchor",
    "build_magnetic_anchor",
]

#: `train_anchor.py` is the fifth of these and trains rather than instantiating,
#: so it has its own function name and is not run here: a training belongs in the
#: workflows tier, not in a check about signatures.
TRAINED = ("train_anchor", "train_anchor")


def module(name):
    return importlib.import_module(f"tests.golden.{name}")


@pytest.mark.parametrize(
    "name,function",
    [(n, "build_anchor") for n in BUILDERS] + [TRAINED],
)
def test_the_output_path_has_no_default(name, function):
    signature = inspect.signature(getattr(module(name), function))
    parameter = signature.parameters["model_path"]

    assert parameter.default is inspect.Parameter.empty, (
        f"{name}.{function} defaults its output path, so calling it with no "
        f"argument overwrites the committed anchor"
    )


@pytest.mark.parametrize("name", BUILDERS)
def test_calling_it_without_a_path_is_a_type_error(name):
    """The behaviour the signature buys, stated so it survives a refactor that
    reintroduces a default with a different spelling."""
    with pytest.raises(TypeError):
        module(name).build_anchor()  # pylint: disable=no-value-for-parameter


@pytest.mark.parametrize("name", BUILDERS)
def test_the_main_block_passes_the_committed_path(name):
    """Running the file as a script is still meant to rebuild the anchor in
    place; only the importable function refuses to guess."""
    source = Path(inspect.getfile(module(name))).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "build_anchor"
    ]

    assert calls, f"{name} never calls build_anchor"
    assert any("MODEL_PATH" in call for call in calls), calls


@pytest.mark.parametrize("name", ["build_mace_anchor", "build_dipole_anchor"])
def test_a_builder_still_builds_when_given_a_path(name, tmp_path):
    """Only the two that need no optional dependency; the LES and magnetic
    builders are covered by their own families' tests."""
    destination = tmp_path / "anchor.model"

    written = module(name).build_anchor(destination)

    assert written == destination
    assert destination.exists()
    assert torch.load(destination, map_location="cpu") is not None
    assert (tmp_path / "anchor.build.json").exists(), (
        "the sidecar has to land beside the model it documents, or a build into a "
        "temporary directory overwrites the committed one"
    )
