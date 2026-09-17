"""Shared fixtures and the one comparison helper of the mace-torch test suite."""

import numpy as np
import pytest
import torch

#: Tolerance per dtype, `(atol, rtol)`. Each mirrors a row of
#: `tests/golden/harness.py`, which this package cannot import: the packages
#: are installed and tested without the repository's `tests/` tree on the
#: path. A change here is a tolerance change and goes through its own reviewed
#: PR.
#:
#: * float64 mirrors `closed_form_fp64`: an implementation against the closed
#:   form it implements, in one process, where the two sides differ only in
#:   the order of the arithmetic.
#: * float32 mirrors `fp32`: the same fp64 literals evaluated in fp32.
#:   Relative, with an absolute floor for values legitimately near zero.
TOLERANCES = {
    torch.float64: (1e-12, 1e-12),
    torch.float32: (5e-5, 1e-3),
}


@pytest.fixture(
    params=[torch.float32, torch.float64], ids=["float32", "float64"], autouse=True
)
def dtype(request):
    """Modules read `torch.get_default_dtype()` at construction, so every test
    in this suite runs once per supported dtype."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(previous)


#: Restricts a test to fp64 by overriding the `dtype` parametrization, so the
#: test generates one case rather than a skip. For claims whose *method* needs
#: fp64: `gradcheck` and finite-difference slopes compare against difference
#: quotients that are rounding noise at fp32.
fp64_only = pytest.mark.parametrize(
    "dtype", [torch.float64], ids=["float64"], indirect=True
)


def assert_close(actual, expected, what: str = "") -> None:
    """`|actual - expected| <= atol + rtol * |expected|` at the row of the
    active default dtype."""
    atol, rtol = TOLERANCES[torch.get_default_dtype()]
    actual = np.asarray(
        actual.detach().cpu().numpy() if torch.is_tensor(actual) else actual,
        dtype=float,
    )
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{what}: {actual.shape} vs {expected.shape}"
    deviation = np.abs(actual - expected)
    bound = atol + rtol * np.abs(expected)
    worst = int(np.argmax(deviation - bound)) if deviation.size else 0
    assert np.all(deviation <= bound), (
        f"{what}: worst deviation {deviation.ravel()[worst]:.3e} at index {worst} "
        f"exceeds the {torch.get_default_dtype()} bound"
    )
