"""Shared fixtures and the one comparison helper of the mace-torch test suite."""

import numpy as np
import pytest
import torch

#: An implementation against the closed form it implements, in one process, at
#: fp64. Mirrors the `closed_form_fp64` row of `tests/golden/harness.py` (atol
#: 1e-12, rtol 1e-12), which this package cannot import: the packages are
#: installed and tested without the repository's `tests/` tree on the path. A
#: change here is a tolerance change and goes through its own reviewed PR.
CLOSED_FORM_FP64_ATOL = 1e-12
CLOSED_FORM_FP64_RTOL = 1e-12


@pytest.fixture(name="fp64")
def fixture_fp64():
    """Modules read `torch.get_default_dtype()` at construction; every numeric
    claim in this suite is an fp64 claim."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def assert_close(actual, expected, what: str = "") -> None:
    """`|actual - expected| <= atol + rtol * |expected|` at the closed-form row."""
    actual = np.asarray(
        actual.detach().cpu().numpy() if torch.is_tensor(actual) else actual,
        dtype=float,
    )
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{what}: {actual.shape} vs {expected.shape}"
    deviation = np.abs(actual - expected)
    bound = CLOSED_FORM_FP64_ATOL + CLOSED_FORM_FP64_RTOL * np.abs(expected)
    worst = int(np.argmax(deviation - bound)) if deviation.size else 0
    assert np.all(deviation <= bound), (
        f"{what}: worst deviation {deviation.ravel()[worst]:.3e} at index {worst} "
        f"exceeds the closed-form fp64 bound"
    )
