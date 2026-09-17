"""Fixtures for the in-process legacy-vs-v1 comparisons.

`tests/parity/` is one of the two places allowed to import both stacks. The
full harness with process-state snapshot and restore is PAR-1's; until it lands
the tests here compare pure-math blocks whose only global is the default dtype,
which the fixture below sets and restores.
"""

import pytest
import torch


@pytest.fixture(name="fp64")
def fixture_fp64():
    """Both stacks read `torch.get_default_dtype()` at construction."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)
