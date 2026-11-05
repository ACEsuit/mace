"""Fixtures for the in-process legacy-vs-v1 comparisons.

`tests/parity/` is one of the two places allowed to import both stacks, which
is what makes the frozen tree a live oracle rather than a folder of JSON.

Running the two in one process is only valid if the first leaves nothing
behind. Every test here that touches the frozen tree runs inside `isolated`,
which snapshots the process's globals and puts them back, and
`test_process_state.py` asserts that it does rather than trusting it.
"""

import pytest
import torch

from tests.parity.process_state import capture_state, restore_state, state_differences

pytest.importorskip("mace_torch", reason="the v1 stack is not installed in this job")


@pytest.fixture(name="fp64")
def fixture_fp64():
    """Both stacks read `torch.get_default_dtype()` at construction."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@pytest.fixture(name="isolated")
def fixture_isolated():
    """Restore every global a legacy run can change, and say if one moved.

    The restoration happens whatever the test did; the assertion afterwards is
    about the fixture doing its job, so a test that leaks is a failure of the
    harness rather than of the comparison it was making.
    """
    before = capture_state()
    try:
        yield before
    finally:
        restore_state(before)
    leaked = state_differences(before, capture_state())
    assert not leaked, "the harness failed to restore: " + "; ".join(leaked)
