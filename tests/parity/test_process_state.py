"""That running the frozen tree leaves nothing behind.

Every other comparison in this directory depends on it. Legacy and the rewrite
run in one process, so if the first changes a global the second inherits it,
the numbers agree, and what was compared is neither stack on its own. That
failure is silent by construction: it makes tests pass.

So the restoration is asserted, and so is the detector that asserts it. A
checker with nothing to check passes, and so does a broken one.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from process_state import (
    WATCHED_ENVIRONMENT,
    capture_state,
    restore_state,
    state_differences,
)

ANCHORS = Path(__file__).resolve().parents[1] / "golden/models"


def run_legacy_once():
    """Load and evaluate a frozen anchor, the way a parity test would.

    Deliberately written the careless way, with no restoration of its own: what
    is being tested is that the harness cleans up after code like this.
    """
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    torch.set_default_dtype(torch.float64)
    model = torch.load(
        ANCHORS / "tiny_scaleshift.model", map_location="cpu", weights_only=False
    )
    torch.randn(4)
    np.random.random(4)
    return model


def test_the_detector_sees_each_global_move():
    """Self-test. Every entry of the snapshot is checked by moving it.

    Without this the restoration test below would pass just as happily against
    a detector that looked at nothing.
    """
    before = capture_state()
    try:
        cases = {
            "environment": lambda: os.environ.__setitem__(
                WATCHED_ENVIRONMENT[0], "a-different-value"
            ),
            # Whichever it is not, since setting it to what it already is
            # disturbs nothing and the case would pass without testing.
            "default dtype": lambda: torch.set_default_dtype(
                torch.float32
                if torch.get_default_dtype() is torch.float64
                else torch.float64
            ),
            "the torch random": lambda: torch.randn(1),
            "the numpy random": lambda: np.random.random(1),
        }
        for expected, disturb in cases.items():
            reference = capture_state()
            disturb()
            changed = state_differences(reference, capture_state())
            assert any(expected in line for line in changed), (
                f"moving {expected!r} produced {changed}, which does not mention it"
            )
            restore_state(reference)
    finally:
        restore_state(before)


def test_a_clean_snapshot_reports_nothing():
    """The other half: it must not cry wolf."""
    first = capture_state()
    assert state_differences(first, capture_state()) == []


def test_the_harness_restores_the_process_after_a_legacy_run(isolated):
    """The requirement itself, exercised on a real load and evaluation.

    The fixture asserts the restoration on its way out; what this adds is that
    the run inside it genuinely disturbed something, so the fixture is not
    being congratulated for a no-op.
    """
    before = capture_state()
    model = run_legacy_once()

    assert model is not None
    moved = state_differences(before, capture_state())
    assert moved, (
        "the legacy run changed no global at all, so this test is not "
        "exercising the restoration it exists for"
    )


def test_an_absent_environment_variable_comes_back_absent():
    """Restoring is not the same as setting it to the value it used to have.

    The frozen tree sets this on import, and a machine where it was never set
    has to end up without it rather than with an empty string.
    """
    name = WATCHED_ENVIRONMENT[0]
    original = os.environ.pop(name, None)
    try:
        snapshot = capture_state()
        os.environ[name] = "1"
        restore_state(snapshot)
        assert name not in os.environ
    finally:
        if original is not None:
            os.environ[name] = original


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_default_dtype_survives_a_legacy_run(dtype, isolated):
    """A foreign default dtype is a known live hazard in the frozen tree."""
    torch.set_default_dtype(dtype)
    inner = capture_state()
    run_legacy_once()
    restore_state(inner)
    assert torch.get_default_dtype() == dtype
