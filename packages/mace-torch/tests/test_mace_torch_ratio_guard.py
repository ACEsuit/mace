"""The replay-to-real guard, against the frozen tree's arithmetic.

`mace/cli/run_train.py:443-457` computes `(total - replay) / replay` and, below
the threshold, extends each fine-tuning head by `int(threshold / ratio)` copies
of itself. The expected counts below are that arithmetic written out by hand.
"""

from __future__ import annotations

import pytest
from mace_torch.finetune.ratio import RatioGuardError, repeat_count


def legacy(reference: int, others: int, threshold: float) -> int:
    """The frozen tree's lines, transcribed: the length multiplier they give."""
    ratio = others / reference
    if ratio < threshold:
        return 1 + int(threshold / ratio)
    return 1


@pytest.mark.parametrize(
    ("reference", "others", "threshold", "copies"),
    [
        (1000, 200, 0.1, 1),  # well above: untouched
        (1000, 100, 0.1, 1),  # exactly at: `<` is strict, untouched
        (1000, 99, 0.1, 2),  # just below: one extra copy
        (1000, 50, 0.1, 3),  # ratio 0.05, int(2.0) extra copies
        (10000, 30, 0.1, 34),  # ratio 0.003, int(33.33) extra copies
        (100, 7, 0.5, 8),  # another threshold
    ],
)
def test_the_count_is_the_frozen_trees(reference, others, threshold, copies):
    assert repeat_count(reference, others, threshold) == copies
    assert repeat_count(reference, others, threshold) == legacy(
        reference, others, threshold
    )


def test_the_boundary_is_strict():
    """At the threshold nothing happens; one structure fewer and it does."""
    assert repeat_count(1000, 100, 0.1) == 1
    assert repeat_count(1000, 99, 0.1) > 1


def test_a_reference_with_no_structures_is_refused():
    with pytest.raises(RatioGuardError, match="reference head has no"):
        repeat_count(0, 10, 0.1)


def test_other_heads_with_no_structures_are_refused():
    """Where the frozen tree divides by zero."""
    with pytest.raises(RatioGuardError, match="nothing to repeat"):
        repeat_count(10, 0, 0.1)
