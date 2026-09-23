"""Repeating the other heads when one head outnumbers them all.

A replay set of ten thousand structures beside a fine-tuning set of fifty
makes the fine-tune a footnote. The frozen tree guards against it
(``mace/cli/run_train.py:443-457``): when the other heads' structures number
fewer than ``threshold`` times the replay head's, each of them is repeated.

The arithmetic is the frozen tree's exactly, including the part that reads
oddly. The ratio is the *other* heads' total over the reference head's, and a
head below the threshold is extended by ``int(threshold / ratio)`` copies of
itself, so it ends up ``1 + int(threshold / ratio)`` times as long. At a ratio
of exactly the threshold nothing is repeated, since the test is a strict
``<``.

Nothing here knows which head is the replay one. The configuration names the
reference head, and the rest is arithmetic on counts.
"""

from __future__ import annotations

__all__ = ["RatioGuardError", "repeat_count"]


class RatioGuardError(ValueError):
    """A ratio that cannot be formed from the heads as they stand."""


def repeat_count(reference: int, others: int, threshold: float) -> int:
    """How many times each other head's structures appear after the guard.

    Args:
        reference: The reference head's training structures.
        others: Every other head's, together.
        threshold: Below this ratio of ``others / reference`` they are repeated.

    Returns:
        ``1`` when no repetition is needed, otherwise ``1 + int(threshold /
        ratio)``: the original and the added copies.

    Raises:
        RatioGuardError: If either count is zero, where the frozen tree divides
            by zero. A reference head with no structures has nothing to be
            outnumbered by, and other heads with none have nothing to repeat.
    """
    if reference <= 0:
        raise RatioGuardError(
            "the reference head has no training structures, so there is no "
            "ratio to guard. Check the head the guard names."
        )
    if others <= 0:
        raise RatioGuardError(
            "the heads other than the reference have no training structures "
            "between them, so there is nothing to repeat and no fine-tune to "
            "protect."
        )
    ratio = others / reference
    if ratio >= threshold:
        return 1
    return 1 + int(threshold / ratio)
