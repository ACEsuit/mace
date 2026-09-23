"""Freezing the groups of a model up to a level, as the frozen tree layers them.

Legacy's ``--freeze N`` (``mace/tools/scripts_utils.py:944-960``) freezes from
the bottom of the model up: one or more freezes the node embedding, five the
interactions, six the products, seven the readouts. The gaps are the frozen
tree's, and so is what a freeze does: the group's parameters stop taking
gradients **and** its learning-rate factor goes to zero, so an optimizer built
from the factors alone agrees with one built from the flags.

A soft freeze is the factors without the flags, a group trained slowly rather
than not at all, and it needs nothing here: it is the per-group factors of the
schedule section, set by hand.
"""

from __future__ import annotations

from torch import nn

from mace_torch.train.optimizers import GROUP_MARKERS

__all__ = ["FREEZE_LEVELS", "freeze", "frozen_groups"]

#: The level from which each group is frozen.
FREEZE_LEVELS: dict[str, int] = {
    "embedding": 1,
    "interactions": 5,
    "products": 6,
    "readouts": 7,
}


def frozen_groups(level: int | None) -> tuple[str, ...]:
    """The groups a level freezes, in the order the model runs them."""
    if not level:
        return ()
    return tuple(group for group, start in FREEZE_LEVELS.items() if level >= start)


def freeze(model: nn.Module, level: int | None) -> dict[str, float]:
    """Stop the groups a level names from training.

    Returns:
        The learning-rate factor of each frozen group, which is zero, for the
        schedule's per-group factors. An empty mapping for no freeze.

    Raises:
        ValueError: If a group the level names holds no parameter at all, which
            is a freeze that freezes nothing.
    """
    factors: dict[str, float] = {}
    for group in frozen_groups(level):
        marker = GROUP_MARKERS[group]
        held = [
            parameter
            for name, parameter in model.named_parameters()
            if marker in f".{name}."
        ]
        if not held:
            raise ValueError(
                f"freezing at level {level} names the {group!r} group, and the "
                f"model holds no parameter in it."
            )
        for parameter in held:
            parameter.requires_grad = False
        factors[group] = 0.0
    return factors
