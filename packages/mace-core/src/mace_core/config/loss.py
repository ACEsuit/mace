"""What the run is scored on, as weights keyed by observable name.

Legacy has fourteen weight flags, seven for the first stage and seven for the
second, one pair per property it happens to support. A property it does not
support has no weight, so adding one means adding two flags and a branch that
reads them.

Here a weight is an entry keyed by the observable's own name, so declaring a
property is what gives it a weight, and the second stage is an override map
rather than a parallel set of flags. A loss's own hyperparameters live in a
typed ``params`` beside the weights, which is what keeps ``--huber_delta`` and
anything like it a setting rather than a flag.

Constructing the loss from this is the training ticket's; what lives here is
the schema.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from mace_core.config.section import FrozenSection

__all__ = ["HuberLoss", "LossConfig", "LossKind", "UniversalLoss", "WeightedLoss"]


class WeightedLoss(FrozenSection):
    """Squared error, weighted per observable. The default."""

    kind: Literal["weighted"] = "weighted"


class HuberLoss(FrozenSection):
    """Huber loss.

    Args:
        delta: The crossover between the quadratic and linear regimes, in the
            units of whatever it scores.
    """

    kind: Literal["huber"] = "huber"
    delta: float = 0.01


class UniversalLoss(FrozenSection):
    """The conditional-Huber variant the foundation-model recipes use.

    Args:
        delta: As :class:`HuberLoss`.
    """

    kind: Literal["universal"] = "universal"
    delta: float = 0.01


#: The loss and, with it, its own hyperparameters. Written kind-as-key, so a
#: file says `[loss.kind.huber]` and `delta` underneath it, and a delta cannot
#: be set on a loss that has none.
LossKind = Annotated[
    WeightedLoss | HuberLoss | UniversalLoss,
    Field(discriminator="kind"),
]


class LossConfig(FrozenSection):
    """The loss, its hyperparameters, and the per-observable weights.

    Args:
        kind: Which loss, with its own settings under it.
        weights: Observable name to weight. An observable the model declares
            and this omits takes ``1.0``: a declared property contributing
            nothing is a thing to ask for, not a default.
        stage_two_weights: Weights that replace ``weights`` in the second
            stage. An observable absent from it keeps its first-stage weight,
            so a stage that only changes the energy says only that.
    """

    kind: LossKind = WeightedLoss()
    weights: dict[str, float] = Field(default_factory=dict)
    stage_two_weights: dict[str, float] = Field(default_factory=dict)

    def weight(self, observable: str, *, stage_two: bool = False) -> float:
        """The weight ``observable`` carries, in the first or the second stage.

        Args:
            observable: An observable name.
            stage_two: Read the second stage's overrides.

        Returns:
            The override if the stage sets one, the first-stage weight if it
            does not, and ``1.0`` if neither names it.
        """
        if stage_two and observable in self.stage_two_weights:
            return self.stage_two_weights[observable]
        return self.weights.get(observable, 1.0)
