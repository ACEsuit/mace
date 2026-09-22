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

from typing import Annotated, Any, Literal

from pydantic import Field

from mace_core.config.section import FrozenSection

__all__ = [
    "HuberLoss",
    "L1L2Loss",
    "LossConfig",
    "LossKind",
    "RegisteredLoss",
    "UniversalLoss",
    "WeightedLoss",
]


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


class L1L2Loss(FrozenSection):
    """Absolute error on the scalars, vector norm on the per-atom quantities.

    Neither is a squared error, so a force wrong by ``(3, 4, 0)`` costs five
    rather than twenty-five. It carries no settings of its own.
    """

    kind: Literal["l1l2"] = "l1l2"


class RegisteredLoss(FrozenSection):
    """A loss from the registry, by the name it registered under.

    The escape from a closed union into an open registry. The kinds above are
    the ones this schema knows the settings of, so they can be validated; a
    loss from another package cannot be, and naming it with its settings is the
    honest way to say that.

    Args:
        name: The registered name.
        settings: What to pass its constructor. Validated by the loss itself.
    """

    kind: Literal["registered"] = "registered"
    name: str
    settings: dict[str, Any] = Field(default_factory=dict)


#: The loss and, with it, its own hyperparameters. Written kind-as-key, so a
#: file says `[loss.kind.huber]` and `delta` underneath it, and a delta cannot
#: be set on a loss that has none.
LossKind = Annotated[
    WeightedLoss | HuberLoss | UniversalLoss | L1L2Loss | RegisteredLoss,
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
