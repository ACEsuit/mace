"""Loss variants: one discriminator beside its settings and observable weights."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field

from mace_core.config.base import ConfigSection, default_choice

__all__ = [
    "LOSS_SHARED_FIELDS",
    "HuberLoss",
    "L1L2Loss",
    "LossConfig",
    "LossWeights",
    "RegisteredLoss",
    "UniversalLoss",
    "WeightedLoss",
    "normalize_loss",
]

LossWeights = dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]]


class _LossSettings(ConfigSection):
    """Per-observable weights. Omitted requested quantities have weight one.

    Stage overrides replace only the weights they name. Structure weights are
    enabled unless explicitly disabled for an observable; label masks always apply.
    """

    weights: LossWeights = Field(default_factory=dict)
    stage_two_weights: LossWeights = Field(default_factory=dict)
    use_structure_weights: dict[str, bool] = Field(default_factory=dict)

    def weight(self, observable: str, *, stage_two: bool = False) -> float:
        if stage_two and observable in self.stage_two_weights:
            return self.stage_two_weights[observable]
        return self.weights.get(observable, 1.0)

    def validate_names(self, names: set[str]) -> None:
        """Reject misspelled or unrequested quantities before building a loss."""
        for field in sorted(LOSS_SHARED_FIELDS):
            unknown = getattr(self, field).keys() - names
            if unknown:
                raise ValueError(
                    f"loss.{field} has unrequested names {sorted(unknown)}; "
                    f"choose from {sorted(names)} or declare them in model.observables"
                )


LOSS_SHARED_FIELDS = frozenset(_LossSettings.model_fields)


class WeightedLoss(_LossSettings):
    """Squared error, weighted per observable. The default."""

    kind: Literal["weighted"] = "weighted"


class HuberLoss(_LossSettings):
    """Huber loss; delta is the crossover in the scored quantity's units."""

    kind: Literal["huber"] = "huber"
    delta: float = Field(default=0.01, gt=0, allow_inf_nan=False)


class UniversalLoss(_LossSettings):
    """Conditional Huber; optionally weight residuals before band selection."""

    kind: Literal["universal"] = "universal"
    delta: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    weight_residuals: bool = False


class L1L2Loss(_LossSettings):
    """Absolute scalar error and vector norm for per-atom quantities."""

    kind: Literal["l1l2"] = "l1l2"


class RegisteredLoss(_LossSettings):
    """An extension's loss; its constructor validates its own settings."""

    kind: Literal["registered"] = "registered"
    name: str = Field(min_length=1)
    settings: dict[str, Any] = Field(default_factory=dict)


def normalize_loss(value: Any) -> Any:
    return default_choice(value, default="weighted", common=LOSS_SHARED_FIELDS)


LossConfig = Annotated[
    WeightedLoss | HuberLoss | UniversalLoss | L1L2Loss | RegisteredLoss,
    Field(discriminator="kind"),
    BeforeValidator(normalize_loss),
]
