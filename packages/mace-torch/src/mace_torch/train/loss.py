"""The loss, as one term per requested output.

What a term is depends on what was declared, not on which properties the code
was written for. The frozen tree has a loss class per combination of
properties, and each of them reads a fixed set of attributes off the batch;
here a term exists because the configuration asked for the quantity, and its
weight is an entry keyed by the same name.

Only the squared-error term is built. The Huber variants, the per-stage
schedules and the transform registry are the composable-loss ticket's, and what
is here is what a mini-batch run of the tiny task needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from mace_core.config.loss import LossConfig
from mace_core.outputs import MACEOutput
from torch import Tensor

from mace_torch.data import TrainingBatch

__all__ = ["LossTerm", "UnsupportedLossError", "WeightedLoss", "weighted_loss"]

#: Where each requested name is found on the typed output. A name absent from
#: this map is read out of `extras`, which is what a declared observable with
#: no core field lands in.
_FIELDS = {
    "energy": "total_energy",
    "forces": "forces",
    "stress": "stress",
    "virials": "virials",
}


class UnsupportedLossError(NotImplementedError):
    """A loss kind this stage does not build."""


@dataclass(frozen=True)
class LossTerm:
    """One quantity's contribution.

    Attributes:
        name: The requested name, which is also the key of its weight and of
            its reference values.
        weight: Its multiplier in the total.
        per_atom: Whether the quantity has one row per atom. It decides which
            structure a row belongs to, and therefore which graph weight
            applies to it.
    """

    name: str
    weight: float
    per_atom: bool


class WeightedLoss:
    """Squared error per term, weighted by term and by structure.

    The per-structure weight comes from the batch rather than from the
    configuration, because it is a property of the structure: a
    ``config_type`` weight and a padding graph's zero are the same mechanism.
    """

    def __init__(self, terms: Sequence[LossTerm]) -> None:
        if not terms:
            raise ValueError(
                "a loss with no terms scores nothing. It is built from the "
                "requested outputs, so an empty one means none were requested."
            )
        self.terms = tuple(terms)

    def __call__(self, output: MACEOutput[Tensor], batch: TrainingBatch) -> Tensor:
        """The total, as a scalar."""
        graph_weight = batch.graph["weight"]
        assert isinstance(graph_weight, Tensor)
        node_weight = graph_weight[batch.graph["batch"]]
        total: Tensor | None = None
        for term in self.terms:
            predicted = _predicted(output, term.name)
            reference = batch.targets[term.name].to(predicted.dtype)
            squared = (predicted - reference.reshape(predicted.shape)) ** 2
            while squared.dim() > 1:
                squared = squared.mean(dim=-1)
            weights = node_weight if term.per_atom else graph_weight
            contribution = term.weight * (weights * squared).mean()
            total = contribution if total is None else total + contribution
        assert total is not None
        return total


def weighted_loss(
    config: LossConfig,
    names: Sequence[str],
    per_atom: Sequence[bool],
    *,
    stage_two: bool = False,
) -> WeightedLoss:
    """The loss for a run that asked for ``names``.

    Raises:
        UnsupportedLossError: For a kind the composable-loss work owns. It says
            which, rather than quietly scoring with a different formula than
            the configuration asked for.
    """
    if config.kind.kind != "weighted":
        raise UnsupportedLossError(
            f"the {config.kind.kind!r} loss is not built here yet. Only "
            f"'weighted' is, and substituting it would score the run with a "
            f"formula the configuration did not ask for."
        )
    return WeightedLoss(
        [
            LossTerm(name, config.weight(name, stage_two=stage_two), atomic)
            for name, atomic in zip(names, per_atom, strict=True)
        ]
    )


def _predicted(output: MACEOutput[Tensor], name: str) -> Tensor:
    """One quantity off the typed output, by its requested name."""
    field = _FIELDS.get(name)
    value = getattr(output, field) if field is not None else output.extras.get(name)
    if value is None:
        raise KeyError(
            f"the model produced no {name!r}, and the loss has a term for it. "
            f"It carries {sorted(output.extras)} beyond its core fields."
        )
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name!r} came back as {type(value)!r} rather than a tensor.")
    return value
