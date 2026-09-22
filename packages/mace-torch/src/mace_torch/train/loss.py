"""The loss, generated from what the model was asked to produce.

A term exists because a quantity was declared, not because a class was written
for the combination. The frozen tree has ten loss classes, one per set of
properties it happens to support, selected by string; adding a property there
means forking a class, and the ten differ from each other in ways that are
easier to write again than to read.

**Three things decide a term's value, and each is declared somewhere.** The
quantity's shape says whether its rows are atoms or structures. Its
*extensivity* says whether the residual is compared per atom, which is what
stops a large structure weighing more for being large. And the weights say how
much it counts: a global one per quantity from the configuration, and two
per-structure ones from the data.

The per-structure weights are also the masking. A structure that carries no
value for a property has weight zero for it, so the term contributes nothing
rather than a NaN, and nothing has to know which properties are missing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from mace_core.config.loss import LossConfig
from mace_core.observables import RequestedOutputs
from mace_core.outputs import MACEOutput
from torch import Tensor

from mace_torch.data import TrainingBatch

__all__ = [
    "LOSS_REGISTRY",
    "GeneratedLoss",
    "LossTerm",
    "UnknownLossError",
    "build_loss",
    "reduce_loss",
    "register_loss",
    "terms_for",
]

#: Where each requested name is found on the typed output. A name absent from
#: this map is read out of `extras`, which is what a declared observable with
#: no core field lands in.
_FIELDS = {
    "energy": "total_energy",
    "forces": "forces",
    "stress": "stress",
    "virials": "virials",
}

#: Custom losses, by the name a configuration selects them with. The decorator
#: is the registration: an external package registers the same way and this
#: file is not edited.
LOSS_REGISTRY: dict[str, Callable[..., torch.nn.Module]] = {}


class UnknownLossError(KeyError):
    """A configured loss name nobody registered."""


def register_loss(name: str) -> Callable[[type], type]:
    """Register a loss under ``name``.

    Raises:
        ValueError: If the name is taken. Two losses under one name is a run
            that trains against whichever was imported last.
    """

    def decorate(loss: type) -> type:
        if name in LOSS_REGISTRY:
            raise ValueError(
                f"{name!r} is already a registered loss, from "
                f"{LOSS_REGISTRY[name]!r}. Two under one name means the run "
                f"scores against whichever module was imported last."
            )
        LOSS_REGISTRY[name] = loss
        return loss

    return decorate


def reduce_loss(raw: Tensor) -> Tensor:
    """The mean of an element-wise loss, across ranks when there are several.

    Stated once, here, because it is part of what a loss value *means*: under
    several ranks the mean of the local elements is not the mean of the batch,
    and a run that reduced locally would report a number that depends on how
    the batch was split. The rule is the frozen tree's: the local sum times the
    world size, over the global element count.
    """
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        total = torch.tensor(raw.numel(), device=raw.device, dtype=raw.dtype)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return raw.sum() * dist.get_world_size() / total
    return raw.mean()


@dataclass(frozen=True)
class LossTerm:
    """One quantity's contribution to the total.

    Attributes:
        name: The requested name, which is also the key of its weight, of its
            reference values and of its per-structure weight.
        weight: Its multiplier, from the configuration.
        per_atom: Whether its rows are atoms rather than structures.
        extensive: Whether the residual is divided by the atom count before it
            is squared.
    """

    name: str
    weight: float
    per_atom: bool
    extensive: bool


def terms_for(
    requested: RequestedOutputs, config: LossConfig, *, stage_two: bool = False
) -> tuple[LossTerm, ...]:
    """A term per requested output, with everything read off a declaration."""
    terms: list[LossTerm] = []
    for observable in requested.observables:
        terms.append(
            LossTerm(
                observable.name,
                config.weight(observable.name, stage_two=stage_two),
                observable.per_atom,
                observable.extensive,
            )
        )
        for request in observable.derivatives:
            name = observable.derivative_name(request.wrt)
            if name not in requested.derivatives:
                continue
            terms.append(
                LossTerm(
                    name,
                    config.weight(name, stage_two=stage_two),
                    request.wrt == "pos",
                    request.extensive,
                )
            )
    return tuple(terms)


class GeneratedLoss(torch.nn.Module):
    """The weighted sum of one squared-error term per requested quantity."""

    def __init__(self, terms: Sequence[LossTerm]) -> None:
        super().__init__()
        if not terms:
            raise ValueError(
                "a loss with no terms scores nothing. The terms come from the "
                "requested outputs, so an empty one means none were requested."
            )
        self.terms = tuple(terms)

    def forward(self, output: MACEOutput[Tensor], batch: TrainingBatch) -> Tensor:
        """The total, as a scalar."""
        counts = _atom_counts(batch)
        total: Tensor | None = None
        for term in self.terms:
            predicted = _predicted(output, term.name)
            reference = batch.targets[term.name].to(predicted.dtype)
            residual = predicted - reference.reshape(predicted.shape)
            if term.extensive:
                # Per atom, so a structure twice the size is not twice the
                # error. The division is on the residual and not on the
                # squared term, which is the frozen tree's and squares the
                # normalisation with it.
                residual = residual / _broadcast(counts, residual)
            weights = _weights(batch, term, residual)
            total_term = term.weight * reduce_loss(weights * residual.square())
            total = total_term if total is None else total + total_term
        assert total is not None
        return total


def build_loss(
    requested: RequestedOutputs, config: LossConfig, *, stage_two: bool = False
) -> torch.nn.Module:
    """The loss a run scores on: a registered one, or the generated one.

    Raises:
        UnknownLossError: Naming the value and listing what is registered.
    """
    kind = config.kind.kind
    if kind == "weighted":
        return GeneratedLoss(terms_for(requested, config, stage_two=stage_two))
    if kind not in LOSS_REGISTRY:
        raise UnknownLossError(
            f"{kind!r} is not a registered loss. The registered names are "
            f"{sorted(LOSS_REGISTRY)}, and 'weighted' is the generated one. "
            f"Register yours with @register_loss."
        )
    settings = {
        name: value
        for name, value in config.kind.model_dump().items()
        if name != "kind"
    }
    return LOSS_REGISTRY[kind](**settings)


def _atom_counts(batch: TrainingBatch) -> Tensor:
    """How many atoms each structure has, from `ptr` and never from `batch`."""
    pointer = batch.graph["ptr"]
    assert isinstance(pointer, Tensor)
    return pointer[1:] - pointer[:-1]


def _broadcast(per_graph: Tensor, like: Tensor) -> Tensor:
    """A per-structure quantity shaped to divide or multiply ``like``."""
    value = per_graph.to(like.dtype)
    return value.reshape(-1, *([1] * (like.dim() - 1)))


def _weights(batch: TrainingBatch, term: LossTerm, like: Tensor) -> Tensor:
    """The two per-structure weights, spread over the rows of a term.

    The structure's own weight and its weight for this property. The second is
    what makes a missing value contribute nothing: it is zero where the
    structure carries no such property.
    """
    graph_weight = batch.graph["weight"]
    assert isinstance(graph_weight, Tensor)
    property_weight = batch.property_weights.get(term.name)
    combined = graph_weight.to(like.dtype)
    if property_weight is not None:
        combined = combined * property_weight.to(like.dtype)
    if term.per_atom:
        node = batch.graph["batch"]
        assert isinstance(node, Tensor)
        combined = combined[node]
    return combined.reshape(-1, *([1] * (like.dim() - 1)))


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
