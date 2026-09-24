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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from mace_core.config.loss import LossConfig, RegisteredLoss
from mace_core.config.training import StageConfig
from mace_core.observables import RequestedOutputs
from mace_core.outputs import MACEOutput
from torch import Tensor

from mace_torch.data import TrainingBatch

__all__ = [
    "LOSS_REGISTRY",
    "GeneratedLoss",
    "HuberLoss",
    "L1L2Loss",
    "LossTerm",
    "TermwiseLoss",
    "UniversalLoss",
    "UnknownLossError",
    "atom_counts",
    "build_loss",
    "predicted",
    "reduce_loss",
    "register_loss",
    "row_weights",
    "terms_for",
]

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
    requested: RequestedOutputs,
    config: LossConfig,
    stage: StageConfig | None = None,
) -> tuple[LossTerm, ...]:
    """A term per requested output, with everything read off a declaration.

    A stage's own weights replace the run's for the quantities it names, and
    only those: a stage that raises the energy weight says only that.
    """
    overrides = {} if stage is None else stage.loss_weights

    def weight_of(name: str) -> float:
        return overrides.get(name, config.weights.get(name, 1.0))

    terms: list[LossTerm] = []
    for observable in requested.observables:
        terms.append(
            LossTerm(
                observable.name,
                weight_of(observable.name),
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
                    weight_of(name),
                    requested.per_atom_derivative(request.wrt),
                    request.extensive,
                )
            )
    return tuple(terms)


class TermwiseLoss(torch.nn.Module):
    """The weighted sum of one term per requested quantity.

    What every loss here shares: which quantities are scored, how a residual is
    normalised, and how the two per-structure weights reach it. What a subclass
    chooses is the one thing that actually differs between the frozen tree's
    ten classes, which is how a residual becomes a number.
    """

    def __init__(self, terms: Sequence[LossTerm]) -> None:
        super().__init__()
        if not terms:
            raise ValueError(
                "a loss with no terms scores nothing. The terms come from the "
                "requested outputs, so an empty one means none were requested."
            )
        self.terms = tuple(terms)

    def elementwise(self, residual: Tensor, term: LossTerm) -> Tensor:
        """The per-element cost of a residual. Squared error by default."""
        return residual.square()

    def element_counts(self, batch: TrainingBatch) -> dict[str, int]:
        """How many elements each term averages over in this batch.

        Read off the cost of a zero residual shaped like the reference, so a
        loss whose cost has another shape than its residual, as a vector norm
        over a force does, is counted by the elements it actually averages.
        """
        counts: dict[str, int] = {}
        for term in self.terms:
            reference = batch.targets[term.name]
            counts[term.name] = self.elementwise(
                torch.zeros_like(reference), term
            ).numel()
        return counts

    def forward(
        self,
        output: MACEOutput[Tensor],
        batch: TrainingBatch,
        totals: Mapping[str, float] | None = None,
    ) -> Tensor:
        """The total, as a scalar.

        Args:
            output: What the model produced for the batch.
            batch: The batch, with its references and weights.
            totals: Each term's element count over a whole set this batch is
                one part of. Each term is then its sum over this batch divided
                by that count, and not the batch's mean, so the parts of a set
                add up to exactly the set's loss, whatever the sizes of its
                structures. Nothing is reduced across ranks in this mode: the
                caller adds the parts up.
        """
        counts = atom_counts(batch)
        total: Tensor | None = None
        for term in self.terms:
            value = predicted(output, term.name)
            reference = batch.targets[term.name].to(value.dtype)
            residual = value - reference.reshape(value.shape)
            if term.extensive:
                # Per atom, so a structure twice the size is not twice the
                # error. The division is on the residual and not on the
                # squared term, which is the frozen tree's and squares the
                # normalisation with it.
                residual = residual / _broadcast(counts, residual)
            cost = self.elementwise(residual, term)
            weights = row_weights(batch, term.name, term.per_atom, cost)
            if totals is None:
                reduced = reduce_loss(weights * cost)
            else:
                reduced = (weights * cost).sum() / totals[term.name]
            total_term = term.weight * reduced
            total = total_term if total is None else total + total_term
        assert total is not None
        return total


class GeneratedLoss(TermwiseLoss):
    """Squared error, which is what every weighted legacy loss reduces to."""


def build_loss(
    requested: RequestedOutputs,
    config: LossConfig,
    stage: StageConfig | None = None,
) -> torch.nn.Module:
    """The loss a run scores on: a registered one, or the generated one.

    Raises:
        UnknownLossError: Naming the value and listing what is registered.
    """
    terms = terms_for(requested, config, stage)
    if isinstance(config.kind, RegisteredLoss):
        # The escape from the closed union into the open registry. The kinds
        # the schema names are the ones whose settings it can validate; a loss
        # from another package cannot be, so its settings travel as written and
        # it validates them itself.
        return _build(config.kind.name, terms, dict(config.kind.settings))
    if config.kind.kind == "weighted":
        return GeneratedLoss(terms)
    settings = {
        name: value
        for name, value in config.kind.model_dump().items()
        if name != "kind"
    }
    return _build(config.kind.kind, terms, settings)


def _build(
    kind: str, terms: Sequence[LossTerm], settings: dict[str, Any]
) -> torch.nn.Module:
    """One registered loss, built.

    Raises:
        UnknownLossError: Naming the value and listing what is registered.
    """
    if kind not in LOSS_REGISTRY:
        raise UnknownLossError(
            f"{kind!r} is not a registered loss. The registered names are "
            f"{sorted(LOSS_REGISTRY)}, and 'weighted' is the generated one. "
            f"Register yours with @register_loss."
        )
    loss = LOSS_REGISTRY[kind]
    # A loss built on the shared term machinery is handed the terms; one
    # written from scratch is handed only what it asked for. Which of the two
    # it is, is a question about its own type rather than about its name.
    if isinstance(loss, type) and issubclass(loss, TermwiseLoss):
        return loss(terms, **settings)
    return loss(**settings)


def atom_counts(batch: TrainingBatch) -> Tensor:
    """How many atoms each structure has, from `ptr` and never from `batch`."""
    pointer = batch.graph["ptr"]
    assert isinstance(pointer, Tensor)
    return pointer[1:] - pointer[:-1]


def _broadcast(per_graph: Tensor, like: Tensor) -> Tensor:
    """A per-structure quantity shaped to divide or multiply ``like``."""
    value = per_graph.to(like.dtype)
    return value.reshape(-1, *([1] * (like.dim() - 1)))


def row_weights(
    batch: TrainingBatch, name: str, per_atom: bool, like: Tensor
) -> Tensor:
    """The two per-structure weights, spread over the rows of a quantity.

    The structure's own weight and its weight for this property. The second is
    what makes a missing value contribute nothing: it is zero where the
    structure carries no such property.
    """
    graph_weight = batch.graph["weight"]
    assert isinstance(graph_weight, Tensor)
    property_weight = batch.property_weights.get(name)
    combined = graph_weight.to(like.dtype)
    if property_weight is not None:
        combined = combined * property_weight.to(like.dtype)
    if per_atom:
        node = batch.graph["batch"]
        assert isinstance(node, Tensor)
        combined = combined[node]
    return combined.reshape(-1, *([1] * (like.dim() - 1)))


def predicted(output: MACEOutput[Tensor], name: str) -> Tensor:
    """One quantity off the typed output, by its requested name.

    Through the output's own accessor rather than a table here. A second table
    of which names are core fields is the thing that drifts: it would agree
    until someone adds a field, and then a loss term would read an empty
    `extras` entry for a quantity the model did produce.
    """
    value = output.get(name)
    if value is None:
        raise KeyError(
            f"the model produced no {name!r}, and the loss has a term for it. "
            f"It carries {sorted(output.extras)} beyond its core fields."
        )
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name!r} came back as {type(value)!r} rather than a tensor.")
    return value


@register_loss("huber")
class HuberLoss(TermwiseLoss):
    """Huber, term by term, on the same normalised residuals.

    Quadratic below ``delta`` and linear above it, so a single bad label costs
    what it is worth rather than what its square is worth. The frozen tree has
    one class per property combination that uses it; here it is the reduction
    and the quantities are whatever was declared.
    """

    def __init__(self, terms: Sequence[LossTerm], delta: float = 0.01) -> None:
        super().__init__(terms)
        self.delta = delta

    def elementwise(self, residual: Tensor, term: LossTerm) -> Tensor:
        return _huber(residual, self.delta)


#: The term the universal loss bands by its reference norm.
_BANDED_TERM = "forces"


@register_loss("universal")
class UniversalLoss(TermwiseLoss):
    """Huber whose crossover falls where the reference force is large.

    The foundation-model recipe. A structure with forces of several hundred eV
    per Angstrom is either a very repulsive geometry or a broken calculation,
    and both deserve less of the model's attention than a near-equilibrium one:
    the crossover drops in bands, so a large reference force is scored almost
    linearly.

    The banding reads the **reference** force, not the predicted one, so what a
    structure costs does not move while the model learns.
    """

    #: The multipliers on ``delta``, for reference force norms under 100, under
    #: 200, under 300 and above. The frozen tree's numbers.
    BANDS = (1.0, 0.7, 0.4, 0.1)
    EDGES = (100.0, 200.0, 300.0)

    def __init__(self, terms: Sequence[LossTerm], delta: float = 0.01) -> None:
        super().__init__(terms)
        self.delta = delta
        self._reference: dict[str, Tensor] = {}

    def forward(
        self,
        output: MACEOutput[Tensor],
        batch: TrainingBatch,
        totals: Mapping[str, float] | None = None,
    ) -> Tensor:
        # The band is a property of the reference, so it is read here and used
        # by `elementwise`, which only sees the residual.
        self._reference = dict(batch.targets)
        try:
            return super().forward(output, batch, totals)
        finally:
            self._reference = {}

    def elementwise(self, residual: Tensor, term: LossTerm) -> Tensor:
        reference = self._reference.get(term.name)
        # The force term alone is banded. Another per-atom vector, the
        # magnetic forces, is scored with the plain crossover, as the frozen
        # tree scores it: a large magnetic force is not a broken calculation.
        if reference is None or term.name != _BANDED_TERM or reference.dim() < 2:
            return _huber(residual, self.delta)
        norms = torch.linalg.vector_norm(reference.to(residual.dtype), dim=-1)
        # From the widest band inwards, so the narrowest one that matches is
        # the one written last. Going the other way leaves every row holding
        # the last band it fell into rather than the first.
        delta = torch.full_like(norms, self.delta * self.BANDS[-1])
        for edge, band in zip(
            reversed(self.EDGES), reversed(self.BANDS[:-1]), strict=True
        ):
            delta = torch.where(norms < edge, self.delta * band, delta)
        return _huber(residual, delta.unsqueeze(-1))


@register_loss("l1l2")
class L1L2Loss(TermwiseLoss):
    """Absolute error on the scalars, vector norm on the per-atom quantities.

    Neither term is a squared error, which is what makes this a registered
    loss rather than a generated one: an energy costs its absolute deviation
    and a force costs the length of its error vector, so a force wrong by
    ``(3, 4, 0)`` costs five rather than twenty-five.
    """

    def elementwise(self, residual: Tensor, term: LossTerm) -> Tensor:
        if term.per_atom and residual.dim() > 1:
            return torch.linalg.vector_norm(residual, dim=-1)
        return residual.abs()


def _huber(residual: Tensor, delta: Tensor | float) -> Tensor:
    """Huber of a residual, elementwise, with a per-element crossover allowed.

    Written out rather than taken from `torch.nn.functional.huber_loss`, which
    takes one scalar delta: the banded variant needs a different crossover per
    row, and two spellings of the same formula is how the two drift.
    """
    magnitude = residual.abs()
    crossover = torch.as_tensor(delta, dtype=residual.dtype, device=residual.device)
    quadratic = 0.5 * residual.square()
    linear = crossover * (magnitude - 0.5 * crossover)
    return torch.where(magnitude <= crossover, quadratic, linear)
