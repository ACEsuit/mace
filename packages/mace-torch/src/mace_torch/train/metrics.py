"""What a run reports about itself, per head and per quantity.

The frozen tree accumulates these in an 815-line module whose accumulator names
one property at a time: seven quantities, each with its own state, its own
``if`` in ``update`` and its own block in ``compute``. Adding a property means
editing all three. Here a quantity is accumulated because it was declared, so
the module is the same size whatever is being fitted.

**A row counts only if it carries a label.** A structure with no forces enters
a batch with a forces weight of zero, and its reference row is whatever the
collate function put there, which is zeros. Averaging that row in reports an
error against a number nobody measured. So every accumulator masks on the same
weight the loss masks on, and it masks the *targets* as well as the residuals.

That is a deliberate difference from the frozen tree, and it moves numbers.
Measured on two structures where only the first carries labels:

===================  ========  =======
quantity             legacy    here
===================  ========  =======
``mae_e``               1.0      1.0
``mae_e_per_atom``      1.5      0.5
``rmse_e_per_atom``     1.803    0.5
``rel_mae_f`` (%)     100.0     50.0
===================  ========  =======

Two separate causes, both in ``mace/tools/train.py``. The per-atom deltas are
appended and then never passed to the filter, so only the totals are masked
(``:659-663``); and the relative denominators are appended before the filter
runs and are never masked at all (``:666-667``), so the target norm is diluted
by every row the model was not asked to fit. ``rmse_e_per_atom`` is the default
error table's energy column, so the first of the two is not an edge case.

**The reduction is over sufficient statistics where it can be.** A sum and a
count reduce across ranks exactly and in constant space. Only the 95th
percentile needs the values themselves, so only that one gathers, and it says
so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
from mace_core.observables import RequestedOutputs
from mace_core.outputs import MACEOutput
from torch import Tensor, nn

from mace_torch.data import TrainingBatch
from mace_torch.train.loss import atom_counts, predicted, row_weights

__all__ = [
    "MetricSpec",
    "RunningMetrics",
    "metric_specs",
]


@dataclass(frozen=True)
class MetricSpec:
    """One quantity the run reports errors for.

    Attributes:
        name: The requested name, which is also the key of its reference values
            and of its per-structure weight.
        per_atom: Whether its rows are atoms rather than structures.
        per_atom_variant: Whether a second set of errors is reported with the
            residual divided by the atom count. True for a quantity that is one
            value per structure and grows with the structure, which is the pair
            of properties that makes a raw error say more about the structure
            sizes than about the model.
    """

    name: str
    per_atom: bool
    per_atom_variant: bool


def metric_specs(requested: RequestedOutputs) -> tuple[MetricSpec, ...]:
    """A spec per requested quantity, with everything read off a declaration."""
    specs: list[MetricSpec] = []
    for observable in requested.observables:
        specs.append(
            MetricSpec(
                observable.name,
                observable.per_atom,
                observable.extensive and not observable.per_atom,
            )
        )
        for request in observable.derivatives:
            name = observable.derivative_name(request.wrt)
            if name not in requested.derivatives:
                continue
            per_atom = request.wrt == "pos"
            specs.append(MetricSpec(name, per_atom, request.extensive and not per_atom))
    return tuple(specs)


@dataclass
class _Samples:
    """The masked values of one quantity, batch by batch.

    Held rather than summed because the 95th percentile is not a sum. The
    sums could be kept alongside and are not: two accumulations of one
    quantity is how the mean and the percentile end up describing different
    sets of rows.
    """

    residual: list[Tensor] = field(default_factory=list)
    reference: list[Tensor] = field(default_factory=list)
    per_atom: list[Tensor] = field(default_factory=list)


class RunningMetrics:
    """Errors over a loader, accumulated batch by batch.

    Args:
        specs: The quantities to report, from :func:`metric_specs`.
        loss: The loss the run scores on. Accumulated here as well because the
            number a checkpoint is selected by has to come from the same pass
            as the errors beside it.

    The loss is reported as the sum over batches divided by the number of
    structures, which is the frozen tree's definition. The training loop's own
    per-epoch number is a mean over *batches*, and the two differ by the batch
    size. Both are stated where they are produced; neither is converted into
    the other.
    """

    def __init__(self, specs: Sequence[MetricSpec], loss: nn.Module) -> None:
        self.specs = tuple(specs)
        self.loss = loss
        self._samples = {spec.name: _Samples() for spec in self.specs}
        self._total_loss = 0.0
        self._structures = 0

    def update(self, output: MACEOutput[Tensor], batch: TrainingBatch) -> None:
        """Score one batch and keep what it contributed."""
        self._total_loss += float(self.loss(output, batch).detach())
        self._structures += int(batch.graph["num_graphs"])
        counts = atom_counts(batch)
        for spec in self.specs:
            value = predicted(output, spec.name).detach()
            reference = batch.targets[spec.name].to(value.dtype).reshape(value.shape)
            # Broadcast, because the weight is one number per row and the
            # mask has to name every component of it: a vector quantity is
            # labelled or not as a whole.
            weights = row_weights(batch, spec.name, spec.per_atom, value)
            keep = weights.expand_as(value) > 0
            samples = self._samples[spec.name]
            residual = reference - value
            samples.residual.append(residual[keep])
            samples.reference.append(reference[keep])
            if spec.per_atom_variant:
                divided = residual / counts.to(value.dtype).reshape(
                    -1, *([1] * (value.dim() - 1))
                )
                samples.per_atom.append(divided[keep])

    def compute(self) -> dict[str, float]:
        """Every error this pass measured, keyed by name.

        A quantity no structure in the pass carried a label for is left out
        entirely rather than reported as zero, since an error over nothing is
        not a small error.
        """
        results = {"loss": self._reduced_loss()}
        for spec in self.specs:
            samples = self._samples[spec.name]
            # The gather comes before the emptiness test, and it has to: the
            # test is then on the joined values, so every rank takes the same
            # branch. Testing the local ones first would have a rank that
            # happened to draw no labelled row skip the gathers the others are
            # waiting in.
            residual = _gather(_joined(samples.residual))
            reference = _gather(_joined(samples.reference))
            divided = _gather(_joined(samples.per_atom))
            if residual.numel() == 0:
                continue
            results.update(_errors(spec.name, residual, reference))
            if spec.per_atom_variant:
                results.update(_absolute_errors(f"{spec.name}_per_atom", divided))
        return results

    def _reduced_loss(self) -> float:
        """The loss per structure, summed across ranks before the division."""
        totals = torch.tensor([self._total_loss, float(self._structures)])
        if _distributed():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if totals[1] == 0:
            raise ValueError(
                "no structure was scored, so there is no loss to report. An "
                "empty loader reaches here as a run that reports a number."
            )
        return float(totals[0] / totals[1])


def _joined(batches: list[Tensor]) -> Tensor | None:
    """One flat tensor from a quantity's batches, empty when there were none."""
    return torch.cat([batch.reshape(-1) for batch in batches]) if batches else None


def _absolute_errors(name: str, residual: Tensor) -> dict[str, float]:
    """Mean, root-mean-square and 95th percentile of a residual."""
    delta = residual.to(torch.float64).cpu().numpy()
    return {
        f"mae_{name}": float(np.mean(np.abs(delta))),
        f"rmse_{name}": float(np.sqrt(np.mean(np.square(delta)))),
        f"q95_{name}": float(np.percentile(np.abs(delta), 95)),
    }


def _errors(name: str, residual: Tensor, reference: Tensor) -> dict[str, float]:
    """The three absolute errors of one quantity, and the two relative ones.

    The relative pair divides by the reference's own magnitude over the same
    rows. It is computed for every quantity and the tables print the ones that
    mean something: a relative error on a quantity carrying an additive offset,
    which an energy does, says more about the offset than about the model.

    There is no relative variant of the per-atom errors. Dividing a residual
    and its reference by the same atom count leaves the ratio where it was, so
    it would be the same number under a second name.
    """
    target = reference.to(torch.float64).cpu().numpy()
    errors = _absolute_errors(name, residual)
    return {
        **errors,
        f"rel_mae_{name}": errors[f"mae_{name}"]
        / (float(np.mean(np.abs(target))) + 1e-9)
        * 100,
        f"rel_rmse_{name}": errors[f"rmse_{name}"]
        / (float(np.sqrt(np.mean(np.square(target)))) + 1e-9)
        * 100,
    }


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _gather(values: Tensor | None) -> Tensor:
    """One rank's masked values, joined with every other rank's.

    The only place the values themselves cross ranks. A percentile is not a
    sum, so it cannot be reduced from statistics, and computing it per rank
    would report the percentile of a shard.

    The lengths differ per rank, because how many rows carry a label is a
    property of the data rather than of the split. So the counts go first and
    the payloads are padded to the longest.
    """
    if values is None:
        values = torch.zeros(0)
    values = values.reshape(-1)
    if not _distributed():
        return values
    world = dist.get_world_size()
    count = torch.tensor([values.numel()], device=values.device)
    counts = [torch.zeros_like(count) for _ in range(world)]
    dist.all_gather(counts, count)
    longest = max(int(entry.item()) for entry in counts)
    padded = torch.zeros(longest, device=values.device, dtype=values.dtype)
    padded[: values.numel()] = values
    buckets = [torch.zeros_like(padded) for _ in range(world)]
    dist.all_gather(buckets, padded)
    return torch.cat(
        [
            bucket[: int(entry.item())]
            for bucket, entry in zip(buckets, counts, strict=True)
        ]
    )
