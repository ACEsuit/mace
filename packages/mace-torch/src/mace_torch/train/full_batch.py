"""The full-batch step: one L-BFGS step per epoch, through a closure.

The closure is the whole training set's loss and its gradient, assembled a
batch at a time. Each batch contributes its terms' sums divided by the terms'
element counts over the whole set, so the parts add up to exactly the loss of
the concatenated set. The frozen tree weighs each batch by its share of the
structures instead (``mace/tools/train.py:523``), which is the same thing for
the energy and not for the forces: their term is a mean over atoms, so on a set
whose structures differ in size a batch of small ones counts as much as a batch
of large ones, and the regime minimises a different objective without failing
anything.

**Under several processes, rank zero drives.** L-BFGS evaluates the closure a
number of times that depends on the line search, so only rank zero runs the
optimizer. Every evaluation starts with rank zero telling the others to
evaluate and sending them its parameters; every rank then scores its own share
of the set, and the gradients and the loss are summed across ranks, so each
evaluation is the whole set's on every rank. When the step is done rank zero
says so, and sends its parameters once more. Each process holds a share that
is not padded, since a repeated structure would be counted twice.

The gradients are summed by hand rather than through the data-parallel
wrapper: the wrapper averages, which would hand L-BFGS a gradient a factor of
the world size smaller than the loss it is paired with, and it synchronises on
every backward, which ranks with different numbers of batches cannot all make.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim import Optimizer

from mace_torch.data import TrainingBatch
from mace_torch.train.ddp import DistributedContext, broadcast_decision
from mace_torch.train.loss import TermwiseLoss

__all__ = ["accumulate", "full_batch_step", "parameter_digest", "set_totals"]

logger = logging.getLogger(__name__)


def set_totals(
    loss: TermwiseLoss,
    batches: Iterable[TrainingBatch],
    processes: DistributedContext,
) -> dict[str, float]:
    """Each term's element count over the whole set, across every rank."""
    names = [term.name for term in loss.terms]
    counts = torch.zeros(len(names), dtype=torch.float64)
    for batch in batches:
        per_term = loss.element_counts(batch)
        counts += torch.tensor([per_term[name] for name in names], dtype=torch.float64)
    if processes.distributed:
        counts = counts.to(_collective_device(processes))
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    empty = [
        name for name, count in zip(names, counts.tolist(), strict=True) if not count
    ]
    if empty:
        raise ValueError(
            f"the training set has no elements for the terms {empty}, so their "
            f"mean over the set is undefined."
        )
    return dict(zip(names, counts.tolist(), strict=True))


def accumulate(
    model: nn.Module,
    batches: Iterable[TrainingBatch],
    loss: TermwiseLoss,
    totals: Mapping[str, float],
    *,
    device: str = "cpu",
    compute: tuple[str, ...] = ("forces",),
) -> Tensor:
    """The set's loss over these batches, its gradient added into ``.grad``.

    A batch at a time, so the set never has to fit in one graph: each batch's
    part is backpropagated as soon as it is scored and its graph freed. Called
    with the gradients zeroed, what is left in them is the gradient of the
    returned loss.
    """
    model.train()
    total: Tensor | None = None
    for batch in batches:
        batch = batch.to(device)
        output = model(batch.graph, compute=compute, training=True)
        value = loss(output, batch, totals)
        value.backward()
        total = value.detach() if total is None else total + value.detach()
    if total is None:
        # A rank whose share of a small set is empty still takes part in
        # every sum, with nothing to add.
        parameter = next(model.parameters())
        return torch.zeros((), dtype=parameter.dtype, device=device)
    return total


def full_batch_step(
    model: nn.Module,
    batches: Callable[[], Iterable[TrainingBatch]],
    loss: torch.nn.Module,
    optimizer: Optimizer,
    *,
    device: str = "cpu",
    compute: tuple[str, ...] = ("forces",),
    clip_grad: float | None = None,
    processes: DistributedContext | None = None,
) -> float:
    """One optimizer step over the whole training set.

    Args:
        model: The model itself, not a data-parallel wrapper around it.
        batches: A fresh pass over this process's share of the set, each time
            it is called, with the ragged tail kept.
        loss: A loss built on the shared terms, which is what can divide each
            term by its count over the whole set.
        optimizer: The optimizer, stepped once through a closure.
        device: Where the batches go.
        compute: The derivatives the loss scores.
        clip_grad: The norm the summed gradient is clipped to, once per
            evaluation, after it is complete.
        processes: Where this process sits among the others.

    Returns:
        The set's loss at the parameters the step started from.

    Raises:
        TypeError: If the loss is not built on the shared terms.
    """
    if not isinstance(loss, TermwiseLoss):
        raise TypeError(
            f"a full-batch stage needs a loss built on TermwiseLoss, which can "
            f"divide each term by its count over the whole set, and "
            f"{type(loss).__name__} is not one."
        )
    processes = processes or DistributedContext(device=device)
    parameters: list[Tensor] = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    totals = set_totals(loss, batches(), processes)
    evaluations = 0

    def closure() -> Tensor:
        nonlocal evaluations
        if processes.distributed:
            if processes.is_main:
                broadcast_decision(True, processes)
            _broadcast(parameters)
        optimizer.zero_grad(set_to_none=True)
        total = accumulate(
            model, batches(), loss, totals, device=device, compute=compute
        )
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        if processes.distributed:
            _sum(parameters, total)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        evaluations += 1
        logger.debug(
            "Rank %d closure evaluation %d: loss %r, parameters %s",
            processes.rank,
            evaluations,
            float(total),
            parameter_digest(parameters),
        )
        return total

    # Typed by torch as returning a float; L-BFGS reads the loss as a tensor
    # and hands back the first evaluation's.
    stepped = cast(Callable[[], float], closure)
    if not processes.distributed:
        return float(optimizer.step(stepped))
    started = torch.zeros((), dtype=parameters[0].dtype)
    if processes.is_main:
        started.fill_(float(optimizer.step(stepped)))
        broadcast_decision(False, processes)
    else:
        while broadcast_decision(False, processes):
            closure()
    _broadcast(parameters)
    started = started.to(_collective_device(processes))
    dist.broadcast(started, src=0)
    return float(started)


def parameter_digest(parameters: Iterable[Tensor]) -> str:
    """A short fingerprint of the exact parameter values, for comparing ranks."""
    digest = hashlib.blake2b(digest_size=8)
    for parameter in parameters:
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _collective_device(processes: DistributedContext) -> str:
    """Where a tensor has to be for the backend to reduce it."""
    return "cpu" if processes.device.split(":", 1)[0] == "cpu" else processes.device


def _broadcast(parameters: Iterable[Tensor]) -> None:
    """Rank zero's parameters, on every rank."""
    with torch.no_grad():
        for parameter in parameters:
            dist.broadcast(parameter.data, src=0)


def _sum(parameters: Iterable[Tensor], total: Tensor) -> None:
    """Every rank's gradients and loss, added up, on every rank."""
    for parameter in parameters:
        assert parameter.grad is not None
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
