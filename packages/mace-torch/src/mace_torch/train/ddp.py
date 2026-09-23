"""Distributed data parallel: which process this is, and how the processes meet.

Four launchers, as legacy has them. ``torchrun`` exports the process's rank,
the world size and its local rank; ``slurm`` is read from SLURM's variables by
:mod:`mace_torch.train.slurm`; ``mpi`` from OpenMPI's, or from Intel MPI's and
PALS's; ``none`` is a single process.

The process-group backend follows the device: NCCL for CUDA, XCCL for XPU and
Gloo for the CPU, which is what makes a distributed run on a laptop possible at
all. So does the rule for ``device_ids``: it names the one accelerator a
process drives, so CUDA passes the local rank, XPU passes the visible device
that rank maps to, and the CPU passes nothing.

**Rank zero decides and writes.** Checkpoints are written by rank zero alone,
into a run directory every rank can read, and a barrier separates the write
from anything that reads it back. Decisions that end or change the run are
taken on rank zero and broadcast, so the processes never disagree about which
epoch comes next.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
from torch import nn

from mace_torch.train.slurm import slurm_environment

__all__ = [
    "LAUNCHERS",
    "DistributedContext",
    "broadcast_decision",
    "init_distributed",
    "process_group_backend",
    "wrap_model",
    "xpu_device_index",
]

Launcher = Literal["slurm", "torchrun", "mpi", "none"]

#: The launchers a distributed run can name.
LAUNCHERS: tuple[str, ...] = ("slurm", "torchrun", "mpi", "none")


@dataclass(frozen=True)
class DistributedContext:
    """Where this process sits among the others.

    Attributes:
        rank: Its index among all processes.
        world_size: How many there are. One for a run that is not distributed.
        local_rank: Its index on its node, which picks its accelerator.
        device: The device this process trains on.
    """

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cpu"

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Whether this process writes and decides."""
        return self.rank == 0

    def barrier(self) -> None:
        """Wait for every process, when there is more than one."""
        if self.distributed:
            dist.barrier()


def xpu_device_index(local_rank: int) -> int:
    """The visible XPU a local rank drives.

    With one tile visible per rank, the local rank can exceed the visible
    count, and torch refuses such an index with an overflow message that names
    neither the rank nor the device.
    """
    try:
        visible = torch.xpu.device_count()
    except (AttributeError, RuntimeError):
        return 0
    return local_rank if local_rank < visible else 0


def process_group_backend(device: str) -> str:
    """The collective backend for a device kind."""
    kind = device.split(":", 1)[0]
    return {"cuda": "nccl", "xpu": "xccl"}.get(kind, "gloo")


def _ranks(launcher: str, environ: Mapping[str, str]) -> tuple[int, int, int]:
    if launcher == "slurm":
        place = slurm_environment(environ)
        os.environ.update(place.exported())
        return place.rank, place.local_rank, place.world_size
    if launcher == "torchrun":
        return (
            int(environ["RANK"]),
            int(environ["LOCAL_RANK"]),
            int(environ["WORLD_SIZE"]),
        )
    if launcher == "mpi":
        if "OMPI_COMM_WORLD_RANK" in environ:
            rank = int(environ["OMPI_COMM_WORLD_RANK"])
            world_size = int(environ["OMPI_COMM_WORLD_SIZE"])
            local_size = int(environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", "1"))
            local_rank = rank % local_size
        else:
            rank = int(environ.get("PMI_RANK", environ.get("PALS_RANKID", "0")))
            # PALS_LOCAL_SIZE is the size of one node, not of the world.
            world_size = int(
                environ.get(
                    "PMI_SIZE",
                    environ.get("PALS_NTASKS", environ.get("WORLD_SIZE", "1")),
                )
            )
            local_rank = int(
                environ.get(
                    "PALS_LOCAL_RANKID", environ.get("MPI_LOCALRANKID", str(rank))
                )
            )
        os.environ.update(
            {
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "LOCAL_RANK": str(local_rank),
            }
        )
        os.environ.setdefault("MASTER_PORT", "33333")
        return rank, local_rank, world_size
    raise ValueError(
        f"{launcher!r} is not a launcher. The launchers are {list(LAUNCHERS)}."
    )


def init_distributed(
    distributed: bool,
    launcher: str | None,
    device: str,
    environ: Mapping[str, str] | None = None,
) -> DistributedContext:
    """Join the process group, or say this is a single process.

    Args:
        distributed: Whether the run asked to be distributed.
        launcher: How the processes were started.
        device: The device kind the run trains on, ``cpu``, ``cuda`` or
            ``xpu``. A process on an accelerator is given the one its local
            rank picks.
        environ: The environment to read. The process's own by default.

    Raises:
        ValueError: If a distributed run names no launcher, or one that does
            not exist.
        KeyError: If the launcher's variables are not set.
    """
    if not distributed or launcher == "none":
        return DistributedContext(device=device)
    if launcher is None:
        raise ValueError(
            "the run is distributed and names no launcher, so there is no way "
            f"to tell which process this is. Set one of {list(LAUNCHERS)}."
        )
    rank, local_rank, world_size = _ranks(launcher, environ or os.environ)
    kind = device.split(":", 1)[0]
    placed = device
    if kind == "cuda":
        torch.cuda.set_device(local_rank)
        placed = f"cuda:{local_rank}"
    elif kind == "xpu":
        placed = f"xpu:{xpu_device_index(local_rank)}"
    if not dist.is_initialized():
        dist.init_process_group(
            backend=process_group_backend(kind),
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    return DistributedContext(
        rank=rank, world_size=world_size, local_rank=local_rank, device=placed
    )


def wrap_model(model: nn.Module, context: DistributedContext) -> nn.Module:
    """The model wrapped so its gradients are averaged across processes.

    ``device_ids`` names the one accelerator the module lives on, so it is
    given for CUDA and XPU and left out for the CPU, where Gloo refuses it.
    """
    if not context.distributed:
        return model
    kind = context.device.split(":", 1)[0]
    if kind == "cuda":
        return nn.parallel.DistributedDataParallel(
            model, device_ids=[context.local_rank]
        )
    if kind == "xpu":
        return nn.parallel.DistributedDataParallel(
            model, device_ids=[xpu_device_index(context.local_rank)]
        )
    return nn.parallel.DistributedDataParallel(model)


def broadcast_decision(decision: bool, context: DistributedContext) -> bool:
    """Rank zero's decision, on every rank.

    Every rank computes the decision from the same reduced numbers, and they
    agree in exact arithmetic. Taking rank zero's anyway is what guarantees
    they agree in floating point too, so no rank trains an epoch the others
    skipped.
    """
    if not context.distributed:
        return decision
    flag = torch.tensor(
        [1 if decision else 0],
        dtype=torch.int64,
        device=context.device
        if process_group_backend(context.device) != "gloo"
        else "cpu",
    )
    dist.broadcast(flag, src=0)
    return bool(flag.item())
