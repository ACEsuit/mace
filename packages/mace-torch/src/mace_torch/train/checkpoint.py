"""Writing a run down so it can be picked up again.

Two artifacts, and they are different things. The **model** is the canonical
weights and the resolved configuration, written by
:mod:`mace_torch.serialization`: no pickle, no module tree, readable by a
different backend. The **run** is the optimizer's state, the schedule's, the
average's and the epoch counter, which mean nothing outside the process that
made them and exist only so an interrupted run continues rather than restarts.

The run half is written with ``torch.save`` for now. The format that replaces
it is the distributed-training ticket's, and stating that here is cheaper than
inventing a second one that then has to be migrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from mace_core.metadata import ModelMetadata
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from mace_torch.serialization import save_checkpoint
from mace_torch.train.ema import ExponentialMovingAverage

__all__ = ["RunState", "read_run_state", "write_model", "write_run_state"]

#: What the run half is called beside the model's two files.
RUN_SUFFIX = ".run.pt"


@dataclass(frozen=True)
class RunState:
    """Where a run had got to.

    Attributes:
        epoch: The next epoch to run, so a resume starts here rather than
            repeating the one that was written.
        best_valid_loss: The lowest validation loss seen, or ``None`` if the
            run had not evaluated yet.
        best_epoch: Which epoch that was.
    """

    epoch: int
    best_valid_loss: float | None = None
    best_epoch: int | None = None


def write_model(path: str | Path, model: nn.Module, metadata: ModelMetadata) -> Path:
    """The weights and the record, as one checkpoint.

    The whole resolved configuration goes in the sidecar, because a checkpoint
    that carried only weights would need its run's command line to be rebuilt
    and that is the thing least likely to still exist.
    """
    return save_checkpoint(path, model, metadata.model_dump(mode="json"))


def write_run_state(
    path: str | Path,
    state: RunState,
    *,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    ema: ExponentialMovingAverage | None = None,
) -> Path:
    """The optimizer, the schedule, the average and the counter."""
    target = Path(path).with_suffix("").with_suffix(RUN_SUFFIX)
    payload: dict[str, Any] = {
        "epoch": state.epoch,
        "best_valid_loss": state.best_valid_loss,
        "best_epoch": state.best_epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
    }
    torch.save(payload, target)
    return target


def read_run_state(
    path: str | Path,
    *,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    ema: ExponentialMovingAverage | None = None,
) -> RunState:
    """Put a run's state back and say where it had got to.

    The average is restored when the run had one and refused when it does not:
    resuming a run that averaged its weights without the average would continue
    from parameters no evaluation ever saw.

    Raises:
        FileNotFoundError: Naming the file, since a resume that silently
            started from scratch would report a first epoch as if it were the
            hundredth.
        ValueError: If the saved run had an average and this one has none, or
            the other way round.
    """
    target = Path(path).with_suffix("").with_suffix(RUN_SUFFIX)
    if not target.is_file():
        raise FileNotFoundError(
            f"{target} does not exist, so there is no run to continue. "
            f"Starting from scratch instead would report the first epoch as if "
            f"it carried on from somewhere."
        )
    payload = torch.load(target, weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload["scheduler"] is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if (ema is None) != (payload["ema"] is None):
        raise ValueError(
            "the saved run and this one disagree about whether the weights "
            "are averaged. Resuming either way continues from parameters the "
            "other half of the run never used."
        )
    if ema is not None:
        ema.load_state_dict(payload["ema"])
    return RunState(
        epoch=payload["epoch"],
        best_valid_loss=payload["best_valid_loss"],
        best_epoch=payload["best_epoch"],
    )
