"""The training loop, written out.

No framework, no callbacks, no hooks. What happens in an epoch is the body of
one function, so a power user can read it and reproduce it as a script, and so
that the order of the four things that interact, the optimizer step, the
average's update, the evaluation and the checkpoint, is visible rather than
distributed across a registry.

Three orderings are load-bearing and each is silent when wrong:

- the average is updated **after** the optimizer step, so it averages the
  weights the run actually took;
- the validation loss is measured **through** the averaged weights when there
  is an average, because those are the weights the checkpoint will hold;
- the second stage's switch happens **before** the epoch it applies to, so the
  epoch that changes the loss weights is trained with them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from mace_core.config.resolved import ResolvedConfig
from mace_core.config.training import InheritOptimizer, StageConfig
from mace_core.stages import BuiltModel, EpochRecord, TrainedModel
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from mace_torch.data import TrainingBatch
from mace_torch.serialization import (
    canonical_state,
    load_canonical_state,
    read_canonical_state,
)
from mace_torch.train.checkpoint import RunState, read_run_state, write_model
from mace_torch.train.checkpoint import write_run_state as _write_run_state
from mace_torch.train.ema import ExponentialMovingAverage
from mace_torch.train.loss import build_loss
from mace_torch.train.optimizers import build_optimizer, build_scheduler

__all__ = ["evaluate", "run_train_stage", "train_one_epoch"]

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[TrainingBatch],
    loss: torch.nn.Module,
    optimizer: Optimizer,
    *,
    device: str = "cpu",
    compute: tuple[str, ...] = ("forces",),
    clip_grad: float | None = None,
    ema: ExponentialMovingAverage | None = None,
) -> float:
    """One pass over the training batches. Returns the mean loss.

    ``training=True`` on the forward is what keeps the derivative's own graph
    alive, and without it a force term has no gradient at all: the loss goes
    down, the forces do not, and nothing says so.
    """
    model.train()
    total, seen = 0.0, 0
    for batch in batches:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.graph, compute=compute, training=True)
        value = loss(output, batch)
        value.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if ema is not None:
            ema.update()
        total += float(value.detach())
        seen += 1
    if seen == 0:
        raise ValueError(
            "the training loader yielded no batches. A run with nothing to "
            "step on finishes and reports an untrained model."
        )
    return total / seen


def evaluate(
    model: nn.Module,
    batches: Iterable[TrainingBatch],
    loss: torch.nn.Module,
    *,
    device: str = "cpu",
    compute: tuple[str, ...] = ("forces",),
) -> float:
    """The mean loss over a loader.

    No ``no_grad``: a force is a gradient, so switching it off would evaluate a
    model that cannot produce the quantity it is being scored on. The graph is
    dropped per batch instead, by not asking for a second derivative.
    """
    model.eval()
    total, seen = 0.0, 0
    for batch in batches:
        batch = batch.to(device)
        output = model(batch.graph, compute=compute, training=False)
        total += float(loss(output, batch).detach())
        seen += 1
    if seen == 0:
        raise ValueError("the validation loader yielded no batches.")
    return total / seen


def run_train_stage(
    config: ResolvedConfig,
    built: BuiltModel[nn.Module, DataLoader],
    *,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> TrainedModel[nn.Module]:
    """Train the model, and leave behind what the run produced.

    Args:
        config: The resolved configuration.
        built: The model and the data it was built from.
        device: Where to train.
        checkpoint_path: Where the best model is written. ``None`` writes
            nothing, which is what a smoke test wants.
        resume: Continue a run written at ``checkpoint_path`` rather than
            starting one.

    Returns:
        The trained model, with the averaged weights in place when the run
        averaged, since those are the ones the best checkpoint holds.
    """
    model = built.model.to(device)
    requested = _requested_names(built)
    loss = build_loss(built.outputs, config.loss)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training.scheduler)
    ema = (
        ExponentialMovingAverage(model.parameters(), config.training.ema.decay)
        if config.training.ema.enabled
        else None
    )

    state = RunState(epoch=0)
    best_state: dict[str, dict[str, Tensor]] | None = None
    if resume:
        if checkpoint_path is None:
            raise ValueError(
                "a resume was asked for and no checkpoint path was given, so "
                "there is nothing to resume from."
            )
        state = read_run_state(
            checkpoint_path, optimizer=optimizer, scheduler=scheduler, ema=ema
        )
        # The best epoch so far belongs to the run that wrote it. Taken from
        # its file, so a resumed run that never improves still ends on it.
        if Path(checkpoint_path).with_suffix(".safetensors").is_file():
            best_state = read_canonical_state(checkpoint_path)

    if config.training.dry_run:
        # Before any epoch and before any file: the point of the flag is to
        # prove the configuration builds a model, and a dry run that wrote one
        # would be indistinguishable from a run of length zero.
        return TrainedModel(model=model, metadata=built.metadata)

    # A resumed run's earlier epochs belong to the run that wrote them:
    # inventing records for them would put numbers in the history that were
    # never measured here.
    history: list[EpochRecord] = []
    best_loss = state.best_valid_loss
    best_epoch = state.best_epoch
    written: Path | None = None
    since_best = 0
    stage = ""

    schedule = config.schedule()
    epoch = state.epoch
    while epoch < config.training.max_num_epochs:
        # Before the epoch it applies to, so the epoch that changes the loss
        # weights is trained with them. The stage is looked up rather than
        # switched into, which is what lets a resume land mid-schedule.
        entering = _stage_at(schedule, epoch)
        if entering.name != stage:
            stage = entering.name
            loss, optimizer, scheduler = _enter_stage(
                entering, config, built, model, optimizer, scheduler
            )
        if epoch == entering.start_epoch:
            # A stage scores with its own weights, so its losses and the ones
            # before it are not on one scale. Its best is the best among its
            # own epochs, and the model the run ends on is the last stage's.
            best_loss, since_best = None, 0

        train_loss = train_one_epoch(
            model,
            built.data.train_loader,
            loss,
            optimizer,
            device=device,
            compute=requested.derivatives,
            clip_grad=config.training.clip_grad,
            ema=ema,
        )

        valid_loss: float | None = None
        if epoch % config.training.eval_interval == 0:
            context = ema.average_parameters() if ema is not None else nullcontext()
            with context:
                valid_loss = evaluate(
                    model,
                    built.data.valid_loader,
                    loss,
                    device=device,
                    compute=requested.derivatives,
                )
                if best_loss is None or valid_loss < best_loss:
                    best_loss, best_epoch, since_best = valid_loss, epoch, 0
                    # Inside the average's context, so what is kept is what
                    # was scored, which is also what the checkpoint holds.
                    best_state = _snapshot(model)
                    if checkpoint_path is not None:
                        written = write_model(checkpoint_path, model, built.metadata)
                else:
                    since_best += 1
        following = _following_epoch(
            schedule, epoch, since_best >= config.training.patience
        )
        if following is None:
            logger.info("Stopping after %d evaluations without improvement", since_best)
        elif following != epoch + 1:
            logger.info(
                "Stage %r stopped improving after %d evaluations; the run moves "
                "to the next stage at epoch %d",
                stage,
                since_best,
                following,
            )
        if valid_loss is not None and checkpoint_path is not None:
            # With the epoch the run continues at, so a resume after a move to
            # the next stage continues there.
            _write_run_state(
                checkpoint_path,
                RunState(following or epoch + 1, best_loss, best_epoch),
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
            )

        _step_schedule(scheduler, valid_loss)
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                stage=stage,
                evaluated_with_ema=valid_loss is not None and ema is not None,
            )
        )
        if following is None:
            break
        epoch = following

    if best_state is not None:
        # The run ends on its best epoch, as the frozen tree does by loading
        # its last checkpoint, which it writes only on an improvement. Ending
        # on the last epoch instead returns a model that is not the one on
        # disk, and every number reported after training describes the
        # wrong one of the two.
        load_canonical_state(model, best_state)
    elif ema is not None:
        # The run's model is the averaged one. Leaving the stepped weights in
        # place would return a model that differs from the checkpoint beside
        # it and from every number the run reported.
        with torch.no_grad():
            for parameter, shadow in zip(
                (p for p in model.parameters() if p.requires_grad),
                ema.shadow,
                strict=True,
            ):
                parameter.copy_(shadow)

    return TrainedModel(
        model=model,
        metadata=built.metadata,
        history=tuple(history),
        best_epoch=best_epoch,
        checkpoint_path=written,
    )


@dataclass(frozen=True)
class _Requested:
    """The names the loss scores and the derivatives the engine computes."""

    names: tuple[str, ...]
    per_atom: tuple[bool, ...]
    derivatives: tuple[str, ...]


def _requested_names(built: BuiltModel[nn.Module, DataLoader]) -> _Requested:
    """What was asked for, read off the built model rather than the config.

    The model is the authority here: it was built from the request, and reading
    the configuration again would let the two drift for a run that overrode
    something between the stages.
    """
    outputs = built.outputs
    names: list[str] = []
    per_atom: list[bool] = []
    for spec in outputs.observables:
        names.append(spec.name)
        per_atom.append(spec.per_atom)
        for request in spec.derivatives:
            name = spec.derivative_name(request.wrt)
            if name not in outputs.derivatives:
                continue
            names.append(name)
            per_atom.append(request.wrt == "pos")
    return _Requested(tuple(names), tuple(per_atom), outputs.derivatives)


def _stage_at(schedule: Sequence[StageConfig], epoch: int) -> StageConfig:
    """Which stage an epoch belongs to.

    Looked up rather than advanced through, so a run resumed at epoch fifty
    lands in the stage epoch fifty belongs to rather than in the first one.
    """
    current = schedule[0]
    for stage in schedule:
        if stage.start_epoch <= epoch:
            current = stage
    return current


def _following_epoch(
    schedule: Sequence[StageConfig], epoch: int, exhausted: bool
) -> int | None:
    """The epoch the run trains next, or ``None`` when it stops.

    A stage that has stopped improving hands over to the next one at the epoch
    that one starts, as the frozen tree moves to its second stage. Only the
    last stage's patience ends the run, since stopping in an earlier one would
    skip the stages the configuration asked for.
    """
    if not exhausted:
        return epoch + 1
    later = [stage.start_epoch for stage in schedule if stage.start_epoch > epoch]
    return min(later) if later else None


def _enter_stage(
    stage: StageConfig,
    config: ResolvedConfig,
    built: BuiltModel[nn.Module, DataLoader],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
) -> tuple[torch.nn.Module, Optimizer, LRScheduler | None]:
    """The loss, optimizer and schedule a stage runs with.

    Everything the stage does not name it keeps. An optimizer it does not name
    is the one already running, with its state: rebuilding it would throw away
    the moments, which is a different run from the one the configuration asks
    for.
    """
    loss = build_loss(built.outputs, config.loss, stage)
    if not isinstance(stage.optimizer, InheritOptimizer):
        optimizer = build_optimizer(
            model, config.training.model_copy(update={"optimizer": stage.optimizer})
        )
        scheduler = build_scheduler(
            optimizer, stage.scheduler or config.training.scheduler
        )
    elif stage.scheduler is not None:
        scheduler = build_scheduler(optimizer, stage.scheduler)
    if stage.lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = stage.lr
    return loss, optimizer, scheduler


def _snapshot(model: nn.Module) -> dict[str, dict[str, Tensor]]:
    """Every operator's canonical tensors, copied, so later steps leave them."""
    return {
        path: {name: value.detach().clone() for name, value in tensors.items()}
        for path, tensors in canonical_state(model).items()
    }


def _step_schedule(scheduler: LRScheduler | None, valid_loss: float | None) -> None:
    """Advance the schedule, with the plateau one given its metric.

    A plateau schedule stepped without a metric raises, and stepped with the
    training loss would react to a different curve than the one it is meant to
    watch. So it is stepped only on an epoch that evaluated.
    """
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        if valid_loss is not None:
            scheduler.step(valid_loss)
        return
    scheduler.step()
