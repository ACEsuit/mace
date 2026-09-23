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
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from mace_core.config.resolved import ResolvedConfig
from mace_core.config.training import (
    InheritOptimizer,
    LBFGSOptimizer,
    StageConfig,
)
from mace_core.stages import EpochRecord, TrainedModel
from mace_core.tables import Row, error_table
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from mace_torch.data import TrainingBatch
from mace_torch.finetune.freeze import freeze
from mace_torch.finetune.lora import inject_lora, merge_lora
from mace_torch.serialization import (
    CheckpointError,
    canonical_state,
    load_canonical_state,
    read_canonical_state,
)
from mace_torch.train.checkpoint import (
    RunState,
    latest_run_checkpoint,
    read_run_checkpoint,
    read_run_state,
    retain_run_checkpoints,
    write_model,
    write_run_checkpoint,
)
from mace_torch.train.contracts import TorchBuiltModel
from mace_torch.train.ema import ExponentialMovingAverage
from mace_torch.train.loss import build_loss
from mace_torch.train.metrics import MetricSpec, RunningMetrics, metric_specs
from mace_torch.train.optimizers import build_optimizer, build_scheduler
from mace_torch.train.tracking import Tracker, epoch_values, open_tracker

logger = logging.getLogger(__name__)

__all__ = [
    "evaluate",
    "evaluate_heads",
    "log_validation",
    "report_errors",
    "run_train_stage",
    "selection_loss",
    "train_one_epoch",
]


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
    specs: Sequence[MetricSpec],
    *,
    device: str = "cpu",
    compute: tuple[str, ...] = ("forces",),
) -> dict[str, float]:
    """Every error one loader measures, including the loss.

    No ``no_grad``: a force is a gradient, so switching it off would evaluate a
    model that cannot produce the quantity it is being scored on. The graph is
    dropped per batch instead, by not asking for a second derivative.
    """
    model.eval()
    metrics = RunningMetrics(specs, loss)
    seen = 0
    for batch in batches:
        batch = batch.to(device)
        output = model(batch.graph, compute=compute, training=False)
        metrics.update(output, batch)
        seen += 1
    if seen == 0:
        raise ValueError("the validation loader yielded no batches.")
    return metrics.compute()


def evaluate_heads(
    model: nn.Module,
    loaders: Mapping[str, Iterable[TrainingBatch]],
    loss: torch.nn.Module,
    specs: Sequence[MetricSpec],
    *,
    device: str = "cpu",
    compute: tuple[str, ...] = ("forces",),
) -> dict[str, dict[str, float]]:
    """One evaluation per head, keyed by head name.

    Kept apart rather than averaged into one number, because a multi-head run
    whose heads are reported as one row cannot say which head got worse.
    """
    return {
        head: evaluate(model, batches, loss, specs, device=device, compute=compute)
        for head, batches in loaders.items()
    }


def selection_loss(per_head: Mapping[str, dict[str, float]], rule: str) -> float:
    """The one number a checkpoint is chosen by, out of the per-head losses.

    The frozen tree uses the **last** head's loss and nothing else
    (``mace/tools/train.py:214``, with the comment saying so). That is a choice
    rather than an oversight, and it is a strange one: which head is last is
    the order the configuration happens to list them in. So it stays reachable
    and it is not the default.

    ``mean_over_heads`` averages them unweighted. Weighting by structure count
    would give the largest head the checkpoint, which is the thing the
    balancing above exists to stop.

    Raises:
        ValueError: On a rule nobody implements, naming the two.
    """
    losses = [metrics["loss"] for metrics in per_head.values()]
    if rule == "mean_over_heads":
        return sum(losses) / len(losses)
    if rule == "last_head":
        return losses[-1]
    raise ValueError(
        f"{rule!r} is not a checkpoint selection rule. They are "
        f"'mean_over_heads', which averages the heads, and 'last_head', which "
        f"is the frozen tree's and depends on the order the heads are listed."
    )


def run_train_stage(
    config: ResolvedConfig,
    built: TorchBuiltModel,
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
        checkpoint_path: Where the best model is written, and the name the
            run checkpoints beside it take. ``None`` writes nothing, which is
            what a smoke test wants.
        resume: Continue from the newest run checkpoint written for
            ``checkpoint_path`` rather than starting a run.

    Returns:
        The trained model, with the averaged weights in place when the run
        averaged, since those are the ones the best checkpoint holds.
    """
    model = built.model.to(device)
    # Before the optimizer and the average are built, since both are built
    # over what trains, and adapting or freezing changes that.
    config = _prepare_finetune(config, model)
    requested = _requested_names(built)
    specs = metric_specs(built.outputs)
    loss = build_loss(built.outputs, config.loss)
    tracker = open_tracker(config)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training.scheduler)
    ema = (
        ExponentialMovingAverage(model.parameters(), config.training.ema.decay)
        if config.training.ema.enabled
        else None
    )

    state = RunState(epoch=0)
    best_state: dict[str, dict[str, Tensor]] | None = None
    schedule = config.schedule()
    stage = ""
    if resume:
        if checkpoint_path is None:
            raise ValueError(
                "a resume was asked for and no checkpoint path was given, so "
                "there is nothing to resume from."
            )
        directory, name = Path(checkpoint_path).parent, Path(checkpoint_path).name
        latest = latest_run_checkpoint(directory, name)
        if latest is None:
            raise FileNotFoundError(
                f"no run checkpoint for {name!r} in {directory}, so there is no "
                f"run to continue. Starting from scratch instead would report "
                f"the first epoch as if it carried on from somewhere."
            )
        state = read_run_state(latest)
        # Every stage up to the one the checkpoint trained in, entered in
        # order, as the run that wrote it entered them: a stage that names its
        # own optimizer builds a new one, and restoring into the first stage's
        # optimizer would have that replace what was restored.
        for entered in schedule:
            if stage == state.stage:
                break
            stage = entered.name
            loss, optimizer, scheduler = _enter_stage(
                entered, config, built, model, optimizer, scheduler
            )
        if stage != state.stage:
            raise CheckpointError(
                f"{latest} was written in the stage {state.stage!r}, and this "
                f"configuration's stages are {[s.name for s in schedule]}. "
                f"Resuming in another stage would train with a schedule the "
                f"run never had."
            )
        resumed = read_run_checkpoint(
            latest, model=model, optimizer=optimizer, scheduler=scheduler, ema=ema
        )
        logger.info("Resuming from %s at epoch %d", latest, state.epoch)
        if resumed.optimizer_state == "reinitialized":
            logger.warning("Resumed with a new optimizer: %s", resumed.reason)
        # The best epoch so far belongs to the run that wrote it. Taken from
        # its model file, so a resumed run that never improves still ends on
        # it.
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
    since_best = state.since_best

    for epoch in range(state.epoch, config.training.max_num_epochs):
        # Before the epoch it applies to, so the epoch that changes the loss
        # weights is trained with them. The stage is looked up rather than
        # switched into, which is what lets a resume land mid-schedule.
        entering = _stage_at(schedule, epoch)
        if entering.name != stage:
            stage = entering.name
            loss, optimizer, scheduler = _enter_stage(
                entering, config, built, model, optimizer, scheduler
            )

        train_loss = train_one_epoch(
            model,
            built.data.train_loader.batches(epoch, drop_last=_drops_tail(entering)),
            loss,
            optimizer,
            device=device,
            compute=requested.derivatives,
            clip_grad=config.training.clip_grad,
            ema=ema,
        )

        valid_loss: float | None = None
        improved = False
        if epoch % config.training.eval_interval == 0:
            context = ema.average_parameters() if ema is not None else nullcontext()
            with context:
                per_head = evaluate_heads(
                    model,
                    built.data.valid_loaders,
                    loss,
                    specs,
                    device=device,
                    compute=requested.derivatives,
                )
                log_validation(epoch, per_head)
                tracker.log(epoch_values(per_head), step=epoch)
                valid_loss = selection_loss(per_head, config.training.checkpoint_metric)
                if best_loss is None or valid_loss < best_loss:
                    best_loss, best_epoch, since_best = valid_loss, epoch, 0
                    improved = True
                    # Inside the average's context, so what is kept is what
                    # was scored, which is also what the checkpoint holds.
                    best_state = _snapshot(model)
                    if checkpoint_path is not None:
                        written = write_model(checkpoint_path, model, built.metadata)
                else:
                    since_best += 1

        _step_schedule(scheduler, valid_loss)
        if checkpoint_path is not None:
            # After the schedule has stepped, so a resume continues with the
            # rate the next epoch would have had, and with the raw weights in
            # place: the average travels as its own state.
            directory, name = Path(checkpoint_path).parent, Path(checkpoint_path).name
            write_run_checkpoint(
                directory,
                name,
                RunState(epoch + 1, best_loss, best_epoch, since_best, stage, improved),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                metadata=built.metadata,
            )
            retain_run_checkpoints(
                directory,
                name,
                keep_improving=config.runtime.keep_checkpoints,
                keep_all=config.runtime.save_all_checkpoints,
            )
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
        if since_best >= config.training.patience:
            break

    if best_state is None and ema is not None:
        # The run's model is the averaged one. Leaving the stepped weights in
        # place would return a model that differs from the checkpoint beside
        # it and from every number the run reported. Before any merge, while
        # what trains is still what the average was kept over.
        with torch.no_grad():
            for parameter, shadow in zip(
                (p for p in model.parameters() if p.requires_grad),
                ema.shadow,
                strict=True,
            ):
                parameter.copy_(shadow)
    if config.finetune.lora.enabled:
        # Folded in before anything is read back or reported: a merged model
        # has the shapes of one never adapted, which is what the checkpoint
        # already holds, since a checkpoint reads each weight as adapted.
        merge_lora(model)
    if best_state is not None:
        # The run ends on its best epoch, as the frozen tree does by loading
        # its last checkpoint, which it writes only on an improvement. Ending
        # on the last epoch instead returns a model that is not the one on
        # disk, and every number reported after training describes the
        # wrong one of the two.
        load_canonical_state(model, best_state)

    if checkpoint_path is not None:
        # The model the run delivers, written once more at its end: the best
        # epoch's weights, averaged when the run averaged and merged when it
        # adapted, which is also what a run that never evaluated ends on.
        written = write_model(checkpoint_path, model, built.metadata)

    report_errors(config, built, model, loss, specs, tracker, device=device)
    tracker.finish()
    return TrainedModel(
        model=model,
        metadata=built.metadata,
        history=tuple(history),
        best_epoch=best_epoch,
        checkpoint_path=written,
    )


def report_errors(
    config: ResolvedConfig,
    built: TorchBuiltModel,
    model: nn.Module,
    loss: torch.nn.Module,
    specs: Sequence[MetricSpec],
    tracker: Tracker,
    *,
    device: str = "cpu",
) -> str:
    """Evaluate every reported loader once and log the table.

    Returns the rendered table as well as logging it, so a caller that wants
    to write it beside the model does not evaluate the run a second time to
    get it.

    The test sets get their own table, as they do in the frozen tree: they are
    not rows of the same table because a reader comparing a validation row
    against a test row across a table boundary is doing it deliberately.
    """
    requested = _requested_names(built)
    rows = [
        Row(
            name=name,
            head=name.split("_", 1)[1],
            metrics=evaluate(
                model,
                batches,
                loss,
                specs,
                device=device,
                compute=requested.derivatives,
            ),
        )
        for name, batches in built.data.reported_loaders().items()
    ]
    table = error_table(
        config.runtime.error_table,
        rows,
        skip_heads=config.data.skip_evaluate_heads,
    )
    logger.info("Errors on the training and validation sets:\n%s", table)
    tracker.summary(
        {
            f"final_{row.name}_{name}": value
            for row in rows
            for name, value in row.metrics.items()
        }
    )
    if built.data.test_loaders:
        test_rows = [
            Row(
                name=name,
                head=name,
                metrics=evaluate(
                    model,
                    batches,
                    loss,
                    specs,
                    device=device,
                    compute=requested.derivatives,
                ),
            )
            for name, batches in built.data.test_loaders.items()
        ]
        logger.info(
            "Errors on the test sets:\n%s",
            error_table(
                config.runtime.error_table,
                test_rows,
                skip_heads=config.data.skip_evaluate_heads,
            ),
        )
    return table


@dataclass(frozen=True)
class _Requested:
    """The names the loss scores and the derivatives the engine computes."""

    names: tuple[str, ...]
    per_atom: tuple[bool, ...]
    derivatives: tuple[str, ...]


def _requested_names(built: TorchBuiltModel) -> _Requested:
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


def _drops_tail(stage: StageConfig) -> bool:
    """Whether this stage drops the epoch's ragged last batch.

    A full-batch optimizer steps once an epoch through a closure over
    everything it was given, so a dropped tail is a slice of the dataset it
    never sees. A mini-batch one drops it, because a short last batch is a
    noisier gradient with the same learning rate behind it. The frozen tree
    spells the same distinction as ``drop_last=(not args.lbfgs)``.
    """
    return not isinstance(stage.optimizer, LBFGSOptimizer)


def log_validation(epoch: int, per_head: Mapping[str, dict[str, float]]) -> None:
    """One line per head, with the head named on every one of them.

    Named on every line rather than once per block: the lines are read in a log
    next to the other heads' and next to the next epoch's, where a heading
    several lines up has stopped applying.

    **The line does not depend on the error table.** It reports what the run
    measured, so no configuration can silence it and no branch can go
    unreached. The frozen tree writes one branch per table type, with no
    fallback: ``DipoleMAE`` matches none of them and prints nothing at all, and
    the two stress-or-virials branches are guarded on a lookup into a
    ``defaultdict``, so the stress branch is taken whether or not a stress was
    measured and the virials one below it is dead. Both are pinned in
    ``tests/unit/test_valid_err_log.py``.
    """
    for head, metrics in per_head.items():
        reported = ", ".join(
            f"{name}={value:.4g}"
            for name, value in sorted(metrics.items())
            if name == "loss" or name.startswith(("rmse_", "mae_"))
        )
        logger.info("epoch %d, head %s: %s", epoch, head, reported)


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


def _enter_stage(
    stage: StageConfig,
    config: ResolvedConfig,
    built: TorchBuiltModel,
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


def _prepare_finetune(config: ResolvedConfig, model: nn.Module) -> ResolvedConfig:
    """Adapt or freeze the model as the fine-tune section asks.

    Returns:
        The configuration with each frozen group's learning-rate factor at
        zero, beside whatever factors it already set, so an optimizer built
        from the factors alone agrees with the flags on the parameters.
    """
    settings = config.finetune
    if settings.lora.enabled:
        inject_lora(model, rank=settings.lora.rank, alpha=settings.lora.alpha)
    factors = freeze(model, settings.freeze)
    if not factors:
        return config
    scheduler = config.training.scheduler
    merged = {**scheduler.group_factors, **factors}
    training = config.training.model_copy(
        update={"scheduler": scheduler.model_copy(update={"group_factors": merged})}
    )
    return config.model_copy(update={"training": training})


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
