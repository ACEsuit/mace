"""The optimizer, the schedule, and the second stage.

Two shapes legacy has that do not survive.

**LBFGS is a stage, not a boolean.** The TRN-5 contract (#1639) excludes
EMA and plateau schedules from full-batch stages. Legacy's optimizer swap
leaves its scheduler attached to the previous optimizer, but its EMA still
updates the model parameters once per epoch. V1's refusal is an explicit
stage policy, not a claim that averaging full-batch updates is undefined.
An averaged mini-batch checkpoint can be resumed in a separate L-BFGS run.

**An optimizer's own tuning is a field, not a flag.** Naming ``schedulefree``
in an enum carries none of its three betas and warmup, so legacy has them as
top-level flags that mean nothing under the other optimizers. They live with
their optimizer here, the way a loss's delta lives with its loss.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, model_validator

from mace_core.config.base import ConfigSection, default_choice
from mace_core.config.loss import LossWeights

__all__ = [
    "SCHEDULER_SHARED_FIELDS",
    "AdamOptimizer",
    "AdamWOptimizer",
    "ConstantSchedule",
    "EMAConfig",
    "ExponentialSchedule",
    "InheritOptimizer",
    "LBFGSOptimizer",
    "OptimizerKind",
    "PlateauSchedule",
    "ScheduleFreeOptimizer",
    "SchedulerConfig",
    "StageConfig",
    "StageOptimizerKind",
    "StageTwoConfig",
    "TrainingConfig",
]


class AdamOptimizer(ConfigSection):
    """Adam.

    Args:
        beta: The first moment's decay, legacy's single ``--beta``.
        amsgrad: The AMSGrad variant.
    """

    kind: Literal["adam"] = "adam"
    beta: float = Field(default=0.9, ge=0, lt=1, allow_inf_nan=False)
    amsgrad: bool = True


class AdamWOptimizer(ConfigSection):
    """AdamW: Adam with the decay applied to the weights rather than the grad.

    Args:
        beta: As :class:`AdamOptimizer`.
        amsgrad: As :class:`AdamOptimizer`.
    """

    kind: Literal["adamw"] = "adamw"
    beta: float = Field(default=0.9, ge=0, lt=1, allow_inf_nan=False)
    amsgrad: bool = True


class ScheduleFreeOptimizer(ConfigSection):
    """The schedule-free variant, whose tuning surface is its own.

    Args:
        beta1: First moment decay.
        beta2: Second moment decay.
        warmup_steps: Steps over which the learning rate is warmed up.
    """

    kind: Literal["schedulefree"] = "schedulefree"
    beta1: float = Field(default=0.9, ge=0, lt=1, allow_inf_nan=False)
    beta2: float = Field(default=0.999, ge=0, lt=1, allow_inf_nan=False)
    warmup_steps: int = Field(default=0, ge=0)


class LBFGSOptimizer(ConfigSection):
    """Full-batch L-BFGS, which steps once per epoch through a closure.

    A stage running this is a full-batch stage: every epoch is one optimizer
    step over the whole training set, and the set keeps its ragged tail. It
    excludes EMA and plateau schedules under the TRN-5 stage contract.

    The defaults are the frozen tree's, which builds the optimizer with these
    three and leaves the rest at torch's.

    Args:
        lr: The step length the line search starts from. Not
            ``training.lr``, which is a mini-batch rate: the frozen tree never
            passes it, so a full-batch stage starts from torch's 1.0.
        history_size: How many past updates approximate the curvature.
        max_iter: Closure evaluations a single step may take, line search
            included.
        line_search_fn: ``"strong_wolfe"``, or ``None`` for a fixed step.
    """

    kind: Literal["lbfgs"] = "lbfgs"
    lr: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    history_size: int = Field(default=200, ge=1)
    max_iter: int = Field(default=20, ge=1)
    line_search_fn: Literal["strong_wolfe"] | None = "strong_wolfe"


#: Which optimizer, with its own hyperparameters under it.
OptimizerKind = Annotated[
    AdamOptimizer | AdamWOptimizer | ScheduleFreeOptimizer | LBFGSOptimizer,
    Field(discriminator="kind"),
]


class InheritOptimizer(ConfigSection):
    """Keep the optimizer the previous stage was running.

    A kind rather than an absent field. `None` under a kinds field cannot say
    whether it means "none of them" or "not written", and the two are
    different answers; naming the intention removes the question.
    """

    kind: Literal["inherit"] = "inherit"


#: A stage's optimizer, which may also be the one before it.
StageOptimizerKind = Annotated[
    InheritOptimizer
    | AdamOptimizer
    | AdamWOptimizer
    | ScheduleFreeOptimizer
    | LBFGSOptimizer,
    Field(discriminator="kind"),
]


class _SchedulerSettings(ConfigSection):
    """Multipliers on the base learning rate, by parameter-group name."""

    group_factors: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]] = (
        Field(default_factory=dict)
    )


SCHEDULER_SHARED_FIELDS = frozenset(_SchedulerSettings.model_fields)


class ConstantSchedule(_SchedulerSettings):
    """No learning-rate schedule at all."""

    kind: Literal["constant"] = "constant"


class PlateauSchedule(_SchedulerSettings):
    """Reduce the learning rate when the validation loss stops improving.

    Args:
        factor: What the rate is multiplied by on a reduction.
        patience: Evaluations without improvement before reducing.
    """

    kind: Literal["plateau"] = "plateau"
    factor: float = Field(default=0.8, gt=0, lt=1, allow_inf_nan=False)
    patience: int = Field(default=50, ge=0)


class ExponentialSchedule(_SchedulerSettings):
    """Multiply the learning rate by a constant every epoch.

    Args:
        gamma: The per-epoch factor.
    """

    kind: Literal["exponential"] = "exponential"
    gamma: float = Field(default=0.9993, gt=0, allow_inf_nan=False)


def normalize_scheduler(value: Any) -> Any:
    return default_choice(value, default="plateau", common=SCHEDULER_SHARED_FIELDS)


SchedulerConfig = Annotated[
    ConstantSchedule | PlateauSchedule | ExponentialSchedule,
    Field(discriminator="kind"),
    BeforeValidator(normalize_scheduler),
]


class EMAConfig(ConfigSection):
    """The exponential moving average of the weights.

    Args:
        enabled: Whether to keep one.
        decay: Its decay.
    """

    enabled: bool = False
    decay: float = Field(default=0.99, ge=0, le=1, allow_inf_nan=False)


class StageConfig(ConfigSection):
    """One stage of a run: where it starts and what changes when it does.

    A run is a sequence of these. The frozen tree has exactly two, and the
    second one is written into the loop: it swaps the loss and steps a
    different scheduler from a hardcoded epoch. Two is enough until it is not,
    and what it costs is that a third is a code change rather than a line of
    configuration.

    Everything here is an override. What a stage does not name it keeps from
    the settings around it, so a stage that only lowers the learning rate says
    only that.

    Args:
        name: What it is called, in the log and in the run's history.
        start_epoch: The epoch it begins at. Stages are ordered by it, and the
            first one starts at zero whatever it says.
        lr: Its learning rate. ``None`` keeps the previous one.
        optimizer: Its optimizer. Inheriting is a kind rather than an absence.
        scheduler: Its learning-rate schedule. ``None`` keeps the previous one.
        loss_weights: Per-observable weights that replace the run's for this
            stage. An observable it does not name keeps its weight, so a stage
            that only raises the energy weight says only that.
    """

    name: str = Field(default="main", min_length=1)
    start_epoch: int = Field(default=0, ge=0)
    lr: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    optimizer: StageOptimizerKind = InheritOptimizer()
    scheduler: SchedulerConfig | None = None
    loss_weights: LossWeights = Field(default_factory=dict)


class StageTwoConfig(ConfigSection):
    """The second stage, which legacy spells SWA.

    Both spellings name one setting in legacy, as option-string aliases of one
    dest, so they collapse to one field here and the ``stage_two`` spelling is
    the one kept.

    Args:
        enabled: Whether there is a second stage.
        start_epoch: The epoch it begins at. Required when enabled, because a
            second stage that never starts is a run the user thinks is doing
            something it is not.
        lr: The learning rate it runs at.
        optimizer: Its optimizer. Defaults to inheriting the first stage's.
    """

    enabled: bool = False
    start_epoch: int | None = Field(default=None, ge=1)
    lr: float = Field(default=0.001, ge=0, allow_inf_nan=False)
    optimizer: StageOptimizerKind = InheritOptimizer()


class TrainingConfig(ConfigSection):
    """The optimizer, the schedule, the stages, and the loop's own limits.

    Args:
        optimizer: Which optimizer, with its hyperparameters.
        lr: The base learning rate.
        weight_decay: The decay applied to the decaying parameter groups.
        batch_size: Training batch size.
        valid_batch_size: Validation batch size, which is a separate setting
            because validation has no gradients to hold.
        max_num_epochs: Epoch ceiling.
        patience: Evaluations without improvement before stopping.
        eval_interval: Epochs between evaluations.
        clip_grad: Gradient-norm clip. ``None`` does not clip.
        scheduler: The learning-rate schedule.
        ema: The weight average.
        checkpoint_metric: Which number the best checkpoint is chosen by, out
            of the per-head validation losses. ``mean_over_heads`` averages
            them unweighted. ``last_head`` is the frozen tree's, and it depends
            on the order the heads happen to be listed in.
        head_balancing: How a multi-head epoch is made up. ``balanced``
            up-samples every head to the largest, so each contributes the same
            number of steps whatever its size. ``proportional`` is the frozen
            tree's single shuffled pool, where a head is visited in proportion
            to its size; it is what a comparison against legacy needs. With one
            head the two are the same sequence.
        stage_two: The second stage, as the short spelling of a two-stage
            list. `schedule()` expands it.
        stages: The stages, when a run wants more than two or wants to name
            what changes at each. Mutually exclusive with `stage_two`.
        dry_run: Build everything and stop before the first step.
    """

    optimizer: OptimizerKind = AdamOptimizer()
    lr: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    weight_decay: float = Field(default=5e-7, ge=0, allow_inf_nan=False)
    batch_size: int = Field(default=10, ge=1)
    valid_batch_size: int = Field(default=10, ge=1)
    max_num_epochs: int = Field(default=2048, ge=0)
    patience: int = Field(default=2048, ge=0)
    eval_interval: int = Field(default=1, ge=1)
    clip_grad: float | None = Field(default=10.0, ge=0, allow_inf_nan=False)
    scheduler: SchedulerConfig = PlateauSchedule()
    ema: EMAConfig = EMAConfig()
    checkpoint_metric: Literal["mean_over_heads", "last_head"] = "mean_over_heads"
    head_balancing: Literal["balanced", "proportional"] = "balanced"
    stage_two: StageTwoConfig = StageTwoConfig()
    stages: tuple[StageConfig, ...] = ()
    dry_run: bool = False

    @model_validator(mode="after")
    def _valid_stages(self):
        if self.stages and self.stage_two.enabled:
            raise ValueError("choose training.stages or training.stage_two, not both")
        if self.stage_two.enabled and self.stage_two.start_epoch is None:
            raise ValueError("training.stage_two.enabled requires start_epoch")
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("training.stages requires a unique name for every stage")
        starts = [stage.start_epoch for stage in self.stages]
        if len(starts) != len(set(starts)):
            raise ValueError("training.stages requires distinct start_epoch values")
        return self

    def schedule(self) -> tuple[StageConfig, ...]:
        """The stages this run goes through, in order, starting at zero.

        The two-stage setting is a stage list written the short way, so it is
        expanded here rather than read separately by the loop: a loop that knew
        about both would be the special case this exists to remove. Naming both
        is refused, because the two would have to be reconciled and there is no
        reading of that which is obviously right.

        Raises:
            ConfigError: If both spellings are used. An enabled second stage
                that names no start is refused earlier, when the resolved
                configuration is built.
        """
        from mace_core.config.base import ConfigError

        if self.stages and self.stage_two.enabled:
            raise ConfigError(
                "both `training.stages` and `training.stage_two` are set. The "
                "second is the short spelling of a two-stage list, so writing "
                "both leaves two schedules to reconcile. Keep one."
            )
        if self.stages:
            ordered = sorted(self.stages, key=lambda stage: stage.start_epoch)
            return tuple(
                stage if index else stage.model_copy(update={"start_epoch": 0})
                for index, stage in enumerate(ordered)
            )
        if not self.stage_two.enabled:
            return (StageConfig(name="main", start_epoch=0),)
        # An enabled second stage with no start is refused when the resolved
        # configuration is built, so by here it has one.
        assert self.stage_two.start_epoch is not None
        return (
            StageConfig(name="main", start_epoch=0),
            StageConfig(
                name="stage_two",
                start_epoch=self.stage_two.start_epoch,
                lr=self.stage_two.lr,
                optimizer=self.stage_two.optimizer,
            ),
        )
