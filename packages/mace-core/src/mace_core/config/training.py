"""The optimizer, the schedule, and the second stage.

Two shapes legacy has that do not survive.

**LBFGS is a stage, not a boolean.** ``--lbfgs`` swaps the optimizer object
*after* the scheduler and the EMA have been built around the old one
(``mace/cli/run_train.py:971-976``), leaving both attached to a full-batch
closure that steps once an epoch. A plateau scheduler watching a per-epoch
closure and an EMA averaging one step per epoch are not wrong so much as
meaningless. Here a stage names its optimizer, so the combination is a
validation error rather than a silently inert pair of components.

**An optimizer's own tuning is a field, not a flag.** Naming ``schedulefree``
in an enum carries none of its three betas and warmup, so legacy has them as
top-level flags that mean nothing under the other optimizers. They live with
their optimizer here, the way a loss's delta lives with its loss.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from mace_core.config.section import FrozenSection

__all__ = [
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
    "ScheduleKind",
    "SchedulerConfig",
    "StageOptimizerKind",
    "StageTwoConfig",
    "TrainingConfig",
]


class AdamOptimizer(FrozenSection):
    """Adam.

    Args:
        beta: The second moment's decay, legacy's single ``--beta``.
        amsgrad: The AMSGrad variant.
    """

    kind: Literal["adam"] = "adam"
    beta: float = 0.9
    amsgrad: bool = True


class AdamWOptimizer(FrozenSection):
    """AdamW: Adam with the decay applied to the weights rather than the grad.

    Args:
        beta: As :class:`AdamOptimizer`.
        amsgrad: As :class:`AdamOptimizer`.
    """

    kind: Literal["adamw"] = "adamw"
    beta: float = 0.9
    amsgrad: bool = True


class ScheduleFreeOptimizer(FrozenSection):
    """The schedule-free variant, whose tuning surface is its own.

    Args:
        beta1: First moment decay.
        beta2: Second moment decay.
        warmup_steps: Steps over which the learning rate is warmed up.
    """

    kind: Literal["schedulefree"] = "schedulefree"
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 0


class LBFGSOptimizer(FrozenSection):
    """Full-batch L-BFGS, which steps once per epoch through a closure.

    A stage running this forbids an EMA and a plateau schedule. Both are built
    around a per-step optimizer, and against one step an epoch neither does
    what its name says.
    """

    kind: Literal["lbfgs"] = "lbfgs"


#: Which optimizer, with its own hyperparameters under it.
OptimizerKind = Annotated[
    AdamOptimizer | AdamWOptimizer | ScheduleFreeOptimizer | LBFGSOptimizer,
    Field(discriminator="kind"),
]


class InheritOptimizer(FrozenSection):
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


class ConstantSchedule(FrozenSection):
    """No learning-rate schedule at all."""

    kind: Literal["constant"] = "constant"


class PlateauSchedule(FrozenSection):
    """Reduce the learning rate when the validation loss stops improving.

    Args:
        factor: What the rate is multiplied by on a reduction.
        patience: Evaluations without improvement before reducing.
    """

    kind: Literal["plateau"] = "plateau"
    factor: float = 0.8
    patience: int = 50


class ExponentialSchedule(FrozenSection):
    """Multiply the learning rate by a constant every epoch.

    Args:
        gamma: The per-epoch factor.
    """

    kind: Literal["exponential"] = "exponential"
    gamma: float = 0.9993


#: Which learning-rate schedule, with its own settings under it.
ScheduleKind = Annotated[
    ConstantSchedule | PlateauSchedule | ExponentialSchedule,
    Field(discriminator="kind"),
]


class SchedulerConfig(FrozenSection):
    """The learning-rate schedule and the per-parameter-group factors.

    Args:
        kind: Which schedule.
        group_factors: Multipliers on the base rate, by parameter-group name.
            An unnamed group takes ``1.0``.
    """

    kind: ScheduleKind = PlateauSchedule()
    group_factors: dict[str, float] = Field(default_factory=dict)


class EMAConfig(FrozenSection):
    """The exponential moving average of the weights.

    Args:
        enabled: Whether to keep one.
        decay: Its decay.
    """

    enabled: bool = False
    decay: float = 0.99


class StageTwoConfig(FrozenSection):
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
    start_epoch: int | None = None
    lr: float = 0.001
    optimizer: StageOptimizerKind = InheritOptimizer()


class TrainingConfig(FrozenSection):
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
        stage_two: The second stage.
        dry_run: Build everything and stop before the first step.
    """

    optimizer: OptimizerKind = AdamOptimizer()
    lr: float = 0.01
    weight_decay: float = 5e-7
    batch_size: int = 10
    valid_batch_size: int = 10
    max_num_epochs: int = 2048
    patience: int = 2048
    eval_interval: int = 1
    clip_grad: float | None = 10.0
    scheduler: SchedulerConfig = SchedulerConfig()
    ema: EMAConfig = EMAConfig()
    stage_two: StageTwoConfig = StageTwoConfig()
    dry_run: bool = False
