"""Stages: what changes when one begins, and what a run keeps across them.

The frozen tree has exactly two stages and the second is written into the loop,
which swaps the loss and steps a different scheduler from a hardcoded epoch.
Here a stage is a row of configuration and the loop looks up which one an epoch
belongs to, so a third stage is a line rather than a code change.

The lookup rather than a switch is what makes a resume land in the right stage:
a run continued at epoch fifty belongs to the stage epoch fifty belongs to.
"""

from __future__ import annotations

import pytest
from mace_core.config.base import ConfigError
from mace_core.config.loss import LossConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.config.training import (
    AdamWOptimizer,
    ConstantSchedule,
    SchedulerConfig,
    StageConfig,
    StageTwoConfig,
    TrainingConfig,
)
from mace_torch.train.loop import _following_epoch, _stage_at
from pydantic import ValidationError


def resolved(**training) -> ResolvedConfig:
    return ResolvedConfig(training=TrainingConfig(**training))


# ---------------------------------------------------------------------------
# What a schedule is
# ---------------------------------------------------------------------------


def test_a_run_that_names_no_stage_has_one():
    assert [stage.name for stage in resolved().schedule()] == ["main"]


def test_the_short_two_stage_spelling_expands_to_two():
    """Legacy's SWA switch, as a schedule rather than as a branch in the loop."""
    schedule = resolved(
        stage_two=StageTwoConfig(enabled=True, start_epoch=4, lr=0.001)
    ).schedule()
    assert [(s.name, s.start_epoch, s.lr) for s in schedule] == [
        ("main", 0, None),
        ("stage_two", 4, 0.001),
    ]


def test_a_second_stage_that_never_begins_is_refused():
    """It is a run doing something other than what its configuration says.

    Refused when the configuration is built, not when the schedule is read, so
    a run that would do this never starts.
    """
    with pytest.raises(ValidationError, match="start_epoch"):
        resolved(stage_two=StageTwoConfig(enabled=True))


def test_writing_both_spellings_is_refused():
    """The short one is the long one written short, so both leaves two
    schedules to reconcile and no reading of that is obviously right."""
    with pytest.raises(ConfigError, match="Keep one"):
        resolved(
            stages=(StageConfig(),),
            stage_two=StageTwoConfig(enabled=True, start_epoch=2),
        ).schedule()


def test_the_stages_are_ordered_and_the_first_starts_at_zero():
    """A schedule whose earliest stage began at epoch three would leave the
    first three epochs in no stage at all."""
    schedule = resolved(
        stages=(
            StageConfig(name="late", start_epoch=9),
            StageConfig(name="early", start_epoch=3),
        )
    ).schedule()
    assert [(s.name, s.start_epoch) for s in schedule] == [("early", 0), ("late", 9)]


def test_the_second_stage_takes_its_loss_weights_from_the_loss_section():
    """One thing said in two places, joined where both are visible."""
    config = ResolvedConfig(
        loss=LossConfig(weights={"energy": 1.0}, stage_two_weights={"energy": 1000.0}),
        training=TrainingConfig(stage_two=StageTwoConfig(enabled=True, start_epoch=2)),
    )
    assert config.schedule()[1].loss_weights == {"energy": 1000.0}


def test_a_stage_that_names_its_own_weights_keeps_them():
    config = ResolvedConfig(
        loss=LossConfig(stage_two_weights={"energy": 1000.0}),
        training=TrainingConfig(
            stages=(
                StageConfig(),
                StageConfig(
                    name="stage_two", start_epoch=2, loss_weights={"energy": 7.0}
                ),
            )
        ),
    )
    assert config.schedule()[1].loss_weights == {"energy": 7.0}


# ---------------------------------------------------------------------------
# Which stage an epoch belongs to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "epoch,expected",
    [(0, "main"), (3, "main"), (4, "stage_two"), (50, "stage_two")],
)
def test_an_epoch_is_looked_up_rather_than_switched_into(epoch, expected):
    """Which is what lets a run resumed at epoch fifty land in the right one."""
    schedule = resolved(
        stage_two=StageTwoConfig(enabled=True, start_epoch=4)
    ).schedule()
    assert _stage_at(schedule, epoch).name == expected


def test_three_stages_need_no_code():
    """The claim the frozen tree's two cannot make."""
    schedule = resolved(
        stages=(
            StageConfig(name="warm", start_epoch=0, lr=0.01),
            StageConfig(name="main", start_epoch=5, lr=0.001),
            StageConfig(
                name="polish",
                start_epoch=8,
                lr=0.0001,
                optimizer=AdamWOptimizer(),
                scheduler=SchedulerConfig(kind=ConstantSchedule()),
            ),
        )
    ).schedule()
    assert [_stage_at(schedule, epoch).name for epoch in (0, 4, 5, 7, 8, 99)] == [
        "warm",
        "warm",
        "main",
        "main",
        "polish",
        "polish",
    ]


# ---------------------------------------------------------------------------
# Where a run goes when a stage stops improving
# ---------------------------------------------------------------------------


def test_a_stage_still_improving_trains_the_next_epoch():
    schedule = resolved(
        stage_two=StageTwoConfig(enabled=True, start_epoch=4)
    ).schedule()
    assert _following_epoch(schedule, 1, exhausted=False) == 2


def test_an_earlier_stage_out_of_patience_hands_over_to_the_next():
    """As the frozen tree moves to its second stage rather than stopping, so
    the stage the configuration asked for is not skipped."""
    schedule = resolved(
        stage_two=StageTwoConfig(enabled=True, start_epoch=4)
    ).schedule()
    assert _following_epoch(schedule, 1, exhausted=True) == 4


def test_the_next_stage_is_the_nearest_one_ahead():
    schedule = resolved(
        stages=(
            StageConfig(name="warm"),
            StageConfig(name="main", start_epoch=5),
            StageConfig(name="polish", start_epoch=8),
        )
    ).schedule()
    assert _following_epoch(schedule, 2, exhausted=True) == 5
    assert _following_epoch(schedule, 6, exhausted=True) == 8


def test_the_last_stage_out_of_patience_ends_the_run():
    schedule = resolved(
        stage_two=StageTwoConfig(enabled=True, start_epoch=4)
    ).schedule()
    assert _following_epoch(schedule, 6, exhausted=True) is None
