"""The whole training configuration, and the rules that span its sections.

One immutable validated object at the head of the pipeline. What makes it worth
having is not the sections, which are only fields grouped by subject, but the
**validators between them**: every one of them replaces a place where legacy
notices an incoherent configuration and keeps training anyway.

Legacy's answer to each is a log line, and a log line is what a run that takes
two days does not have a reader for:

* E0s from a foundation model with no foundation model configured. Legacy
  asserts this for one of the two kinds and not the other
  (``mace/cli/run_train.py:527``).
* A dipole or dielectric model with E0s set. Under
  ``--finetune_dipoles_polarizabilities`` legacy warns and rewrites the
  request to ``average`` (``:153-159``), because the model it is about to
  build has no atomic-energy term at all.
* L-BFGS beside an EMA or a plateau schedule, which legacy builds and then
  leaves attached to an optimizer that steps once an epoch (``:971-976``).
* Pseudolabels asked for and also read from a file, which is two sources for
  one set of labels.

Each is a validation error here, raised before anything is built, naming the
fields that disagree.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator

from mace_core.config.base import ReforgeBaseConfig
from mace_core.config.data import DataConfig
from mace_core.config.e0s import FOUNDATION_E0_KINDS
from mace_core.config.loss import LossConfig
from mace_core.config.model import ModelConfig
from mace_core.config.runtime import RuntimeConfig
from mace_core.config.section import FrozenSection
from mace_core.config.training import TrainingConfig

__all__ = ["ENERGY_OBSERVABLE", "FinetuneConfig", "PseudolabelConfig", "ResolvedConfig"]

#: The observable an isolated-atom energy shifts. A model that does not declare
#: it has no atomic-energy term for an E0 to reach, whatever it is called.
ENERGY_OBSERVABLE = "energy"


class PseudolabelConfig(FrozenSection):
    """Replay labels regenerated from the foundation model, or read back.

    Args:
        enabled: Generate them during this run.
        labels_from: Read them from a previous run's artifact instead.
    """

    enabled: bool = False
    labels_from: Path | None = None


class FinetuneConfig(FrozenSection):
    """The foundation model a run starts from.

    Reserved. Its full field set lands with the fine-tuning tickets; what is
    here is what the cross-section validators in this module have to read, so
    that the rules ship with the sections they constrain rather than waiting
    for the section that completes them.

    Args:
        foundation_model: The artifact to start from. ``None`` is a run
            trained from scratch, and it is what makes an E0 kind that reads a
            foundation model an error.
        pseudolabels: Where the replay labels come from.
    """

    foundation_model: str | None = None
    pseudolabels: PseudolabelConfig = PseudolabelConfig()


class ResolvedConfig(ReforgeBaseConfig):
    """A validated run configuration, immutable once built.

    Args:
        runtime: Where the run writes and how it is driven.
        data: The structures and the heads.
        model: The architecture and the declared observables.
        loss: What the run is scored on.
        training: The optimizer, the schedule and the stages.
        finetune: The foundation model, reserved.
    """

    model_config = ReforgeBaseConfig.model_config | {"frozen": True}

    runtime: RuntimeConfig = RuntimeConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    finetune: FinetuneConfig = FinetuneConfig()

    @model_validator(mode="after")
    def _e0s_that_read_a_foundation_model_need_one(self) -> ResolvedConfig:
        if self.finetune.foundation_model is not None:
            return self
        offenders = sorted(
            f"data.heads.{name}.e0s.{head.e0s.kind}"
            for name, head in self.data.heads.items()
            if head.e0s.kind in FOUNDATION_E0_KINDS
        )
        if offenders:
            raise ValueError(
                f"{offenders} read the isolated-atom energies out of a "
                f"foundation model, and finetune.foundation_model is not set. "
                f"Set it, or choose an E0 kind that does not need one."
            )
        return self

    @model_validator(mode="after")
    def _a_model_without_atomic_energies_cannot_carry_e0s(self) -> ResolvedConfig:
        """Read off the declared observables, not off the model's name.

        Legacy asks which class it is about to build, so the rule is a list of
        class names that goes stale the moment a model is added. What makes an
        E0 meaningless is that there is no energy for it to shift, and the
        declaration says that directly.
        """
        if ENERGY_OBSERVABLE in self.model.observables:
            return self
        offenders = sorted(
            f"data.heads.{name}.e0s"
            for name, head in self.data.heads.items()
            if "e0s" in head.model_fields_set
        )
        if offenders:
            raise ValueError(
                f"{offenders} set isolated-atom energies and "
                f"model.observables does not declare {ENERGY_OBSERVABLE!r}, so "
                f"there is no energy for them to shift. Declare it, or drop "
                f"the E0s."
            )
        return self

    @model_validator(mode="after")
    def _pseudolabels_come_from_one_place(self) -> ResolvedConfig:
        pseudolabels = self.finetune.pseudolabels
        if pseudolabels.enabled and pseudolabels.labels_from is not None:
            raise ValueError(
                "finetune.pseudolabels sets both enabled and labels_from, "
                "which asks for the labels to be generated in this run and "
                "read from "
                f"{pseudolabels.labels_from} at the same time. Set one."
            )
        return self

    @model_validator(mode="after")
    def _lbfgs_steps_too_rarely_for_an_ema_or_a_plateau(self) -> ResolvedConfig:
        for stage, optimizer in self._stages():
            if optimizer != "lbfgs":
                continue
            if self.training.ema.enabled:
                raise ValueError(
                    f"{stage} runs lbfgs and training.ema.enabled is set. "
                    f"L-BFGS steps once per epoch through a full-batch "
                    f"closure, so an average over its steps is an average "
                    f"over epochs and not what the field means."
                )
            if self.training.scheduler.kind.kind == "plateau":
                raise ValueError(
                    f"{stage} runs lbfgs and training.scheduler is plateau. "
                    f"A plateau schedule reduces the rate on evaluations "
                    f"without improvement, and L-BFGS takes one step between "
                    f"them. Use a constant schedule for an L-BFGS stage."
                )
        return self

    def _stages(self) -> list[tuple[str, str]]:
        """Each stage that runs, with the optimizer kind it runs."""
        stages = [("training.optimizer", self.training.optimizer.kind)]
        stage_two = self.training.stage_two
        if stage_two.enabled and stage_two.optimizer.kind != "inherit":
            stages.append(("training.stage_two.optimizer", stage_two.optimizer.kind))
        return stages

    @model_validator(mode="after")
    def _a_second_stage_says_when_it_starts(self) -> ResolvedConfig:
        stage_two = self.training.stage_two
        if stage_two.enabled and stage_two.start_epoch is None:
            raise ValueError(
                "training.stage_two.enabled is set and start_epoch is not. A "
                "second stage with no start is a run that reports two stages "
                "and runs one."
            )
        return self
