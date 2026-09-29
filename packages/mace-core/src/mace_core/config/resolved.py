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

from typing import Any, Literal

from pydantic import Field, model_validator

from mace_core.config.base import ReforgeBaseConfig
from mace_core.config.data import DataConfig
from mace_core.config.e0s import FOUNDATION_E0_KINDS, E0sIsolatedAtoms
from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.config.loss import LossConfig
from mace_core.config.model import ModelConfig
from mace_core.config.pseudolabels import PseudolabelConfig
from mace_core.config.runtime import RuntimeConfig
from mace_core.config.section import FrozenSection
from mace_core.config.training import StageConfig, TrainingConfig

__all__ = [
    "ENERGY_OBSERVABLE",
    "FinetuneConfig",
    "LoRAConfig",
    "PseudolabelConfig",
    "ResolvedConfig",
]

#: The observable an isolated-atom energy shifts. A model that does not declare
#: it has no atomic-energy term for an E0 to reach, whatever it is called.
ENERGY_OBSERVABLE = "energy"


class LoRAConfig(FrozenSection):
    """Low-rank adapters on the linear maps, trained instead of the model.

    Args:
        enabled: Whether to adapt at all.
        rank: The rank of each update, per irrep. Legacy's default.
        alpha: The update is scaled by ``alpha / rank``.
    """

    enabled: bool = False
    rank: int = Field(default=4, ge=1)
    alpha: float = 1.0


class FinetuneConfig(FrozenSection):
    """The foundation model a run starts from, and how much of it moves.

    Args:
        foundation_model: The artifact to start from, a v1 checkpoint. ``None``
            is a run trained from scratch, and it is what makes an E0 kind that
            reads a foundation model an error.
        element_table: Which elements the fine-tune's model is built over.
            ``"foundation"``, the default, keeps every element the foundation
            model has, so a model fine-tuned on two of them can be fine-tuned
            again on a third. ``"data"`` keeps only the elements the training
            data holds, which is what legacy does unless
            ``--foundation_model_elements`` is set.
        transfer_readout: Start each head's readout from the foundation
            model's, as ``readout_from`` names it. Off, only the backbone is
            transferred and the readouts start fresh. Legacy's
            ``--foundation_model_readout``, on by default there too.
        lora: Train low-rank adapters instead of the weights.
        freeze: Freeze the groups up to this level, as legacy's ``--freeze``
            layers them: one or more freezes the node embedding, five the
            interactions, six the products and seven the readouts. ``None`` or
            ``0`` freezes nothing. A negative value is refused: legacy documents
            ``-1`` as freezing the last layer and does nothing with it, since
            every threshold is a ``>=`` on a positive number.
        pseudolabels: Where the replay labels come from.
        preset: A named set of defaults for a kind of fine-tune.
            ``"multihead"`` is the frozen tree's multi-head fine-tune: a
            learning rate of 1e-4, an exponential moving average with decay
            0.99999, and the published replay datasets relabelled by the
            foundation model. Each is a default, and a value the configuration
            writes wins.
        preset_values: Which configuration path each preset value landed on,
            filled in when the configuration is resolved, so the resolved
            configuration says where a number came from.
    """

    foundation_model: str | None = None
    element_table: Literal["foundation", "data"] = "foundation"
    transfer_readout: bool = True
    lora: LoRAConfig = LoRAConfig()
    freeze: int | None = Field(default=None, ge=0)
    pseudolabels: PseudolabelConfig = PseudolabelConfig()
    preset: Literal["multihead"] | None = None
    preset_values: dict[str, str] = Field(default_factory=dict)


#: The multi-head preset's values, by configuration path. A path the configuration
#: writes itself is left alone.
MULTIHEAD_PRESET: dict[str, Any] = {
    "training.lr": 1e-4,
    "training.ema": {"enabled": True, "decay": 0.99999},
    "finetune.pseudolabels.enabled": True,
}


def _written(data: dict[str, Any], path: str) -> bool:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _write(data: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """``data`` with ``value`` at ``path``, copying every level it passes."""
    head, _, rest = path.partition(".")
    section = dict(data.get(head) or {})
    if rest:
        section = _write(section, rest, value)
        return {**data, head: section}
    return {**data, head: value}


class ResolvedConfig(ReforgeBaseConfig):
    """A validated run configuration, immutable once built.

    Args:
        runtime: Where the run writes and how it is driven.
        data: The structures and the heads.
        model: The architecture and the declared observables.
        loss: What the run is scored on.
        training: The optimizer, the schedule and the stages.
        finetune: The foundation model, reserved.
        electrostatics: The long-range solver, for the models that carry one.
    """

    model_config = ReforgeBaseConfig.model_config | {"frozen": True}

    runtime: RuntimeConfig = RuntimeConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    finetune: FinetuneConfig = FinetuneConfig()
    electrostatics: ElectrostaticsConfig = ElectrostaticsConfig()

    def schedule(self) -> tuple[StageConfig, ...]:
        """The stages this run goes through, with their loss weights filled in.

        The short two-stage spelling keeps its weights in the loss section,
        because that is where every other weight lives, and a stage carries its
        own. Reconciling the two needs both sections, which is why it happens
        here rather than in either of them: a stage written the short way and a
        loss section that names second-stage weights are one thing said in two
        places, and the loop should meet it already joined.
        """
        stages = self.training.schedule()
        if not self.loss.stage_two_weights:
            return stages
        return tuple(
            stage.model_copy(update={"loss_weights": dict(self.loss.stage_two_weights)})
            if stage.name == "stage_two" and not stage.loss_weights
            else stage
            for stage in stages
        )

    @model_validator(mode="before")
    @classmethod
    def _fill_fine_tune_defaults(cls, data: Any) -> Any:
        """The preset, then the E0s of the heads it relabels, in that order."""
        return cls._replay_heads_take_the_foundation_s_e0s(cls._apply_the_preset(data))

    @staticmethod
    def _apply_the_preset(data: Any) -> Any:
        """Write a named preset's values where the configuration left a gap.

        Before validation, so a value the configuration wrote is told apart
        from a default: a written value always wins, whatever it is, and the
        preset records the paths it filled. There is no flag that forces a
        preset over a written value, since nothing needs forcing.
        """
        if not isinstance(data, dict):
            return data
        finetune = data.get("finetune") or {}
        if not isinstance(finetune, dict) or finetune.get("preset") != "multihead":
            return data
        filled: dict[str, str] = {}
        for path, value in MULTIHEAD_PRESET.items():
            if path == "finetune.pseudolabels.enabled" and _written(
                data, "finetune.pseudolabels.labels_from"
            ):
                continue
            if not _written(data, path):
                data = _write(data, path, value)
                filled[path] = "multihead"
        # A resolved configuration read back has every value written and the
        # record of which ones the preset gave; the record is kept.
        recorded = dict(finetune.get("preset_values") or {})
        return _write(data, "finetune.preset_values", {**recorded, **filled})

    @staticmethod
    def _replay_heads_take_the_foundation_s_e0s(data: Any) -> Any:
        """A relabelled head's isolated-atom energies are the foundation's.

        Its energies are the foundation model's own, so the only E0s they are
        consistent with are that model's, from the head that labelled them.
        Written into a head that says nothing about its E0s; a head that says
        something else is refused after validation.
        """
        if not isinstance(data, dict):
            return data
        finetune = data.get("finetune") or {}
        heads = (data.get("data") or {}).get("heads") or {}
        if not isinstance(finetune, dict) or not isinstance(heads, dict):
            return data
        pseudolabels = finetune.get("pseudolabels") or {}
        if not isinstance(pseudolabels, dict) or not (
            pseudolabels.get("enabled") or pseudolabels.get("labels_from")
        ):
            return data
        named = pseudolabels.get("heads") or [
            name
            for name, head in heads.items()
            if isinstance(head, dict) and head.get("curated")
        ]
        for name in named:
            head = heads.get(name)
            if isinstance(head, dict) and "e0s" not in head:
                spec: dict[str, Any] = {"kind": "foundation"}
                if head.get("readout_from"):
                    spec["head"] = head["readout_from"]
                data = _write(data, f"data.heads.{name}.e0s", spec)
        return data

    @model_validator(mode="before")
    @classmethod
    def _an_error_table_for_what_is_read_out(cls, data: Any) -> Any:
        """The error table defaults to one the declared observables can fill.

        ``PerAtomRMSE`` reports energies and forces, which a model reading out
        a dipole has neither of. Written into the configuration, so the
        resolved one says which table the run prints.
        """
        if not isinstance(data, dict):
            return data
        model = data.get("model") or {}
        observables = model.get("observables") if isinstance(model, dict) else None
        runtime = data.get("runtime") or {}
        if (
            observables is None
            or ENERGY_OBSERVABLE in observables
            or not isinstance(runtime, dict)
            or "error_table" in runtime
        ):
            return data
        table = "DipolePolarRMSE" if "polarizability" in observables else "DipoleRMSE"
        return {**data, "runtime": {**runtime, "error_table": table}}

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
    def _a_loss_for_what_is_read_out(self) -> ResolvedConfig:
        """The universal loss is refused for a model that reads out no energy.

        What makes it universal is the band it puts on the force term by the
        reference force's norm. A model with no energy has no forces, so it
        would get a plain Huber under another name, and asking for it says the
        configuration was written for another model.
        """
        if ENERGY_OBSERVABLE in self.model.observables:
            return self
        if self.loss.kind.kind == "universal":
            raise ValueError(
                f"loss.kind is 'universal', which bands the force term, and "
                f"model.observables {list(self.model.observables)} has no "
                f"energy and so no forces. Use 'weighted', which is the frozen "
                f"tree's dipole and polarizability loss, or 'huber'."
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
        # Against the default rather than against what was written: a resolved
        # configuration writes every field, the default E0s included, and
        # reading it back has to give the same configuration.
        offenders = sorted(
            f"data.heads.{name}.e0s"
            for name, head in self.data.heads.items()
            if head.e0s != E0sIsolatedAtoms()
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

    @property
    def pseudolabelled_heads(self) -> tuple[str, ...]:
        """The heads whose structures the foundation model relabels.

        The ones ``finetune.pseudolabels.heads`` names, or every head reading
        a published replay dataset when it names none. Empty when the run
        relabels nothing.
        """
        pseudolabels = self.finetune.pseudolabels
        if not (pseudolabels.enabled or pseudolabels.labels_from is not None):
            return ()
        if pseudolabels.heads:
            return pseudolabels.heads
        return tuple(
            name for name, head in self.data.heads.items() if head.curated is not None
        )

    @model_validator(mode="after")
    def _pseudolabels_name_heads_that_exist(self) -> ResolvedConfig:
        pseudolabels = self.finetune.pseudolabels
        if not (pseudolabels.enabled or pseudolabels.labels_from is not None):
            return self
        if self.finetune.foundation_model is None:
            raise ValueError(
                "finetune.pseudolabels asks for replay labels from the "
                "foundation model, and finetune.foundation_model is not set."
            )
        unknown = sorted(set(pseudolabels.heads) - set(self.data.heads))
        if unknown:
            raise ValueError(
                f"finetune.pseudolabels.heads names {unknown}, which are not "
                f"heads of this run; its heads are {sorted(self.data.heads)}."
            )
        if not self.pseudolabelled_heads:
            raise ValueError(
                "finetune.pseudolabels is on and no head is relabelled: none "
                "names a published replay dataset. Name the heads in "
                "finetune.pseudolabels.heads."
            )
        for name in self.pseudolabelled_heads:
            head = self.data.heads[name]
            if head.e0s.kind != "foundation":
                raise ValueError(
                    f"data.heads.{name}.e0s is {head.e0s.kind!r}, and the head "
                    f"is relabelled by the foundation model, whose energies "
                    f"are consistent only with its own isolated-atom energies. "
                    f"Drop the e0s, or set them to kind 'foundation'."
                )
            labelled_by = getattr(head.e0s, "head", None)
            if (
                labelled_by is not None
                and head.readout_from is not None
                and labelled_by != head.readout_from
            ):
                raise ValueError(
                    f"data.heads.{name} takes its E0s from the foundation head "
                    f"{labelled_by!r} and its readout, and so its labels, from "
                    f"{head.readout_from!r}. They are one head."
                )
        return self

    @model_validator(mode="after")
    def _lbfgs_steps_too_rarely_for_an_ema_or_a_plateau(self) -> ResolvedConfig:
        for optimizer, scheduler in self._stages():
            if optimizer[1] != "lbfgs":
                continue
            if self.training.ema.enabled:
                raise ValueError(
                    f"{optimizer[0]} runs lbfgs and training.ema.enabled is "
                    f"set. L-BFGS steps once per epoch through a full-batch "
                    f"closure, so an average over its steps is an average "
                    f"over epochs and not what the field means."
                )
            if scheduler[1] == "plateau":
                raise ValueError(
                    f"{optimizer[0]} runs lbfgs and {scheduler[0]} is plateau. "
                    f"A plateau schedule reduces the rate on evaluations "
                    f"without improvement, and L-BFGS takes one step between "
                    f"them. Use a constant schedule for an L-BFGS stage."
                )
        return self

    def _stages(self) -> list[tuple[tuple[str, str], tuple[str, str]]]:
        """Each stage's optimizer and schedule, as the field that sets each
        and the kind it sets.

        Walked in the order the stages run, since a stage that inherits its
        optimizer or keeps its schedule runs whatever the one before it set,
        and that is the field an error has to name.
        """
        training = self.training
        optimizer: tuple[str, str] = ("training.optimizer", training.optimizer.kind)
        scheduler: tuple[str, str] = (
            "training.scheduler",
            training.scheduler.kind.kind,
        )
        stages: list[tuple[tuple[str, str], tuple[str, str]]] = [(optimizer, scheduler)]
        if training.stages:
            stages = []
            ordered = sorted(
                enumerate(training.stages), key=lambda item: item[1].start_epoch
            )
            for index, stage in ordered:
                field = f"training.stages[{index}]"
                if stage.optimizer.kind != "inherit":
                    optimizer = (f"{field}.optimizer", stage.optimizer.kind)
                if stage.scheduler is not None:
                    scheduler = (f"{field}.scheduler", stage.scheduler.kind.kind)
                stages.append((optimizer, scheduler))
        elif training.stage_two.enabled:
            if training.stage_two.optimizer.kind != "inherit":
                optimizer = (
                    "training.stage_two.optimizer",
                    training.stage_two.optimizer.kind,
                )
            stages.append((optimizer, scheduler))
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
