"""Every legacy training flag, and what became of it.

One row per argparse *dest*, because a dest is what a namespace is keyed on, so
this table's domain and the object it reads are the same set. Not one row per
option string: ten of the flags are two spellings of one dest, and a table
keyed on spellings would claim twenty settings where there are ten.

Four things can have happened to a flag, and the difference between the last
two is the one worth having:

:class:`Kept`
    it is a field of the resolved config, named here by its dotted path.
:class:`Reserved`
    it is a field of a section this ticket does not build. The name is
    recorded so the flag is accounted for and so the shim can refuse a run
    that sets it, instead of the setting vanishing.
:class:`Merged`
    it became part of something else. ``applied`` says whether this module can
    carry the value across; where it cannot, a run that set it is refused for
    the same reason.
:class:`Dropped`
    it is gone, with the reason.

**A flag nobody can carry is refused, not ignored.** The failure this exists to
prevent is a user moving a working command line onto the new engine and getting
a run that trains something else: ``--default_dtype float32`` merges into a
precision section that does not exist yet, and silently training at float64 is
a worse answer than saying so. :func:`from_namespace` therefore takes the
parser's own defaults, and refuses a value that differs from one for a
destination it cannot reach.

The dispositions are the inventory's and are implemented here rather than
re-derived. Whether a flag survives is a decision that was made once, with the
source line in front of whoever made it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "LEGACY_TRAIN_DESTS",
    "Disposition",
    "Dropped",
    "Kept",
    "LegacyFlagError",
    "Merged",
    "Reserved",
    "from_namespace",
]


class LegacyFlagError(ValueError):
    """A legacy command line the new configuration cannot carry faithfully."""


@dataclass(frozen=True)
class Kept:
    """The flag is a field. ``path`` is its dotted path in the resolved config."""

    path: str


@dataclass(frozen=True)
class Reserved:
    """The flag is a field of a section another ticket builds.

    Args:
        section: The dotted path of the section it will live under.
        note: What owns it.
    """

    section: str
    note: str


@dataclass(frozen=True)
class Merged:
    """The flag became part of something else.

    Args:
        into: What it became part of.
        note: Why, in one line.
        applied: Whether :func:`from_namespace` can carry the value across. A
            merge into a section that does not exist yet cannot be, and a run
            that set the flag is refused rather than quietly losing it.
    """

    into: str
    note: str
    applied: bool = False


@dataclass(frozen=True)
class Dropped:
    """The flag is gone. ``reason`` says why, and is not optional."""

    reason: str


Disposition = Kept | Reserved | Merged | Dropped


def _kept(**paths: str) -> dict[str, Disposition]:
    return {dest: Kept(path) for dest, path in paths.items()}


def _reserved(section: str, note: str, *dests: str) -> dict[str, Disposition]:
    return {dest: Reserved(section, note) for dest in dests}


def _merged(into: str, note: str, *dests: str, applied: bool = False):
    return {dest: Merged(into, note, applied) for dest in dests}


#: Run and infrastructure.
_RUNTIME: dict[str, Disposition] = {
    **_kept(
        name="runtime.name",
        seed="runtime.seed",
        work_dir="runtime.work_dir",
        device="runtime.device",
        distributed="runtime.distributed",
        launcher="runtime.launcher",
        log_level="runtime.log_level",
        plot="runtime.plot",
        plot_frequency="runtime.plot_frequency",
        error_table="runtime.error_table",
        restart_latest="runtime.restart_latest",
        keep_checkpoints="runtime.keep_checkpoints",
        save_all_checkpoints="runtime.save_all_checkpoints",
    ),
    **_merged(
        "runtime.work_dir",
        "one work directory with a stated layout, instead of six defaults of '.'",
        "log_dir",
        "model_dir",
        "checkpoints_dir",
        "results_dir",
        "downloads_dir",
    ),
    **_merged(
        "the precision config",
        "a per-op-class precision choice rather than one global dtype",
        "default_dtype",
    ),
    **_merged(
        "model.backend",
        "one backend name; the third flag only chose between building in a "
        "layout and converting after, and canonical weights remove the "
        "conversion",
        "enable_cueq",
        "enable_oeq",
        "only_cueq",
    ),
    **_merged(
        "the config file itself",
        "reading a config file is the entry point's job, not a field in it",
        "config",
    ),
    "plot_interaction_e": Dropped(
        "a legacy plotting mode with no v1 consumer; plotting is the CLI's"
    ),
    "save_cpu": Dropped(
        "v1 checkpoints are device-agnostic, so there is nothing to ask"
    ),
}

#: Model architecture.
_MODEL: dict[str, Disposition] = {
    **_kept(
        r_max="model.r_max",
        radial_type="model.radial_type",
        num_radial_basis="model.num_radial_basis",
        num_cutoff_basis="model.num_cutoff_basis",
        pair_repulsion="model.pair_repulsion",
        distance_transform="model.distance_transform",
        apply_cutoff="model.apply_cutoff",
        interaction="model.interaction",
        interaction_first="model.interaction_first",
        max_ell="model.max_ell",
        correlation="model.correlation",
        use_agnostic_product="model.use_agnostic_product",
        num_interactions="model.num_interactions",
        radial_MLP="model.radial_mlp",
        hidden_irreps="model.hidden_irreps",
        edge_irreps="model.edge_irreps",
        use_edge_irreps_first="model.use_edge_irreps_first",
        num_channels="model.num_channels",
        max_L="model.max_L",
        gate="model.readout.gate",
        MLP_irreps="model.readout.mlp_irreps",
        scaling="model.scaling",
    ),
    **_merged(
        "model.model plus model.observables",
        "a registry name and a declared output list, not one class name that "
        "also decides what is computed",
        "model",
        applied=True,
    ),
    **_merged(
        "model.readout",
        "which layers read out is the readout's shape, not two booleans "
        "beside the interaction widths",
        "use_last_readout_only",
        "use_embedding_readout",
        applied=True,
    ),
    **_merged(
        "model.clebsch_gordan_basis",
        "recorded model state rather than a flag, because on the frozen tree "
        "it silently follows what is installed",
        "use_reduced_cg",
        applied=True,
    ),
    **_merged(
        "the dataset statistics",
        "a measured property of the data, carried in the model's metadata",
        "avg_num_neighbors",
        "compute_avg_num_neighbors",
        "atomic_numbers",
        "mean",
        "std",
    ),
    **_merged(
        "model.observables",
        "declaring a property is what computes it",
        "compute_stress",
        "compute_forces",
        "compute_polarizability",
        "compute_atomic_dipole",
        "compute_magforces",
        applied=True,
    ),
    "return_electrostatic_potentials": Dropped(
        "the frozen polar model returns None for it whatever it is set to, so "
        "there is no quantity behind it to carry"
    ),
    "use_so3": Dropped("an SO(3) variant with no trained artifact and no consumer"),
}

#: Data, files and property keys.
_DATA: dict[str, Disposition] = {
    **_kept(
        train_file="data.heads.default.train_file",
        valid_file="data.heads.default.valid_file",
        test_file="data.heads.default.test_file",
        E0s="data.heads.default.e0s",
        keep_isolated_atoms="data.heads.default.keep_isolated_atoms",
        config_type_weights="data.heads.default.config_type_weights",
        test_dir="data.test_dir",
        valid_fraction="data.valid_fraction",
        num_workers="data.num_workers",
        pin_memory="data.pin_memory",
        statistics_file="data.statistics_file",
        embedding_specs="data.embedding_specs",
        skip_evaluate_heads="data.skip_evaluate_heads",
        elec_temp_key="data.graph_input_keys.elec_temp",
        total_spin_key="data.graph_input_keys.total_spin",
        total_charge_key="data.graph_input_keys.total_charge",
    ),
    **_merged(
        "the property-key convention",
        "the file keys every labelled dataset on disk was written against, "
        "which is a data contract and not a per-run setting",
        "energy_key",
        "forces_key",
        "virials_key",
        "stress_key",
        "dipole_key",
        "polarizability_key",
        "charges_key",
        "head_key",
        "magmom_key",
        "magforces_key",
    ),
    **_merged(
        "the dataset layer",
        "whether a test set is split across files is something the loader "
        "reads off the dataset",
        "multi_processed_test",
    ),
    **_reserved("data.les", "the electrostatics extra", "les_arguments"),
}

#: Loss.
_LOSS: dict[str, Disposition] = {
    **_merged(
        "loss.kind",
        "a loss carries its own hyperparameters, so a delta cannot be set on "
        "a loss that has none",
        "huber_delta",
        applied=True,
    ),
    **_merged(
        "loss.weights",
        "a weight keyed by the observable's own name, so declaring a property "
        "is what gives it one",
        "energy_weight",
        "forces_weight",
        "virials_weight",
        "stress_weight",
        "dipole_weight",
        "polarizability_weight",
        "magforces_weight",
        applied=True,
    ),
    **_merged(
        "loss.stage_two_weights",
        "the second stage overrides the weights it names, instead of a "
        "parallel set of seven flags",
        "swa_energy_weight",
        "swa_forces_weight",
        "swa_virials_weight",
        "swa_stress_weight",
        "swa_dipole_weight",
        "swa_polarizability_weight",
        "swa_magforces_weight",
        applied=True,
    ),
    **_merged(
        "loss.kind",
        "the named schemes become a loss with its own settings under it",
        "loss",
        applied=True,
    ),
}

#: Optimizer, scheduler and training control.
_TRAINING: dict[str, Disposition] = {
    **_kept(
        weight_decay="training.weight_decay",
        batch_size="training.batch_size",
        valid_batch_size="training.valid_batch_size",
        lr="training.lr",
        max_num_epochs="training.max_num_epochs",
        patience="training.patience",
        eval_interval="training.eval_interval",
        clip_grad="training.clip_grad",
        dry_run="training.dry_run",
        ema="training.ema.enabled",
        ema_decay="training.ema.decay",
    ),
    **_merged(
        "training.optimizer",
        "an optimizer carries its own tuning, so a hyperparameter lands under "
        "the optimizer it belongs to and the schedulefree trio cannot be set "
        "under Adam; lbfgs is a stage's optimizer rather than a boolean "
        "bolted onto one",
        "optimizer",
        "lbfgs",
        "beta",
        "amsgrad",
        "beta1_schedulefree",
        "beta2_schedulefree",
        "warmup_steps_schedulefree",
        applied=True,
    ),
    **_merged(
        "training.scheduler.kind",
        "a schedule carries its own settings, under the schedule they belong to",
        "scheduler",
        "lr_factor",
        "scheduler_patience",
        "lr_scheduler_gamma",
        applied=True,
    ),
    **_merged(
        "training.scheduler.group_factors",
        "typed per-parameter-group factors",
        "lr_params_factors",
        applied=True,
    ),
    **_merged(
        "training.stage_two",
        "one second stage with a start, a rate and its own optimizer",
        "swa",
        "start_swa",
        "swa_lr",
        applied=True,
    ),
}

#: Fine-tuning, multihead and foundation models. The section is reserved: this
#: ticket builds only what its cross-section validators read.
_FINETUNE: dict[str, Disposition] = {
    **_kept(foundation_model="finetune.foundation_model"),
    **_reserved(
        "finetune.pseudolabels",
        "the pseudolabel replay work",
        "pseudolabel_replay",
        "pseudolabel_replay_compute_stress",
    ),
    **_kept(
        freeze="finetune.freeze",
        foundation_model_readout="finetune.transfer_readout",
        lora="finetune.lora.enabled",
        lora_rank="finetune.lora.rank",
        lora_alpha="finetune.lora.alpha",
    ),
    # A legacy fine-tune adds a replay head nobody declared, with defaults
    # that depend on which foundation model it starts from. Here that head is
    # declared like any other, so these flags become one head's settings.
    **_reserved(
        "data.heads",
        "the legacy flag port",
        "multiheads_finetuning",
        "foundation_head",
        "weight_pt_head",
        "num_samples_pt",
        "real_pt_data_ratio_threshold",
        "pt_train_file",
        "pt_valid_file",
        "subselect_pt",
        "filter_type_pt",
        "allow_random_padding_pt",
    ),
    **_reserved(
        "finetune",
        "loading a foundation model by name",
        "foundation_model_kwargs",
    ),
    **_reserved(
        "finetune",
        "the fine-tuning tickets",
        "foundation_model_elements",
        "finetune_dipoles_polarizabilities",
    ),
    **_reserved("data.heads", "the multi-head data work", "heads"),
    "force_mh_ft_lr": Dropped(
        "a preset is a declared default the user overrides, so there is "
        "nothing to force"
    ),
}

#: Sections another ticket owns entirely.
_EXTRAS: dict[str, Disposition] = {
    **_kept(
        kspace_cutoff_factor="electrostatics.kspace_cutoff_factor",
        atomic_multipoles_max_l="model.polar.multipole_max_l",
        atomic_multipoles_smearing_width="model.polar.multipole_width",
        field_feature_max_l="model.polar.feature_max_l",
        num_recursion_steps="model.polar.num_recursion_steps",
        field_si="model.polar.feature_self_interaction",
        include_electrostatic_self_interaction="model.polar.energy_self_interaction",
        add_local_electron_energy="model.polar.add_local_electron_energy",
        quadrupole_feature_corrections="model.polar.quadrupole_feature_corrections",
    ),
    **_merged(
        "model.polar",
        "written as Python literals on the frozen command line, parsed here",
        "field_feature_widths",
        "field_feature_norms",
        "fixedpoint_update_config",
        "field_readout_config",
        applied=True,
    ),
    "field_norm_factor": Dropped(
        "the frozen polar model stores it as a buffer and never reads it: its "
        "energy is the same at 1 and at 5"
    ),
    **_reserved(
        "model.magnetic",
        "the magnetic ticket",
        "m_max",
        "max_m_ell",
        "num_mag_radial_basis",
        "num_mag_radial_basis_one_body",
        "use_magmom_one_body",
        "train_one_body_contribution",
    ),
    **_merged(
        "the data transforms",
        "a training-data augmentation, which is the data layer's and not the model's",
        "data_aug_magmom",
        "data_aug_magmom_mode",
    ),
    **_reserved(
        "runtime.tracking",
        "the experiment-tracking work",
        "wandb",
        "wandb_dir",
        "wandb_project",
        "wandb_entity",
        "wandb_name",
        "wandb_log_hypers",
    ),
}

#: Every ``mace_run_train`` dest, and what became of it. The test asserts this
#: is exactly the parser's dest set, so a flag added to the frozen tree fails
#: rather than passing unnoticed.
LEGACY_TRAIN_DESTS: dict[str, Disposition] = {
    **_RUNTIME,
    **_MODEL,
    **_DATA,
    **_LOSS,
    **_TRAINING,
    **_FINETUNE,
    **_EXTRAS,
}


#: The legacy weight flag of each observable, first stage and second. Written
#: out because the seven are a closed set in the frozen tree and deriving them
#: from a name would claim a weight for an observable that never had a flag.
_WEIGHT_FLAGS: dict[str, str] = {
    "energy_weight": "energy",
    "forces_weight": "forces",
    "virials_weight": "virials",
    "stress_weight": "stress",
    "dipole_weight": "dipole",
    "polarizability_weight": "polarizability",
    "magforces_weight": "magforces",
}

#: Which observable each ``--compute_*`` flag declares.
_COMPUTE_FLAGS: dict[str, str] = {
    "compute_forces": "forces",
    "compute_stress": "stress",
    "compute_polarizability": "polarizability",
    "compute_atomic_dipole": "dipole",
    "compute_magforces": "magforces",
}

#: The hyperparameters each optimizer takes, as legacy dest to v1 field.
_OPTIMIZER_SETTINGS: dict[str, dict[str, str]] = {
    "adam": {"beta": "beta", "amsgrad": "amsgrad"},
    "adamw": {"beta": "beta", "amsgrad": "amsgrad"},
    "schedulefree": {
        "beta1_schedulefree": "beta1",
        "beta2_schedulefree": "beta2",
        "warmup_steps_schedulefree": "warmup_steps",
    },
    "lbfgs": {},
}

#: The settings each schedule takes. Legacy's two scheduler names map onto the
#: kinds; anything else is left to the schema to refuse by name.
_SCHEDULE_KINDS: dict[str, str] = {
    "ReduceLROnPlateau": "plateau",
    "ExponentialLR": "exponential",
}

_SCHEDULE_SETTINGS: dict[str, dict[str, str]] = {
    "plateau": {"lr_factor": "factor", "scheduler_patience": "patience"},
    "exponential": {"lr_scheduler_gamma": "gamma"},
}


def _set(target: dict[str, Any], path: str, value: Any) -> None:
    """Write ``value`` at a dotted ``path`` in a nested dict, creating levels."""
    *parents, leaf = path.split(".")
    for name in parents:
        target = target.setdefault(name, {})
    target[leaf] = value


def _unreachable(dest: str, disposition: Dropped | Reserved | Merged) -> str:
    """Why a flag's value cannot be carried, in the words of its disposition."""
    if isinstance(disposition, Dropped):
        return f"--{dest} is gone: {disposition.reason}"
    if isinstance(disposition, Reserved):
        return (
            f"--{dest} belongs to {disposition.section}, which {disposition.note} "
            f"brings and this configuration does not carry yet"
        )
    return f"--{dest} merged into {disposition.into}: {disposition.note}"


def from_namespace(
    namespace: Any,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn a legacy argparse namespace into a resolved-config mapping.

    Args:
        namespace: Anything with the legacy dests as attributes, which is what
            ``build_default_arg_parser().parse_args()`` returns.
        defaults: The parser's own defaults, by dest. Given, a flag set to
            anything else whose destination this module cannot reach is an
            error rather than a silent loss. Omitted, those flags are ignored
            and the caller has said it does not mind.

    Returns:
        A mapping to validate into the resolved config. Not the config itself:
        building it here would make this module the place the schema is
        constructed, and the schema has one of those already.

    Raises:
        LegacyFlagError: When ``defaults`` is given and the command line sets a
            flag whose destination does not exist yet. The message names the
            flag and what became of it.
    """
    values: dict[str, Any] = {}
    unreachable: list[str] = []
    for dest, disposition in LEGACY_TRAIN_DESTS.items():
        if not hasattr(namespace, dest):
            continue
        value = getattr(namespace, dest)
        if isinstance(disposition, Kept):
            if defaults is None or value != defaults.get(dest):
                _set(values, disposition.path, value)
            continue
        if isinstance(disposition, Merged) and disposition.applied:
            continue  # carried by the collapses below
        if defaults is not None and value != defaults.get(dest):
            unreachable.append(_unreachable(dest, disposition))
    _collapse(namespace, values)
    if unreachable:
        raise LegacyFlagError(
            "this command line sets flags the new configuration cannot carry, "
            "and running it anyway would train something else:\n  "
            + "\n  ".join(sorted(unreachable))
        )
    return values


def _read(namespace: Any, dest: str, fallback: Any = None) -> Any:
    return getattr(namespace, dest, fallback)


def _collapse(namespace: Any, values: dict[str, Any]) -> None:
    """Carry the flags that became part of something else.

    Each group is one legacy setting spread over several flags, or several
    flags that are one setting here. Written as its own function per group so
    a reader can check one against the flags it replaces.
    """
    _collapse_model(namespace, values)
    _collapse_polar(namespace, values)
    _collapse_observables(namespace, values)
    _collapse_loss(namespace, values)
    _collapse_optimizer(namespace, values)
    _collapse_schedule(namespace, values)
    _collapse_stage_two(namespace, values)


def _collapse_model(namespace: Any, values: dict[str, Any]) -> None:
    if (model := _read(namespace, "model")) is not None:
        _set(values, "model.model", model)
    if (reduced := _read(namespace, "use_reduced_cg")) is not None:
        _set(values, "model.clebsch_gordan_basis", "reduced" if reduced else "full")
    for dest, field in (
        ("use_last_readout_only", "last_only"),
        ("use_embedding_readout", "from_embedding"),
    ):
        if (value := _read(namespace, dest)) is not None:
            _set(values, f"model.readout.{field}", value)


#: The one field-update and the one field-readout block the frozen tree has,
#: by the class name its command line takes and the name v1 records.
_FIELD_UPDATES = {"AgnosticEmbeddedOneBodyVariableUpdate": "embedded_one_body"}
_FIELD_READOUTS = {"OneBodyMLPFieldReadout": "one_body_mlp"}
_POTENTIAL_EMBEDDINGS = {"AgnosticChargeBiasedLinearPotentialEmbedding"}


def _literal(value: Any) -> Any:
    """A flag the frozen command line takes as a Python literal in a string."""
    import ast

    return ast.literal_eval(value) if isinstance(value, str) else value


def _collapse_polar(namespace: Any, values: dict[str, Any]) -> None:
    """The charge-aware flags that arrive as literals, and the two block choices.

    ``nonlinearity_cls`` inside ``--fixedpoint_update_config`` is not carried:
    the frozen update block discards it (``field_blocks.py:480``), so every
    value of it builds the same model.

    Raises:
        LegacyFlagError: For a block the frozen tree could name and v1 does not
            build, since building the one that exists would be another model.
    """
    if (widths := _literal(_read(namespace, "field_feature_widths"))) is not None:
        _set(values, "model.polar.feature_widths", [float(w) for w in widths])
    if (norms := _literal(_read(namespace, "field_feature_norms"))) is not None:
        _set(values, "model.polar.feature_norms", [float(n) for n in norms])
    update = _literal(_read(namespace, "fixedpoint_update_config")) or {}
    readout = _literal(_read(namespace, "field_readout_config")) or {}
    kind = update.get("type", "AgnosticEmbeddedOneBodyVariableUpdate")
    embedding = update.get(
        "potential_embedding_cls", "AgnosticChargeBiasedLinearPotentialEmbedding"
    )
    reading = readout.get("type", "OneBodyMLPFieldReadout")
    unknown = [
        f"{flag} names {name!r}; the blocks are {sorted(known)}"
        for flag, name, known in (
            ("--fixedpoint_update_config", kind, _FIELD_UPDATES),
            ("--fixedpoint_update_config", embedding, _POTENTIAL_EMBEDDINGS),
            ("--field_readout_config", reading, _FIELD_READOUTS),
        )
        if name not in known
    ]
    if unknown:
        raise LegacyFlagError("; ".join(unknown) + ".")
    if update:
        _set(values, "model.polar.field_update", _FIELD_UPDATES[kind])
    if readout:
        _set(values, "model.polar.field_readout", _FIELD_READOUTS[reading])


def _collapse_observables(namespace: Any, values: dict[str, Any]) -> None:
    """The five `--compute_*` booleans become the declared output list.

    The energy is not among them: legacy has no `--compute_energy`, because
    every model it can build produces one.
    """
    declared = ["energy"]
    declared += [
        observable
        for dest, observable in _COMPUTE_FLAGS.items()
        if _read(namespace, dest)
    ]
    if len(declared) > 1 or _read(namespace, "compute_forces") is not None:
        _set(values, "model.observables", declared)


def _collapse_loss(namespace: Any, values: dict[str, Any]) -> None:
    if (loss := _read(namespace, "loss")) is not None:
        settings: dict[str, Any] = {}
        if (
            loss in {"huber", "universal"}
            and (delta := _read(namespace, "huber_delta")) is not None
        ):
            settings["delta"] = delta
        _set(values, "loss.kind", {loss: settings})
    for stage, prefix in (("weights", ""), ("stage_two_weights", "swa_")):
        written = {
            observable: weight
            for dest, observable in _WEIGHT_FLAGS.items()
            if (weight := _read(namespace, f"{prefix}{dest}")) is not None
        }
        if written:
            _set(values, f"loss.{stage}", written)


def _collapse_optimizer(namespace: Any, values: dict[str, Any]) -> None:
    """`--lbfgs` wins over `--optimizer`, which is what legacy does by swapping
    the object after the fact."""
    kind = "lbfgs" if _read(namespace, "lbfgs") else _read(namespace, "optimizer")
    if kind is None:
        return
    settings = {
        field: value
        for dest, field in _OPTIMIZER_SETTINGS.get(kind, {}).items()
        if (value := _read(namespace, dest)) is not None
    }
    _set(values, "training.optimizer", {kind: settings})


def _collapse_schedule(namespace: Any, values: dict[str, Any]) -> None:
    scheduler = _read(namespace, "scheduler")
    if scheduler is None:
        return
    kind = _SCHEDULE_KINDS.get(scheduler, scheduler)
    settings = {
        field: value
        for dest, field in _SCHEDULE_SETTINGS.get(kind, {}).items()
        if (value := _read(namespace, dest)) is not None
    }
    _set(values, "training.scheduler.kind", {kind: settings})
    factors = _group_factors(_read(namespace, "lr_params_factors"))
    if factors:
        _set(values, "training.scheduler.group_factors", factors)


#: The suffix every key of the legacy factor mapping carries, and the groups
#: here do not.
_FACTOR_SUFFIX = "_lr_factor"


def _group_factors(raw: Any) -> dict[str, float]:
    """The per-group factors, keyed by the group names this schema uses.

    Legacy takes JSON in a string, keyed ``embedding_lr_factor`` and so on,
    and always has a value: its default sets all four to one. So the value is
    parsed, the suffix dropped, and a mapping of ones dropped altogether,
    since every factor at one is no factor and recording it would make a run
    look tuned when it was not.

    Raises:
        LegacyFlagError: On a key without the suffix, which names no group,
            and on JSON that is not a mapping.
    """
    if not raw:
        return {}
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, Mapping):
        raise LegacyFlagError(
            f"--lr_params_factors is {raw!r}, which is not a mapping of "
            f"'<group>{_FACTOR_SUFFIX}' to a factor."
        )
    factors: dict[str, float] = {}
    for key, value in parsed.items():
        if not key.endswith(_FACTOR_SUFFIX):
            raise LegacyFlagError(
                f"--lr_params_factors names {key!r}, which is not a group. The "
                f"keys are '<group>{_FACTOR_SUFFIX}', for example "
                f"'embedding{_FACTOR_SUFFIX}'."
            )
        factors[key[: -len(_FACTOR_SUFFIX)]] = float(value)
    if all(value == 1.0 for value in factors.values()):
        return {}
    return factors


def _collapse_stage_two(namespace: Any, values: dict[str, Any]) -> None:
    stage: dict[str, Any] = {}
    if (enabled := _read(namespace, "swa")) is not None:
        stage["enabled"] = enabled
    if (start := _read(namespace, "start_swa")) is not None:
        stage["start_epoch"] = start
    if (lr := _read(namespace, "swa_lr")) is not None:
        stage["lr"] = lr
    if stage:
        _set(values, "training.stage_two", stage)
