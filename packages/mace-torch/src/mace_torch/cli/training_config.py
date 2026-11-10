"""Translate command-line syntax once, then hand off a resolved configuration.

Legacy flags are an input language, not a second training data model. Settings
retired by the agreed configuration contract are refused with their disposition.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from mace_core.config import ConfigError, read_config_file
from mace_core.config.cli import apply_overrides, set_value
from mace_core.config.legacy import COMPUTE_FLAGS, LegacyFlagError, from_namespace
from mace_core.config.loss import LOSS_SHARED_FIELDS, normalize_loss
from mace_core.config.resolved import ResolvedConfig
from mace_core.config.training import SCHEDULER_SHARED_FIELDS, normalize_scheduler
from mace_core.observables import (
    DEFAULT_CATALOGUE,
    UnknownObservableError,
    resolve_requested,
)

from mace_torch.cli.eval_configs import boolean
from mace_torch.cli.legacy_flags import DEFAULTS, DIRECT_FLAGS


def _literal(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def legacy_document(given: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(given) - set(DEFAULTS)
    if unknown:
        raise LegacyFlagError(f"unknown legacy options: {sorted(unknown)}")
    values: dict[str, Any] = dict(given)
    if "foundation_model_kwargs" in values:
        options = values["foundation_model_kwargs"]
        # Invalid literals retain their value for the option-specific error below.
        with suppress(ValueError, SyntaxError):
            options = _literal(options)
        if isinstance(options, dict) and not options:
            values["foundation_model_kwargs"] = DEFAULTS["foundation_model_kwargs"]
    handled = {
        "heads",
        "default_dtype",
        "enable_cueq",
        "enable_oeq",
        "only_cueq",
        "log_dir",
        "model_dir",
        "checkpoints_dir",
        "results_dir",
        "wandb_log_hypers",
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
    }
    replay_options = {
        "weight_pt_head",
        "num_samples_pt",
        "real_pt_data_ratio_threshold",
        "pt_train_file",
        "pt_valid_file",
        "subselect_pt",
        "filter_type_pt",
        "allow_random_padding_pt",
    }
    if values.get("multiheads_finetuning") and not values.get("foundation_model"):
        raise LegacyFlagError("--multiheads_finetuning requires --foundation_model")
    if set(given) & replay_options and not values.get("multiheads_finetuning"):
        raise LegacyFlagError(
            "replay flags require --multiheads_finetuning true and a foundation; "
            "otherwise set the ordinary heads under data.heads"
        )
    check = argparse.Namespace(
        **{key: value for key, value in values.items() if key not in handled}
    )
    from_namespace(check, defaults=DEFAULTS)
    namespace = argparse.Namespace(
        **{
            key: value
            for key, value in values.items()
            if value is not None and key not in handled
        }
    )
    for name in ("radial_MLP", "config_type_weights", "embedding_specs"):
        if hasattr(namespace, name):
            setattr(namespace, name, _literal(getattr(namespace, name)))
    if hasattr(namespace, "skip_evaluate_heads"):
        namespace.skip_evaluate_heads = tuple(
            value.strip()
            for value in namespace.skip_evaluate_heads.split(",")
            if value.strip()
        )
    if hasattr(namespace, "les_arguments") and isinstance(namespace.les_arguments, str):
        namespace.les_arguments = read_config_file(namespace.les_arguments)
    if hasattr(namespace, "model"):
        # The old class spelling is translated only here; no class is imported.
        family = namespace.model.removesuffix("MACE")
        if not family:
            raise LegacyFlagError(
                "--model=MACE selected a scale-shift model with a forced zero shift "
                "in legacy training. V1 uses explicit model kinds; choose "
                "model: scale_shift in the configuration (with measured shift), "
                "or retain the legacy engine for identical training defaults."
            )
        namespace.model = {
            "ScaleShift": "scale_shift",
            "AtomicDipoles": "dipole",
            "AtomicDielectric": "dielectric",
            "Polar": "polar",
            "MACELES": "les",
            "Magnetic": "magnetic",
            "MagneticScaleShift": "magnetic",
        }.get(family, namespace.model)
    if hasattr(namespace, "scaling"):
        namespace.scaling = namespace.scaling.removesuffix("_scaling").replace(
            "no", "none"
        )
    if getattr(namespace, "hidden_irreps", None):
        from mace_core.clebsch_gordan.irreps import Irreps

        terms = Irreps.parse(namespace.hidden_irreps).terms
        widths = {count for count, _ in terms}
        if len(widths) != 1:
            raise LegacyFlagError(
                "--hidden_irreps requires the same channel count for every irrep"
            )
        namespace.num_channels = widths.pop()
        namespace.hidden_irreps = "+".join(str(irrep) for _, irrep in terms)
    document = from_namespace(namespace)
    if "default_dtype" in values:
        document["precision"] = {"model": values["default_dtype"]}
    for flag in ("enable_cueq", "only_cueq", "enable_oeq"):
        try:
            values[flag] = boolean(str(values.get(flag, False)))
        except argparse.ArgumentTypeError as error:
            raise LegacyFlagError(f"--{flag}: {error}") from error
    # V1 prefers CuEq for every supported op; both legacy spellings select it.
    backends = [
        name
        for enabled, name in (
            (values.get("enable_cueq") or values.get("only_cueq"), "cueq"),
            (values.get("enable_oeq"), "oeq"),
        )
        if enabled
    ]
    if len(backends) > 1:
        raise LegacyFlagError(
            "choose one backend: --enable_cueq/--only_cueq or --enable_oeq"
        )
    if backends:
        document.setdefault("model", {})["backend"] = backends[0]
    runtime = document.setdefault("runtime", {})
    directories = {
        str(given[name])
        for name in (
            "log_dir",
            "model_dir",
            "checkpoints_dir",
            "results_dir",
        )
        if given.get(name) is not None
    }
    if directories:
        if len(directories) > 1 or (
            "work_dir" in given and str(given["work_dir"]) not in directories
        ):
            raise LegacyFlagError(
                "V1 uses one --work_dir; separate output directories "
                "cannot be represented"
            )
        runtime["work_dir"] = directories.pop()
    if (
        "wandb_log_hypers" in given
        and given["wandb_log_hypers"] != DEFAULTS["wandb_log_hypers"]
    ):
        raise LegacyFlagError(
            "V1 logs the resolved config; --wandb_log_hypers cannot select a subset"
        )
    data = document.setdefault("data", {})
    if values.get("heads"):
        head_defaults = data.pop("heads", {}).get("default", {})
        heads = _literal(values["heads"])
        data["heads"] = {}
        for name, settings in heads.items():
            if settings.get("test_dir") is not None:
                raise LegacyFlagError(
                    f"heads.{name}.test_dir is not a per-head V1 setting; "
                    "use the global test_dir or explicit per-head test_file paths"
                )
            translated = legacy_document(settings)
            unsupported = {key for key, value in translated.items() if value} - {"data"}
            if unsupported:
                raise LegacyFlagError(
                    f"heads.{name} contains non-data settings: {sorted(unsupported)}"
                )
            data["heads"][name] = {
                **head_defaults,
                **translated.get("data", {}).get("heads", {}).get("default", {}),
            }
    if values.get("foundation_model") and values.get("multiheads_finetuning"):
        if values.get("pt_train_file") is None:
            raise LegacyFlagError(
                "legacy automatic replay needs an explicit pt_train_file "
                "or a V1 finetune preset"
            )
        replay = {
            "train_file": values["pt_train_file"],
            "valid_file": values.get("pt_valid_file"),
            "e0s": {"kind": "foundation"},
            "weight": values.get("weight_pt_head", 1.0),
            "readout_from": values.get("foundation_head"),
            "subselect": {
                "method": values.get("subselect_pt", "random"),
                "filtering": values.get("filter_type_pt", "none"),
                "num_samples": values.get("num_samples_pt", 10000),
                "allow_random_padding": values.get("allow_random_padding_pt", True),
            },
        }
        data["heads"] = {**data.get("heads", {}), "pt_head": replay}
        data["ratio_guard"] = {
            "reference": "pt_head",
            "threshold": values.get("real_pt_data_ratio_threshold", 0.1),
        }
    elif values.get("foundation_head"):
        for settings in data.get("heads", {}).values():
            settings["readout_from"] = values["foundation_head"]
    model = document.setdefault("model", {})
    if model.get("model") in {"dipole", "dielectric"}:
        model["observables"] = ["dipole"] + (
            ["polarizability"] if model["model"] == "dielectric" else []
        )
    elif values.get("loss") in {"stress", "virials", "huber", "universal"}:
        model["observables"] = list(
            dict.fromkeys(
                [
                    *model.get("observables", ["energy", "forces"]),
                    "virials" if values["loss"] == "virials" else "stress",
                ]
            )
        )
        if values["loss"] == "universal" and model.get("model") == "magnetic":
            model["observables"].append("magforces")
            document["loss"]["use_structure_weights"]["magforces"] = False
    elif values.get("loss") == "energy_forces_dipole":
        model["observables"] = ["energy", "forces", "dipole"]
    training = document.setdefault("training", {})
    if values.get("swa") and values.get("start_swa") is None:
        training.setdefault("stage_two", {})["start_epoch"] = int(
            int(values.get("max_num_epochs", DEFAULTS["max_num_epochs"])) * 0.75
        )
    if values.get("lbfgs") and "scheduler" not in given:
        training["scheduler"] = {"kind": "constant"}
    return document


def configuration(arguments: argparse.Namespace) -> ResolvedConfig:
    values = vars(arguments).copy()
    path = values.pop("config", None)
    values = {key: value for key, value in values.items() if key in DEFAULTS}
    document = apply_overrides(read_config_file(path), {}) if path else {}
    for section in ("data", "model", "training", "finetune", "runtime", "loss"):
        if section in document and not isinstance(document[section], Mapping):
            raise ConfigError(f"{section} must be a configuration section (a mapping)")
    if "loss" in document:
        document["loss"] = normalize_loss(document["loss"])
    training = document.get("training", {})
    if "scheduler" in training:
        training["scheduler"] = normalize_scheduler(training["scheduler"])
    for key, (target, _) in DIRECT_FLAGS.items():
        if key not in values:
            continue
        if "<head>" in target:
            heads = document.get("data", {}).get("heads", {})
            if not isinstance(heads, Mapping):
                raise ConfigError("data.heads must be a mapping of head names")
            if len(heads) > 1:
                raise ConfigError(
                    f"--{key} needs a single head, but found {sorted(heads)}; "
                    "set each head's file under data.heads"
                )
            target = target.replace("<head>", next(iter(heads), "default"))
        set_value(document, target, values.pop(key))
    if values:
        # A parameter-only override belongs to the choice in the file. Without
        # that context the compatibility translator can only use CLI defaults.
        training = document.get("training", {})
        if isinstance(training.get("optimizer"), Mapping):
            values.setdefault(
                "optimizer", training["optimizer"].get("kind", DEFAULTS["optimizer"])
            )
        scheduler = training.get("scheduler", {})
        choice = scheduler.get("kind") if isinstance(scheduler, Mapping) else None
        if choice is not None and not values.get("lbfgs"):
            values.setdefault("scheduler", choice)
        if "num_radial_basis" in values and "radial_type" not in values:
            radial = document.get("model", {}).get("radial_basis", {})
            if "kind" in radial:
                values["radial_type"] = radial["kind"]
        if values.get("multiheads_finetuning"):
            values["foundation_model"] = document.get("finetune", {}).get(
                "foundation_model"
            )
        # A hyperparameter override keeps the file's loss semantics. Replaying a
        # legacy loss preset here would also change observables and weighting.
        delta = values.get("huber_delta")
        if delta is not None:
            selected_loss = values.get(
                "loss", document.get("loss", {}).get("kind", "weighted")
            )
            if selected_loss not in {"huber", "universal"}:
                raise ConfigError("--huber_delta requires a huber or universal loss")
        translated = legacy_document(values)
        if delta is not None:
            translated.setdefault("loss", {})["delta"] = delta
        declared = document.get("model", {}).get("observables")
        if declared is not None and any(flag in values for flag in COMPUTE_FLAGS):
            selected = list(declared)
            for flag, observable in COMPUTE_FLAGS.items():
                if flag not in values:
                    continue
                if values[flag] and observable not in selected:
                    selected.append(observable)
                elif not values[flag] and observable in selected:
                    selected.remove(observable)
            translated.setdefault("model", {})["observables"] = selected
        heads = document.get("data", {}).get("heads", {})
        updates = translated.get("data", {}).get("heads", {})
        if "default" in updates and "heads" not in values and heads:
            if len(heads) != 1:
                raise LegacyFlagError(
                    "head data flags need a single head; set each head's values "
                    "under data.heads in the configuration"
                )
            updates[next(iter(heads))] = updates.pop("default")
        # An E0 flag selects a complete source. Fields belonging to the former
        # method, such as a foundation head, cannot accompany a literal table.
        for name, settings in updates.items():
            if "e0s" in settings and name in heads:
                heads[name].pop("e0s", None)
        # Changing a tagged choice replaces that choice's settings. Merging
        # across variants would keep, for example, Adam's beta in ScheduleFree.
        training = document.get("training", {})
        changed = translated.get("training", {})
        for previous, incoming, key, shared in (
            (training, changed, "optimizer", set()),
            (training, changed, "scheduler", SCHEDULER_SHARED_FIELDS),
            (
                document,
                translated,
                "loss",
                LOSS_SHARED_FIELDS,
            ),
        ):
            old, new = previous.get(key), incoming.get(key)
            if (
                isinstance(old, Mapping)
                and isinstance(new, Mapping)
                and "kind" in new
                and old.get("kind") != new["kind"]
            ):
                previous[key] = {
                    name: value for name, value in old.items() if name in shared
                }
        document = apply_overrides(document, translated)
        if values.get("foundation_head"):
            for settings in document.get("data", {}).get("heads", {}).values():
                settings["readout_from"] = values["foundation_head"]
    return _resolve_document(document)


def _resolve_document(document: dict[str, Any]) -> ResolvedConfig:
    if "model" in document and not isinstance(document["model"], Mapping):
        raise ConfigError("model must be a configuration section (a mapping)")
    observables = document.get("model", {}).get("observables", ())
    if not isinstance(observables, (list, tuple)):
        return ResolvedConfig.from_dict(document)
    if "virials" in document.get("model", {}).get(
        "observables", ()
    ) and not document.get("catalogue"):
        # A virial is an extensive negative strain derivative. The default
        # catalogue declares stress instead; persist the alternate declaration
        # for the loss, data loader and checkpoint to use the same convention.
        catalogue = DEFAULT_CATALOGUE.model_dump(mode="json")
        for observable in catalogue["observables"]:
            if observable["name"] == "energy":
                for derivative in observable["derivatives"]:
                    if derivative["wrt"] == "strain":
                        derivative.update(
                            name="virials", sign=-1, units="eV", extensive=True
                        )
        document["catalogue"] = catalogue
    config = ResolvedConfig.from_dict(document)
    try:
        resolve_requested(
            config.model.observables, config.catalogue or DEFAULT_CATALOGUE
        )
    except UnknownObservableError as error:
        raise ConfigError(f"model.observables: {error}") from error
    from mace_torch.models.registry import model_architecture

    model_architecture(config.model.model)
    return config
