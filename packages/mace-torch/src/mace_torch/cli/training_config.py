"""Translate command-line syntax once, then hand off a resolved configuration.

Legacy flags are an input language, not a second training data model. Settings
retired by the agreed configuration contract are refused with their disposition.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from typing import Any

from mace_core.config import read_config_file
from mace_core.config.cli import apply_overrides
from mace_core.config.legacy import LegacyFlagError, from_namespace
from mace_core.config.resolved import ResolvedConfig

from mace_torch.cli.legacy_flags import DEFAULTS


def _literal(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def legacy_document(given: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(given) - set(DEFAULTS)
    if unknown:
        raise LegacyFlagError(f"unknown legacy options: {sorted(unknown)}")
    values: dict[str, Any] = dict(given)
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
        "downloads_dir",
        "wandb",
        "wandb_dir",
        "wandb_project",
        "wandb_entity",
        "wandb_name",
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
        **{key: value for key, value in given.items() if key not in handled}
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
    backends = [
        name
        for flag, name in (("enable_cueq", "cueq"), ("enable_oeq", "oeq"))
        if values.get(flag)
    ]
    if len(backends) > 1:
        raise LegacyFlagError("choose only one of --enable_cueq and --enable_oeq")
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
            "downloads_dir",
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
    if any(key.startswith("wandb") for key in values):
        runtime["wandb"] = {
            **({"enabled": values["wandb"]} if "wandb" in values else {}),
            **{
                field: values[flag]
                for flag, field in (
                    ("wandb_project", "project"),
                    ("wandb_entity", "entity"),
                    ("wandb_name", "name"),
                    ("wandb_dir", "directory"),
                )
                if flag in values
            },
        }
    data = document.setdefault("data", {})
    if values.get("heads"):
        head_defaults = data.pop("heads", {}).get("default", {})
        heads = _literal(values["heads"])
        data["heads"] = {}
        for name, settings in heads.items():
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
    training = document.setdefault("training", {})
    if values.get("swa") and values.get("start_swa") is None:
        training.setdefault("stage_two", {})["start_epoch"] = int(
            int(values.get("max_num_epochs", DEFAULTS["max_num_epochs"])) * 0.75
        )
    if values.get("lbfgs") and "scheduler" not in given:
        training["scheduler"] = {"kind": {"kind": "constant"}}
    return document


def configuration(arguments: argparse.Namespace) -> ResolvedConfig:
    values = vars(arguments).copy()
    path = values.pop("config", None)
    values = {key: value for key, value in values.items() if key in DEFAULTS}
    document = read_config_file(path) if path else {}
    # A legacy file consists of flag destinations; a structured file is validated
    # unchanged so a misspelled section gets the schema's nearest-key error.
    legacy_file = (
        bool(document)
        and bool(set(document) & set(DEFAULTS) - {"model"})
        and not any(
            isinstance(document.get(key), dict) for key in ResolvedConfig.model_fields
        )
    )
    if legacy_file:
        return ResolvedConfig.from_dict(legacy_document({**document, **values}))
    from mace_torch.cli.commands import TRAIN_FLAGS

    for flag in TRAIN_FLAGS:
        key = flag.option.removeprefix("--")
        if key in values:
            flag.apply(document, values.pop(key))
    if values:
        if values.get("multiheads_finetuning"):
            values["foundation_model"] = document.get("finetune", {}).get(
                "foundation_model"
            )
        translated = legacy_document(values)
        heads = document.get("data", {}).get("heads", {})
        updates = translated.get("data", {}).get("heads", {})
        if "default" in updates and "heads" not in values and heads:
            if len(heads) != 1:
                raise LegacyFlagError(
                    "head data flags need a single head; set each head's values "
                    "under data.heads in the configuration"
                )
            updates[next(iter(heads))] = updates.pop("default")
        document = apply_overrides(document, translated)
        if values.get("foundation_head"):
            for settings in document.get("data", {}).get("heads", {}).values():
                settings["readout_from"] = values["foundation_head"]
    return ResolvedConfig.from_dict(document)
