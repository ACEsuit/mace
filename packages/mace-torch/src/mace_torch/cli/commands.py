"""The ``mace`` commands this package implements.

Declared in the ``mace.commands`` entry point group, which is how ``mace``
finds them; nothing in :mod:`mace_core.cli` names this package.

``mace train`` runs from a configuration file and takes eight explicit flags,
spelled as the legacy training script spells them. Anything else is set in the
file.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import yaml
from mace_core.cli import Command, ConfigFlag, set_value
from mace_core.config.base import ConfigError
from mace_core.config.resolved import ResolvedConfig

__all__ = ["EXPORT_CONFIG", "TRAIN", "TRAIN_FLAGS"]

#: The head a flag's file goes to when the configuration names none. The same
#: name the legacy translator gives the head it builds from --train_file.
DEFAULT_HEAD = "default"


def _into_the_one_head(key: str):
    """A writer for a flag that belongs to a head: the only head the file
    declares, or ``default`` when it declares none."""

    def write(document: MutableMapping[str, Any], value: Any) -> None:
        data = document.get("data", {})
        heads = data.get("heads", {}) if isinstance(data, MutableMapping) else {}
        if isinstance(heads, MutableMapping) and len(heads) > 1:
            raise ConfigError(
                f"--{key} sets the file of one head, and the configuration "
                f"declares {sorted(heads)}. Set {key} under the head it belongs "
                f"to in the file."
            )
        head = next(iter(heads), DEFAULT_HEAD) if heads else DEFAULT_HEAD
        set_value(document, f"data.heads.{head}.{key}", value)

    return write


TRAIN_FLAGS: tuple[ConfigFlag, ...] = (
    ConfigFlag("--name", "runtime.name", "Name of the run and its files."),
    ConfigFlag("--seed", "runtime.seed", "Random seed.", type=int),
    ConfigFlag("--work_dir", "runtime.work_dir", "Where the run writes."),
    ConfigFlag("--device", "runtime.device", "cpu, cuda, mps or xpu."),
    ConfigFlag(
        "--train_file",
        "data.heads.<head>.train_file",
        "Training structures, for a configuration with at most one head.",
        write=_into_the_one_head("train_file"),
    ),
    ConfigFlag(
        "--valid_file",
        "data.heads.<head>.valid_file",
        "Validation structures, for a configuration with at most one head.",
        write=_into_the_one_head("valid_file"),
    ),
    ConfigFlag(
        "--foundation_model",
        "finetune.foundation_model",
        "Fine-tune this foundation model: a path or a published name.",
    ),
    ConfigFlag(
        "--max_num_epochs",
        "training.max_num_epochs",
        "Epoch ceiling.",
        type=int,
    ),
)


def _train(arguments: argparse.Namespace) -> int:
    from mace_torch.cli.run_train import train

    return train(arguments.configuration)


TRAIN = Command(
    help="Train a model from a configuration file.",
    run=_train,
    schema=ResolvedConfig,
    flags=TRAIN_FLAGS,
)


def _export_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", type=Path, help="A v1 model, either of its files.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write it: .yaml, .yml or .json.",
    )


#: What the stored configuration can be written as. TOML is read but not
#: written: it has no null, and a resolved configuration holds some.
_WRITERS = {
    ".json": lambda payload: json.dumps(payload, indent=2) + "\n",
    ".yaml": lambda payload: yaml.safe_dump(payload, sort_keys=False),
    ".yml": lambda payload: yaml.safe_dump(payload, sort_keys=False),
}


def _export_config(arguments: argparse.Namespace) -> int:
    from mace_core.metadata import ModelMetadata

    from mace_torch.serialization import CheckpointError, read_sidecar

    writer = _WRITERS.get(arguments.output.suffix.lower())
    if writer is None:
        raise SystemExit(
            f"mace model export-config: cannot write {arguments.output.suffix!r}; "
            f"use {', '.join(_WRITERS)}"
        )
    try:
        document = read_sidecar(arguments.model)
    except CheckpointError as error:
        raise SystemExit(f"mace model export-config: {error}") from error
    metadata = ModelMetadata.model_validate(document["config"])
    resolved = ResolvedConfig.from_dict(metadata.config.resolved).to_resolved_dict()
    arguments.output.write_text(writer(resolved))
    return 0


EXPORT_CONFIG = Command(
    help="Write the configuration a model was trained with to a file.",
    run=_export_config,
    arguments=_export_arguments,
)
