"""The ``mace`` commands this package implements.

Declared in the ``mace.commands`` entry point group, which is how ``mace``
finds them; nothing in :mod:`mace_core.cli` names this package.

Commands use explicit flags and resolved configuration files. Optional
frameworks and plotting dependencies are imported only when a command runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from mace_core.cli import Command
from mace_core.config.resolved import ResolvedConfig

__all__ = ["EXPORT_CONFIG", "TRAIN", "VALIDATE_CONFIG"]


def _train(arguments: argparse.Namespace) -> int:
    from mace_torch.cli.run_train import train
    from mace_torch.cli.training_config import configuration

    return train(configuration(arguments))


def _train_arguments(parser: argparse.ArgumentParser) -> None:
    from mace_torch.cli.legacy_flags import add_arguments

    add_arguments(parser)


TRAIN = Command(
    help="Train a model from a configuration file.",
    run=_train,
    arguments=_train_arguments,
)


def _validation_arguments(parser: argparse.ArgumentParser) -> None:
    _train_arguments(parser)
    parser.add_argument(
        "--show",
        choices=("minimal", "resolved"),
        help="Print explicit settings or the resolved configuration as YAML.",
    )


def _display_config(config: ResolvedConfig, *, minimal: bool) -> dict:
    def ordered(payload, resolved):
        if isinstance(payload, dict):
            # A selected union's discriminator is needed even when the Python
            # constructor supplied it by default. Put it first for readability.
            result = {"kind": resolved["kind"]} if "kind" in resolved else {}
            result.update(
                {
                    key: ordered(value, resolved[key])
                    for key, value in payload.items()
                    if key != "kind"
                }
            )
            return result
        if isinstance(payload, list):
            return [
                ordered(value, full)
                for value, full in zip(payload, resolved, strict=True)
            ]
        return payload

    resolved = config.to_resolved_dict()
    return ordered(config.to_user_dict() if minimal else resolved, resolved)


def _validate_config(arguments: argparse.Namespace) -> int:
    from mace_torch.cli.training_config import configuration

    config = configuration(arguments)
    if arguments.show:
        print(
            yaml.safe_dump(
                _display_config(config, minimal=arguments.show == "minimal"),
                sort_keys=False,
            ),
            end="",
        )
    else:
        print("Configuration is valid.")
    return 0


VALIDATE_CONFIG = Command(
    help="Validate training settings without reading datasets or building a model.",
    run=_validate_config,
    arguments=_validation_arguments,
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


def _module_command(
    module: str, help: str, *, arguments: str = "add_arguments", run: str = "run"
) -> Command:
    import importlib

    def add(parser):
        getattr(importlib.import_module(f"mace_torch.cli.{module}"), arguments)(parser)

    def execute(args):
        return getattr(importlib.import_module(f"mace_torch.cli.{module}"), run)(args)

    return Command(help=help, run=execute, arguments=add)


def _eval_arguments(parser):
    from mace_torch.cli.eval_configs import add_arguments
    from mace_torch.cli.polar_density_cube import build_parser

    add_arguments(parser)
    subcommands = parser.add_subparsers(dest="eval_action")
    subcommands.add_parser("density-cube", parents=[build_parser()], add_help=False)


def _eval(args):
    if args.eval_action == "density-cube":
        from mace_torch.cli.polar_density_cube import run

        for path in run(args):
            print(path)
        return 0
    from mace_torch.cli.eval_configs import run

    return run(args)


def _pseudolabel_arguments(parser):
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True, type=Path)


def _pseudolabel(args):
    from mace_torch.finetune.stages import pseudolabel

    print(pseudolabel(ResolvedConfig.load(args.config), args.head, args.output))
    return 0


def _lammps_arguments(parser):
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--native",
        nargs=2,
        type=int,
        metavar=("NODE_SLOTS", "EDGE_SLOTS"),
        help=(
            "Compile native.pt2 for the LAMMPS pair style. "
            "Node slots include ghosts and one padding atom."
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--head")
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--backend", help="Request a kernel backend when loading.")
    parser.add_argument(
        "--halo-exchange", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--compile-regions", action=argparse.BooleanOptionalAction, default=True
    )


def _lammps(args):
    from mace_torch.deploy.bundle import export_lammps

    print(
        export_lammps(
            args.checkpoint,
            args.output,
            head=args.head,
            dtype=args.dtype,
            backend=args.backend,
            halo_exchange=args.halo_exchange,
            compile_regions=args.compile_regions,
            native_capacity=args.native,
            device=args.device,
        )
    )
    return 0


def _convert_device_arguments(parser):
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", required=True)


def _convert_device(args):
    from mace_torch.deploy.loader import load_deployed

    load_deployed(args.model, device=args.device)
    print(
        f"{args.model}: loaded on {args.device}. V1 checkpoints are "
        f"device-independent; no rewrite is needed."
    )
    return 0


EVAL = Command(
    help="Evaluate structures, or write density cubes.",
    run=_eval,
    arguments=_eval_arguments,
)
SELECT_HEAD = _module_command(
    "model", "Extract one model head.", arguments="select_arguments", run="select_head"
)
CONVERT = _module_command(
    "model",
    "Convert full-basis weights to the reduced basis.",
    arguments="convert_arguments",
    run="convert",
)
CONVERT_DEVICE = Command(
    help="Verify a checkpoint loads on a device.",
    run=_convert_device,
    arguments=_convert_device_arguments,
)
PREPARE = _module_command("preprocess_data", "Write v2 dataset shards and manifests.")
PSEUDOLABEL = Command(
    help="Generate replay labels with the training implementation.",
    run=_pseudolabel,
    arguments=_pseudolabel_arguments,
)
PLOT = _module_command("plot_train", "Plot training results.")


def _retired_md(args):
    raise ValueError(
        "active-learning MD was retired from V1 (feature inventory): "
        "use ASE dynamics with MACECalculator's committee mean and "
        "variance"
    )


MD = Command(
    help="Retired: use ASE dynamics with the committee calculator.", run=_retired_md
)
MIGRATE = _module_command("migrate", "Translate a legacy command or training YAML.")
EXPORT_LAMMPS = Command(
    help="Export an ML-IAP deployment bundle.", arguments=_lammps_arguments, run=_lammps
)


def _full_basis_arguments(parser):
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)


def _full_basis(args):
    from mace_torch.deploy.full_basis import export_full_basis_weights

    print(export_full_basis_weights(args.bundle, args.output))
    return 0


EXPORT_FULL_BASIS = Command(
    help="Export minimum-norm full-basis contraction weights for external readers.",
    arguments=_full_basis_arguments,
    run=_full_basis,
)


def _select_data(args):
    from mace_torch.finetune.stages import select_structures

    print(select_structures(ResolvedConfig.load(args.config), args.head, args.output))
    return 0


SELECT_DATA = Command(
    help="Write a head's selected structures using its configured selection.",
    run=_select_data,
    arguments=_pseudolabel_arguments,
)
