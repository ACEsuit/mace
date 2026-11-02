"""Translate a legacy invocation into a command and a typed V1 configuration."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

COMMAND_MAP = {
    "mace_run_train": "mace --engine v1 train",
    "mace_eval_configs": "mace --engine v1 eval",
    "mace_select_head": "mace --engine v1 model select-head",
    "mace_convert_device": "mace --engine v1 model convert-device",
    "mace_e3nn_cueq": None,
    "mace_cueq_to_e3nn": None,
    "mace_create_lammps_model": "mace --engine v1 export lammps",
    "mace_finetuning_select": "mace --engine v1 data select",
    "mace_prepare_data": "mace --engine v1 data prepare",
    "mace_plot_train": "mace --engine v1 plot training",
    "mace_active_learning_md": None,
    "mace_polar_density_cube": "mace --engine v1 eval density-cube",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Legacy command after --, or a legacy training YAML file.",
    )


def run(args: argparse.Namespace) -> int:
    import yaml
    from mace_core.config import read_config_file
    from mace_core.config.resolved import ResolvedConfig

    from mace_torch.cli.legacy_flags import add_arguments as train_arguments
    from mace_torch.cli.training_config import legacy_document

    tokens = list(args.command)
    if tokens[:1] == ["--"]:
        tokens.pop(0)
    if not tokens:
        raise ValueError("give a legacy command line or YAML path after --")
    script = Path(tokens.pop(0)).name
    if script.endswith((".yaml", ".yml")):
        document = legacy_document(read_config_file(args.command[-1]))
        command = COMMAND_MAP["mace_run_train"]
    else:
        if script not in COMMAND_MAP:
            raise ValueError(f"unknown legacy command {script!r}")
        command = COMMAND_MAP[script]
        if command is None and script == "mace_active_learning_md":
            print(
                "Use ASE dynamics with MACECalculator's committee mean and "
                "variance; the MD driver was retired from V1."
            )
            return 0
        if command is None:
            print(
                f"{script} is unnecessary on V1: weights are canonical and the"
                f" backend is selected at build time."
            )
            return 0
        if script != "mace_run_train":
            if script == "mace_finetuning_select":
                document, selected_output = selection_document(tokens)
                config = ResolvedConfig.from_dict(document)
                args.output.write_text(
                    yaml.safe_dump(config.to_user_dict(), sort_keys=False)
                )
                print(
                    f"{command} --config {shlex.quote(str(args.output))} "
                    f"--head selected --output {shlex.quote(selected_output)}"
                )
                return 0
            args.output.write_text(
                yaml.safe_dump(
                    {"command": command, "arguments": tokens}, sort_keys=False
                )
            )
            print(command + " " + shlex.join(tokens))
            return 0
        parser = argparse.ArgumentParser(allow_abbrev=False)
        train_arguments(parser)
        parsed = vars(parser.parse_args(tokens))
        file = parsed.pop("config", None)
        document = legacy_document(
            {**(read_config_file(file) if file else {}), **parsed}
        )
    config = ResolvedConfig.from_dict(document)
    payload = (
        config.to_user_dict()
        if config.finetune.foundation_model
        else config.to_resolved_dict()
    )
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"{command} --config {shlex.quote(str(args.output))}")
    return 0


def selection_document(tokens: list[str]) -> tuple[dict, str]:
    """The retired standalone selector's flags become ordinary head settings."""
    import ast

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--configs_pt", required=True)
    parser.add_argument("--configs_ft")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--subselect", choices=("fps", "random"), default="fps")
    parser.add_argument("--model", default="small")
    parser.add_argument("--descriptors")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--default_dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--head_pt")
    parser.add_argument("--head_ft")
    parser.add_argument("--filtering_type", default="none")
    parser.add_argument("--filter_atomic_numbers_pt")
    parser.add_argument("--weight_pt", type=float, default=1.0)
    parser.add_argument("--weight_ft", type=float, default=1.0)
    parser.add_argument(
        "--disallow_random_padding", action="store_false", dest="padding"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(tokens)
    if args.descriptors:
        raise ValueError(
            "cached descriptors are recomputed from the selected V1 foundation model; "
            "remove --descriptors after converting that model"
        )
    selected = {
        "train_file": args.configs_pt,
        "weight": args.weight_pt,
        "e0s": {"kind": "foundation"},
        "readout_from": args.head_pt,
        "subselect": {
            "method": args.subselect,
            "num_samples": args.num_samples,
            "filtering": args.filtering_type,
            "elements": ast.literal_eval(args.filter_atomic_numbers_pt)
            if args.filter_atomic_numbers_pt
            else [],
            "allow_random_padding": args.padding,
        },
    }
    heads = {"selected": selected}
    if args.configs_ft:
        paths = (
            ast.literal_eval(args.configs_ft)
            if args.configs_ft.startswith("[")
            else [args.configs_ft]
        )
        for index, path in enumerate(paths):
            heads[f"fine_tune_{index}"] = {
                "train_file": path,
                "weight": args.weight_ft,
                "readout_from": args.head_ft,
                "e0s": {"kind": "foundation"},
            }
    return {
        "runtime": {"seed": args.seed, "device": args.device},
        "precision": {"model": args.default_dtype},
        "finetune": {"foundation_model": args.model},
        "data": {"heads": heads},
    }, args.output
