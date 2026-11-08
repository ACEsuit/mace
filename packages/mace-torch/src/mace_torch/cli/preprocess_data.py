"""Prepare V1 shards through the shared reader, splitter and statistics code."""

from __future__ import annotations

import argparse
import ast
import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from mace_torch.cli.eval_configs import boolean


def add_arguments(parser: argparse.ArgumentParser, *, suppress: bool = True) -> None:
    parser.add_argument("--config")
    for name in (
        "train_file",
        "valid_file",
        "test_file",
        "atomic_numbers",
        "E0s",
        "heads",
    ):
        parser.add_argument(f"--{name}")
    parser.add_argument("--output", "--h5_prefix", dest="output", type=Path)
    parser.add_argument("--work_dir", type=Path, default=Path())
    for name, default in (
        ("num_process", 2),
        ("batch_size", 16),
        ("seed", 123),
        ("shard_size", 1024),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--r_max", type=float, default=5.0)
    parser.add_argument("--valid_fraction", type=float, default=0.1)
    parser.add_argument("--shuffle", type=boolean, default=True)
    parser.add_argument("--compute_statistics", action="store_true")
    parser.add_argument("--scaling", default="rms_forces_scaling")
    parser.add_argument("--config_type_weights", default='{"Default":1.0}')
    for name, default in (
        ("energy", "REF_energy"),
        ("forces", "REF_forces"),
        ("virials", "REF_virials"),
        ("stress", "REF_stress"),
        ("dipole", "dipole"),
        ("polarizability", "polarizability"),
        ("charges", "REF_charges"),
        ("head", "head"),
    ):
        parser.add_argument(f"--{name}_key", default=default)

    if suppress:
        for action in parser._actions:
            if action.dest != "help":
                action.default = argparse.SUPPRESS


def run(args: argparse.Namespace) -> int:
    from mace_core.config import read_config_file

    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser, suppress=False)
    actions = {action.dest: action for action in parser._actions}
    defaults = {name: action.default for name, action in actions.items()}
    path = getattr(args, "config", None)
    document = read_config_file(path) if path else {}
    if "h5_prefix" in document:
        document["output"] = document.pop("h5_prefix")
    unknown = set(document) - set(actions)
    if unknown:
        raise ValueError(f"unknown prepare options: {sorted(unknown)}")
    document = {
        name: cast(Callable[[str], Any], converter)(str(value))
        if callable(converter := actions[name].type)
        else value
        for name, value in document.items()
    }
    if "compute_statistics" in document:
        document["compute_statistics"] = (
            False
            if document["compute_statistics"] is None
            else boolean(str(document["compute_statistics"]))
        )
    args = argparse.Namespace(
        **{
            **defaults,
            **document,
            **{key: value for key, value in vars(args).items() if key in actions},
        }
    )
    import numpy as np
    from mace_core.config.data import HeadDataConfig
    from mace_core.config.e0s import E0sIsolatedAtoms
    from mace_core.config.legacy import from_namespace
    from mace_core.data import open_dataset
    from mace_core.data.backends.memory import InMemoryBackend
    from mace_core.data.e0_resolution import resolve_e0s
    from mace_core.data.keys import KeySpecification
    from mace_core.data.splitting import random_train_valid_split
    from mace_core.data.statistics import compute_statistics
    from mace_core.data.xyz import ISOLATED_ATOM_CONFIG_TYPE
    from mace_core.data_spec.shard_format import write_shards
    from mace_core.elements import AtomicNumberTable

    if args.train_file is None or args.output is None:
        raise ValueError("mace data prepare requires --train_file and --output")
    if args.heads:
        raise ValueError(
            "prepare one dataset per head, then declare those datasets under data.heads"
        )
    if args.head_key != "head":
        raise ValueError("prepare one dataset per head; --head_key is fixed to 'head'")
    for name in ("batch_size", "num_process", "shard_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    if not math.isfinite(args.r_max) or args.r_max <= 0:
        raise ValueError("--r_max must be finite and positive")
    scaling = {
        "no_scaling": "none",
        "rms_forces_scaling": "rms_forces",
        "rms_dipoles_scaling": "rms_dipoles",
        "std_scaling": "std",
    }.get(args.scaling, args.scaling)
    if scaling not in ("none", "std", "rms_forces", "rms_dipoles"):
        raise ValueError(f"--scaling {args.scaling!r} is unknown")
    destination = args.work_dir / args.output
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"{destination} is not empty; prepare into an empty directory"
        )
    keys = KeySpecification.from_defaults().apply_overrides(
        {
            name: value
            for name, value in vars(args).items()
            if name.endswith("_key") and name != "head_key"
        }
    )

    def literal(value):
        return ast.literal_eval(value) if isinstance(value, str) else value

    head = HeadDataConfig.model_validate(
        {
            "train_file": args.train_file,
            "valid_file": args.valid_file,
            "test_file": args.test_file,
            "config_type_weights": literal(args.config_type_weights),
        }
    )

    def read(source):
        items = []
        for path in source if isinstance(source, tuple) else (source,):
            # Preparation accepts ASE-readable files, including .traj, as the
            # original command did. Select that reader explicitly, without
            # adding a fallback to the registry's format discovery.
            backend = open_dataset(
                path, format="xyz", key_spec=keys, keep_isolated_atoms=True
            )
            if not len(backend):
                raise ValueError(f"{path} holds no configurations")
            items.extend(
                replace(
                    item,
                    weight=item.weight
                    * head.config_type_weights.get(item.config_type, 1.0),
                )
                for item in backend.iter_range()
            )
        return items

    configurations = read(head.train_file)
    references = [
        item for item in configurations if item.config_type == ISOLATED_ATOM_CONFIG_TYPE
    ]
    structures = [
        item for item in configurations if item.config_type != ISOLATED_ATOM_CONFIG_TYPE
    ]
    if head.valid_file is not None:
        train, valid = structures, read(head.valid_file)
    else:
        train, valid = random_train_valid_split(
            structures, args.valid_fraction, args.seed, destination
        )
    test = read(head.test_file) if head.test_file is not None else []
    if not train or not valid:
        raise ValueError("preparation requires nonempty train and valid sets")
    numbers = sorted(
        {
            int(z)
            for item in [*configurations, *valid, *test]
            for z in item.atomic_numbers
        }
    )
    if args.atomic_numbers is not None:
        declared = literal(args.atomic_numbers)
        try:
            declared = list(AtomicNumberTable(declared).zs)
        except (TypeError, ValueError) as failure:
            raise ValueError(f"--atomic_numbers: {failure}") from failure
        if not set(numbers) <= set(declared):
            raise ValueError(
                "--atomic_numbers does not cover every element in the data"
            )
        numbers = declared
    spec = E0sIsolatedAtoms()
    if args.E0s is not None:
        spec = HeadDataConfig.model_validate(
            from_namespace(argparse.Namespace(E0s=args.E0s))["data"]["heads"]["default"]
        ).e0s
    # Validation and test labels cannot contribute to fitted reference energies.
    e0s, provenance = resolve_e0s(
        spec, AtomicNumberTable(numbers), [*references, *train]
    )
    if args.shuffle:
        np.random.default_rng(args.seed).shuffle(train)
    statistics = None
    if args.compute_statistics:
        statistics = compute_statistics(
            InMemoryBackend(train),
            numbers,
            args.r_max,
            e0s,
            scaling=scaling,
        )
    splits = {"train": [*references, *train], "valid": valid}
    if test:
        splits["test"] = test
    for name, configurations in splits.items():
        write_shards(
            configurations,
            destination / name,
            shard_size=args.shard_size,
            key_spec=keys,
            statistics=statistics,
            atomic_numbers=numbers,
            r_max=args.r_max,
            e0_provenance=str(provenance),
            processes=args.num_process,
        )
    if statistics is not None:
        (destination / "statistics.json").write_text(statistics.to_json())
    import json

    (destination / "e0s.json").write_text(json.dumps(e0s, indent=2) + "\n")
    print(destination)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mace data prepare", allow_abbrev=False)
    add_arguments(parser)
    return run(parser.parse_args(argv))
