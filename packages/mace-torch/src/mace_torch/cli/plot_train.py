"""Training curves from JSON-lines results, without framework imports."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min_epoch", type=int, default=0)
    parser.add_argument("--start_stage_two", "--start_swa", type=int, dest="start_swa")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--error_bars", action="store_true")
    parser.add_argument("--keys", default="loss")
    parser.add_argument("--heads")
    parser.add_argument("--output_format", default="png")


def run(args: argparse.Namespace) -> int:
    from mace_torch.train.plots import plot_training

    paths = (
        sorted([*args.path.glob("*.txt"), *args.path.glob("*.jsonl")])
        if args.path.is_dir()
        else [args.path]
    )
    output = args.output or args.path.with_suffix(f".{args.output_format}")
    print(
        plot_training(
            paths,
            output,
            keys=args.keys.split(","),
            min_epoch=args.min_epoch,
            linear=args.linear,
            start_stage_two=args.start_swa,
            heads=args.heads.split(",") if args.heads else None,
            error_bars=args.error_bars,
        )
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mace plot training")
    add_arguments(parser)
    return run(parser.parse_args(argv))
