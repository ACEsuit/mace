import argparse
import dataclasses
import glob
import json
import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 6})

colors = [
    "#1f77b4",  # muted blue
    "#d62728",  # brick red
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]


@dataclasses.dataclass
class RunInfo:
    name: str
    seed: int

name_re = re.compile(r"(?P<name>.+)_run-(?P<seed>\d+)_train.txt")

def parse_path(path: str) -> RunInfo:
    match = name_re.match(os.path.basename(path))
    if not match:
        raise RuntimeError(f"Cannot parse {path}")

    return RunInfo(name=match.group("name"), seed=int(match.group("seed")))

def parse_training_results(path: str) -> List[dict]:
    run_info = parse_path(path)
    results = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["name"] = run_info.name
            d["seed"] = run_info.seed
            results.append(d)

    return results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mace training statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path", help="path to results file or directory", required=True
    )
    parser.add_argument(
        "--min_epoch", help="minimum epoch", default=0, type=int, required=False
    )
    parser.add_argument(
        "--compute_field", help="compute field", action="store_true", default=False, required=False
    )
    return parser.parse_args()

def plot(data: pd.DataFrame, min_epoch: int, output_path: str, compute_field: bool) -> None:
    data = data[data["epoch"] > min_epoch]

    data = data.groupby(["name", "mode", "epoch"]).agg(["mean", "std"]).reset_index()

    valid_data = data[data["mode"] == "eval"]
    train_data = data[data["mode"] == "opt"]

    fig_width = 2.5
    fig_height = 2.5

    if compute_field:
        ncols = 3
        fig_width *= 3
    else:
        ncols = 2
        fig_width *= 2

    fig, axes = plt.subplots(
        nrows=1, ncols=ncols, figsize=(fig_width, fig_height), constrained_layout=True
    )

    ax = axes[0]
    ax.plot(
        valid_data["epoch"],
        valid_data["loss"]["mean"],
        color=colors[0],
        zorder=1,
        label="Validation",
    )
    ax.fill_between(
        x=valid_data["epoch"],
        y1=valid_data["loss"]["mean"] - valid_data["loss"]["std"],
        y2=valid_data["loss"]["mean"] + valid_data["loss"]["std"],
        alpha=0.5,
        zorder=-1,
        color=colors[0],
    )
    ax.plot(
        train_data["epoch"],
        train_data["loss"]["mean"],
        color=colors[3],
        zorder=1,
        label="Training",
    )
    ax.fill_between(
        x=train_data["epoch"],
        y1=train_data["loss"]["mean"] - train_data["loss"]["std"],
        y2=train_data["loss"]["mean"] + train_data["loss"]["std"],
        alpha=0.5,
        zorder=-1,
        color=colors[3],
    )

    ax.set_ylim(bottom=0.0)    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(
        valid_data["epoch"],
        valid_data["mae_e"]["mean"],
        color=colors[0],
        zorder=1,
        label="Energy [eV]",
    )
    ax.fill_between(
        x=valid_data["epoch"],
        y1=valid_data["mae_e"]["mean"] - valid_data["mae_e"]["std"],
        y2=valid_data["mae_e"]["mean"] + valid_data["mae_e"]["std"],
        alpha=0.5,
        zorder=-1,
        color=colors[0],
    )
    ax.plot(
        valid_data["epoch"],
        valid_data["mae_f"]["mean"],
        color=colors[1],
        zorder=1,
        label="Forces [eV/Å]",
    )
    ax.fill_between(
        x=valid_data["epoch"],
        y1=valid_data["mae_f"]["mean"] - valid_data["mae_f"]["std"],
        y2=valid_data["mae_f"]["mean"] + valid_data["mae_f"]["std"],
        alpha=0.5,
        zorder=-1,
        color=colors[1],
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valid MAE")
    ax.set_yscale("log")
    ax.legend()

    if compute_field:
        ax = axes[2]
        ax.plot(
            valid_data["epoch"],
            valid_data["mae_polarisation"]["mean"],
            color=colors[0],
            zorder=1,
            label="Pol [eV/Å²]",
        )
        ax.fill_between(
            x=valid_data["epoch"],
            y1=valid_data["mae_polarisation"]["mean"] - valid_data["mae_polarisation"]["std"],
            y2=valid_data["mae_polarisation"]["mean"] + valid_data["mae_polarisation"]["std"],
            alpha=0.5,
            zorder=-1,
            color=colors[0],
        )
        ax.plot(
            valid_data["epoch"],
            valid_data["mae_bec"]["mean"],
            color=colors[1],
            zorder=1,
            label="BECs [e]",
        )
        ax.fill_between(
            x=valid_data["epoch"],
            y1=valid_data["mae_bec"]["mean"] - valid_data["mae_bec"]["std"],
            y2=valid_data["mae_bec"]["mean"] + valid_data["mae_bec"]["std"],
            alpha=0.5,
            zorder=-1,
            color=colors[1],
        )
        ax.plot(
            valid_data["epoch"],
            valid_data["mae_polarisability"]["mean"],
            color=colors[3],
            zorder=1,
            label="Alp [Å]",
        )
        ax.fill_between(
            x=valid_data["epoch"],
            y1=valid_data["mae_polarisability"]["mean"] - valid_data["mae_polarisability"]["std"],
            y2=valid_data["mae_polarisability"]["mean"] + valid_data["mae_polarisability"]["std"],
            alpha=0.5,
            zorder=-1,
            color=colors[3],
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Valid MAE")
        ax.set_yscale("log")
        ax.legend()

    fig.savefig(output_path)
    plt.close(fig)


def get_paths(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    paths = glob.glob(os.path.join(path, "*_train.txt"))

    if len(paths) == 0:
        raise RuntimeError(f"Cannot find results in '{path}'")

    return paths


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    data = pd.DataFrame(
        results
        for path in get_paths(args.path)
        for results in parse_training_results(path)
    )

    for name, group in data.groupby("name"):
        plot(group, min_epoch=args.min_epoch, output_path=f"{name}.png", compute_field=args.compute_field)


if __name__ == "__main__":
    main()
