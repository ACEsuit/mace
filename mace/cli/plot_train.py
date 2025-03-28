import argparse
import dataclasses
import glob
import json
import os
import re
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 8})
plt.style.use("seaborn-v0_8-paper")


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
        "--path", help="Path to results file (.txt) or directory.", required=True
    )
    parser.add_argument(
        "--min_epoch", help="Minimum epoch.", default=0, type=int, required=False
    )
    parser.add_argument(
        "--start_stage_two",
        "--start_swa",
        help="Epoch that stage two (swa) loss began. Plots dashed line on plot to indicate. If None then assumed tag not used in training.",
        default=None,
        type=int,
        required=False,
        dest="start_swa",
    )
    parser.add_argument(
        "--linear",
        help="Whether to plot linear instead of log scales.",
        default=False,
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--error_bars",
        help="Whether to plot standard deviations.",
        default=False,
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--keys",
        help="Comma-separated list of keys to plot.",
        default="rmse_e,rmse_f",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--output_format",
        help="What file type to save plot as",
        default="png",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--heads",
        help="Comma-separated name of the heads used for multihead training",
        default=None,
        type=str,
        required=False,
    )

    return parser.parse_args()


def plot(
    data: pd.DataFrame,
    min_epoch: int,
    output_path: str,
    output_format: str,
    linear: bool,
    start_swa: int,
    error_bars: bool,
    keys: str,
    heads: str,
) -> None:
    """
    Plots train,validation loss and errors as a function of epoch.
    min_epoch: minimum epoch to plot.
    output_path: path to save the plot.
    output_format: format to save the plot.
    start_swa: whether to plot a dashed line to show epoch when stage two loss (swa) begins.
    error_bars: whether to plot standard deviation of loss.
    linear: whether to plot in linear scale or logscale (default).
    keys: Values to plot.
    heads: Heads used for multihead training.
    """

    labels = {
        "mae_e": "MAE E [meV]",
        "mae_e_per_atom": "MAE E/atom [meV]",
        "rmse_e": "RMSE E [meV]",
        "rmse_e_per_atom": "RMSE E/atom [meV]",
        "q95_e": "Q95 E [meV]",
        "mae_f": "MAE F [meV / A]",
        "rel_mae_f": "Relative MAE F [meV / A]",
        "rmse_f": "RMSE F [meV / A]",
        "rel_rmse_f": "Relative RMSE F [meV / A]",
        "q95_f": "Q95 F [meV / A]",
        "mae_stress": "MAE Stress",
        "rmse_stress": "RMSE Stress [meV / A^3]",
        "rmse_virials_per_atom": " RMSE virials/atom [meV]",
        "mae_virials": "MAE Virials [meV]",
        "rmse_mu_per_atom": "RMSE MU/atom [mDebye]",
    }

    data = data[data["epoch"] > min_epoch]
    if heads is None:
        data = (
            data.groupby(["name", "mode", "epoch"]).agg(["mean", "std"]).reset_index()
        )

        valid_data = data[data["mode"] == "eval"]
        valid_data_dict = {"default": valid_data}
        train_data = data[data["mode"] == "opt"]
    else:
        heads = heads.split(",")
        # Separate eval and opt data
        valid_data = (
            data[data["mode"] == "eval"]
            .groupby(["name", "mode", "epoch", "head"])
            .agg(["mean", "std"])
            .reset_index()
        )
        train_data = (
            data[data["mode"] == "opt"]
            .groupby(["name", "mode", "epoch"])
            .agg(["mean", "std"])
            .reset_index()
        )
        valid_data_dict = {
            head: valid_data[valid_data["head"] == head] for head in heads
        }

    for head, valid_data in valid_data_dict.items():
        fig, axes = plt.subplots(
            nrows=1, ncols=2, figsize=(10, 3), constrained_layout=True
        )

        # ---- Plot loss ----
        ax = axes[0]
        ax.plot(
            train_data["epoch"],
            train_data["loss"]["mean"],
            color=colors[1],
            linewidth=1,
        )
        ax.set_ylabel("Training Loss", color=colors[1])
        ax.set_yscale("log")

        ax2 = ax.twinx()
        ax2.plot(
            valid_data["epoch"],
            valid_data["loss"]["mean"],
            color=colors[0],
            linewidth=1,
        )
        ax2.set_ylabel("Validation Loss", color=colors[0])

        if not linear:
            ax.set_yscale("log")
            ax2.set_yscale("log")

        if error_bars:
            ax.fill_between(
                train_data["epoch"],
                train_data["loss"]["mean"] - train_data["loss"]["std"],
                train_data["loss"]["mean"] + train_data["loss"]["std"],
                alpha=0.3,
                color=colors[1],
            )
            ax.fill_between(
                valid_data["epoch"],
                valid_data["loss"]["mean"] - valid_data["loss"]["std"],
                valid_data["loss"]["mean"] + valid_data["loss"]["std"],
                alpha=0.3,
                color=colors[0],
            )

        if start_swa is not None:
            ax.axvline(
                start_swa,
                color="black",
                linestyle="dashed",
                linewidth=1,
                alpha=0.6,
                label="Stage Two Starts",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", fontsize=4)
        ax.grid(True, linestyle="--", alpha=0.5)

        # ---- Plot selected keys ----
        ax = axes[1]
        twin_axes = []
        for i, key in enumerate(keys.split(",")):
            color = colors[(i + 3)]
            label = labels.get(key, key)

            if i == 0:
                main_ax = ax
            else:
                main_ax = ax.twinx()
                main_ax.spines.right.set_position(("outward", 40 * (i - 1)))
                twin_axes.append(main_ax)

            main_ax.plot(
                valid_data["epoch"],
                valid_data[key]["mean"] * 1e3,
                color=color,
                label=label,
                linewidth=1,
            )

            if error_bars:
                main_ax.fill_between(
                    valid_data["epoch"],
                    (valid_data[key]["mean"] - valid_data[key]["std"]) * 1e3,
                    (valid_data[key]["mean"] + valid_data[key]["std"]) * 1e3,
                    alpha=0.3,
                    color=color,
                )

            main_ax.set_ylabel(label, color=color)
            main_ax.tick_params(axis="y", colors=color)

        if start_swa is not None:
            ax.axvline(
                start_swa,
                color="black",
                linestyle="dashed",
                linewidth=1,
                alpha=0.6,
                label="Stage Two Starts",
            )

        ax.set_xlabel("Epoch")
        ax.set_xlim(left=min_epoch)
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.savefig(
            f"{output_path}_{head}.{output_format}", dpi=300, bbox_inches="tight"
        )
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
        plot(
            group,
            min_epoch=args.min_epoch,
            output_path=name,
            output_format=args.output_format,
            linear=args.linear,
            start_swa=args.start_swa,
            error_bars=args.error_bars,
            keys=args.keys,
            heads=args.heads,
        )


if __name__ == "__main__":
    main()
