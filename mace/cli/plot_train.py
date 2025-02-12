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

fig_width = 2.5
fig_height = 2.1

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
        "--path", help="Path to results file (.txt) or directory.", required=True
    )
    parser.add_argument(
        "--min_epoch", help="Minimum epoch.", default=0, type=int, required=False
    )
    parser.add_argument(
        "--swa_start", help="Epoch that stage two loss (swa) began. Plots dashed line on plot to indicate. If None then assumed tag not used in training.", default=None, type=int, required=False
    )
    parser.add_argument(
        "--linear", help="Whether to plot linear instead of log scales.", default=False, required=False, action="store_true"
    )
    parser.add_argument(
        "--error_bars", help="Whether to plot standard deviations.", default=False, required=False, action='store_true'
    )
    parser.add_argument(
        "--energy_key", help="Which energy error to plot.", default='rmse_e_per_atom', type=str, required=False,
        choices=['mae_e','mae_e_per_atom','rmse_e','rmse_e_per_atom','q95_e']
    )
    parser.add_argument(
        "--force_key", help="Which force error to plot.", default='rmse_f', type=str, required=False,
        choices=['mae_f','rel_mae_f','rmse_f','rel_rmse_f','q95_f']
    )
    parser.add_argument(
        "--output_format", help="What file type to save plot as", default='pdf', type=str, required=False,
    )
    return parser.parse_args()


def plot(data: pd.DataFrame, 
         min_epoch: int, 
         output_path: str, 
         linear: bool, 
         swa_start: int,
         error_bars: bool,
         energy_key: str,
         force_key: str) -> None:

    """
    Plots train or validation loss and energy/force error as a function of epoch.
    min_epoch: minimum epoch to plot.
    output_path: path to save the plot.
    swa_start: whether to plot a dashed line to show epoch when stage two loss (swa) begins.
    error_bars: whether to plot standard deviation of loss.
    linear: whether to plot in linear scale or logscale (default).
    energy_key: key for energy error in data. Should be "mae_e","mae_e_per_atom", "rmse_e", "rmse_e_per_atom" or "q95_e".
    force_key: key for force error in data. Should be "mae_f","rel_mae_f, "rmse_f", "rel_rmse_f" or "q95_f".      
    """

    labels={"mae_e":("MAE E [eV]",1),
            "mae_e_per_atom":("MAE E/atom [eV]",1),
            "rmse_e":("RMSE E [eV]",1),
            "rmse_e_per_atom":("RMSE E/atom [eV]",1), 
            "q95_e":("Q95 E [eV]",1),
            "mae_f":"MAE F",
            "rel_mae_f":"Relative MAE F", 
            "rmse_f":"RMSE F", 
            "rel_rmse_f":"Relative RMSE F",
            "q95_f":"Q95 F"}

    data = data[data["epoch"] > min_epoch]

    data = data.groupby(["name", "mode", "epoch"]).agg([np.mean, np.std]).reset_index()

    valid_data = data[data["mode"] == "eval"]
    train_data = data[data["mode"] == "opt"]

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(2 * fig_width, fig_height), constrained_layout=True
    )  
    
    #Plot loss in training and validation
    ax = axes[0]
    
    if not linear:
        xmin=min(min(valid_data["loss"]["mean"]),min(train_data["loss"]["mean"]))
        ax.set_yscale('log')
    else:
        xmin=0
    xmax=max(max(train_data["loss"]["mean"]), max(train_data["loss"]["mean"]))

    if swa_start != None:
        ax.vlines(swa_start,xmin,xmax, colors=colors[4], linestyles='dashed', label='SWA begin',alpha=0.5,linewidth=0.75)

    ax.plot(
        train_data["epoch"],
        train_data["loss"]["mean"],
        color=colors[3],
        zorder=1,
        label="Training",
    )
    if error_bars:
        ax.fill_between(
            x=train_data["epoch"],
            y1=train_data["loss"]["mean"] - train_data["loss"]["std"],
            y2=train_data["loss"]["mean"] + train_data["loss"]["std"],
            alpha=0.3,
            zorder=-1,
            color=colors[3],
        )

    #Validation
    ax.plot(
        valid_data["epoch"],
        valid_data["loss"]["mean"],
        color=colors[0],
        zorder=1,
        label="Validation",
    )
    if error_bars:
        ax.fill_between(
            x=valid_data["epoch"],
            y1=valid_data["loss"]["mean"] - valid_data["loss"]["std"],
            y2=valid_data["loss"]["mean"] + valid_data["loss"]["std"],
            alpha=0.5,
            zorder=-1,
            color=colors[0],
        )      

    ax.set_ylabel("Loss")
    if linear:
        ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=0)
    ax.set_xlabel("Epoch")
    
    ax.legend()

    #~~~~~~~~~  Plot energy and force errors  ~~~~~~~~~~~
    
    ax = axes[1]
    if not linear:
        ax.set_yscale('log')
        xmin=min(min(valid_data[force_key]["mean"]),min(valid_data[energy_key]["mean"]))
    else:
        xmin=0

    xmax=max(max(valid_data[force_key]["mean"]),max(valid_data[energy_key]["mean"]))
    if swa_start != None:
        ax.vlines(swa_start,xmin,xmax, colors=colors[4], linestyles='dashed', label='SWA begin',alpha=0.5,linewidth=0.75)
    
    
    ax.plot(
        valid_data["epoch"],
        valid_data[energy_key]["mean"]*labels[energy_key][1],      
        color=colors[1],
        zorder=1,
        label=f"{labels[energy_key][0]}",
    )
    if error_bars:
        ax.fill_between(
            x=valid_data["epoch"],
            y1=valid_data[energy_key]["mean"] - valid_data[energy_key]["std"],
            y2=valid_data[energy_key]["mean"] + valid_data[energy_key]["std"],
            alpha=0.5,
            zorder=-1,
            color=colors[1],
        )
    ax.plot(
        valid_data["epoch"],
        valid_data["mae_f"]["mean"],
        color=colors[2],
        zorder=1,
        label=f"{labels[force_key]} [eV/Å]",
    )
    if error_bars:
        ax.fill_between(
            x=valid_data["epoch"],
            y1=valid_data[force_key]["mean"] - valid_data[force_key]["std"],
            y2=valid_data[force_key]["mean"] + valid_data[force_key]["std"],
            alpha=0.5,
            zorder=-1,
            color=colors[2],
        )

    ax.set_xlim(left=0)
    if linear:
        ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Epoch")
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
        plot(group, min_epoch=args.min_epoch, output_path=f"{name}.{args.output_format}", 
             linear=args.linear, swa_start=args.swa_start, error_bars=args.error_bars,
             energy_key=args.energy_key,force_key=args.force_key)        #pdf


if __name__ == "__main__":
    main()
