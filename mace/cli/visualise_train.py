import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed
from torchmetrics import Metric

from mace.tools.utils import filter_nonzero_weight, fold_polarization

plt.rcParams.update({"font.size": 8})
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)  # Only show WARNING and above

colors = [
    "#1f77b4",  # muted blue
    "#d62728",  # brick red
    "#7f7f7f",  # middle gray
    "#2ca02c",  # cooked asparagus green
    "#ff7f0e",  # safety orange
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]

error_type: Dict[str, Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]] = {
    "TotalRMSE": (
        [("rmse_e", "RMSE E [meV]"), ("rmse_f", "RMSE F [meV / A]")],
        [("energy", "Energy per atom [eV]"), ("force", "Force [eV / A]")],
    ),
    "PerAtomRMSE": (
        [("rmse_e_per_atom", "RMSE E/atom [meV]"), ("rmse_f", "RMSE F [meV / A]")],
        [("energy", "Energy per atom [eV]"), ("force", "Force [eV / A]")],
    ),
    "PerAtomRMSEstressvirials": (
        [
            ("rmse_e_per_atom", "RMSE E/atom [meV]"),
            ("rmse_f", "RMSE F [meV / A]"),
            ("rmse_stress", "RMSE Stress [meV / A^3]"),
        ],
        [
            ("energy", "Energy per atom [eV]"),
            ("force", "Force [eV / A]"),
            ("stress", "Stress [eV / A^3]"),
        ],
    ),
    "PerAtomMAEstressvirials": (
        [
            ("mae_e_per_atom", "MAE E/atom [meV]"),
            ("mae_f", "MAE F [meV / A]"),
            ("mae_stress", "MAE Stress [meV / A^3]"),
        ],
        [
            ("energy", "Energy per atom [eV]"),
            ("force", "Force [eV / A]"),
            ("stress", "Stress [eV / A^3]"),
        ],
    ),
    "TotalMAE": (
        [("mae_e", "MAE E [meV]"), ("mae_f", "MAE F [meV / A]")],
        [("energy", "Energy per atom [eV]"), ("force", "Force [eV / A]")],
    ),
    "PerAtomMAE": (
        [("mae_e_per_atom", "MAE E/atom [meV]"), ("mae_f", "MAE F [meV / A]")],
        [("energy", "Energy per atom [eV]"), ("force", "Force [eV / A]")],
    ),
    "DipoleRMSE": (
        [
            ("rmse_mu_per_atom", "RMSE MU/atom [mDebye]"),
            ("rel_rmse_f", "Relative MU RMSE [%]"),
        ],
        [("dipole", "Dipole per atom [Debye]")],
    ),
    "DipoleMAE": (
        [("mae_mu", "MAE MU [mDebye]"), ("rel_mae_f", "Relative MU MAE [%]")],
        [("dipole", "Dipole per atom [Debye]")],
    ),
    "DipolePolarRMSE": (
        [
            ("rmse_mu_per_atom", "RMSE MU/atom [me AA]"),
            ("rmse_alpha_per_atom", "RMSE ALPHA/atom [me AA^2/V]"),
            ("rel_rmse_f", "Relative MU RMSE [%]"),
            ("rmse_polarizability_per_atom", "Relative ALPHA RMSE [%]"),
        ],
        [
            ("dipole", "Dipole per atom [me AA]]"),
            ("polarizability", "Polarizability per atom [e AA^2/V]"),
        ],
    ),
    "EnergyDipoleRMSE": (
        [
            ("rmse_e_per_atom", "RMSE E/atom [meV]"),
            ("rmse_f", "RMSE F [meV / A]"),
            ("rmse_mu_per_atom", "RMSE MU/atom [mDebye]"),
        ],
        [
            ("energy", "Energy per atom [eV]"),
            ("force", "Force [eV / A]"),
            ("dipole", "Dipole per atom [Debye]"),
        ],
    ),
    "PerAtomFieldRMSE": (
        [
            ("rmse_e_per_atom", "RMSE E/atom [meV]"),
            ("rmse_f", "RMSE F [meV / A]"),
            ("rmse_stress", "RMSE Stress [meV / A^3]"),
            ("rmse_polarization", "RMSE P [me / A^2]"),
            ("rmse_becs", "RMSE Z* [e]"),
            ("rmse_polarizability", "RMSE a [e / V / A]"),
        ],
        [
            ("energy", "Energy per atom [eV]"),
            ("force", "Force [eV / A]"),
            ("stress", "Stress [eV / A^3]"),
            ("polarization", "Polarization [e / A^2]"),
            ("becs", "Born Effective Charges [e]"),
            ("polarizability", "Polarizability [ε0]"),
        ],
    ),
}


def _coerce_int(x: Union[str, int, float]) -> Optional[int]:
    try:
        return int(x)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _available_metric_names(df: pd.DataFrame) -> set:
    names = set()
    for c in df.columns:
        if isinstance(c, tuple) and len(c) > 0:
            names.add(c[0])
        elif isinstance(c, str):
            names.add(c)
    return names


def _resolve_metric_key(df: pd.DataFrame, key: str) -> Optional[str]:
    """
    Resolve a desired metric key against what's actually present in the aggregated dataframe.

    Handles:
      - exact match
      - common aliases (rmse_becs -> rmse_bec, polarisation spelling, etc.)
      - simple plural/singular fallback
      - substring fallback (very conservative)
    """
    available = _available_metric_names(df)

    if key in available:
        return key

    # Common aliases you might hit across branches/loggers
    aliases = {
        "rmse_becs": [
            "rmse_bec",
            "rmse_zstar",
            "rmse_Zstar",
            "rmse_born_effective_charges",
        ],
        "mae_becs": ["mae_bec", "mae_zstar", "mae_Zstar", "mae_born_effective_charges"],
        "rmse_polarization": ["rmse_polarisation", "rmse_pol", "rmse_P"],
        "mae_polarization": ["mae_polarisation", "mae_pol", "mae_P"],
        "rmse_polarizability": ["rmse_polarisability", "rmse_alpha", "rmse_alphas"],
        "mae_polarizability": ["mae_polarisability", "mae_alpha", "mae_alphas"],
    }
    for cand in aliases.get(key, []):
        if cand in available:
            return cand

    # Singular/plural tweak (becs vs bec, etc.)
    if key.endswith("s") and key[:-1] in available:
        return key[:-1]
    if key + "s" in available:
        return key + "s"

    # Conservative substring fallback
    # (only if it matches exactly one candidate)
    hits = [a for a in available if isinstance(a, str) and (key in a or a in key)]
    if len(hits) == 1:
        return hits[0]

    return None


class TrainingPlotter:
    def __init__(
        self,
        results_dir: str,
        heads: List[str],
        table_type: str,
        train_valid_data: Dict,
        test_data: Dict,
        output_args: Union[str, Dict[str, bool]],
        device: str,
        plot_frequency: int,
        distributed: bool = False,
        swa_start: Optional[int] = None,
        plot_interaction_e: bool = False,
    ):
        self.results_dir = results_dir
        self.heads = heads
        self.table_type = table_type
        self.train_valid_data = train_valid_data
        self.test_data = test_data
        self.output_args = self._parse_output_args(output_args)
        self.device = device
        self.plot_frequency = plot_frequency
        self.distributed = distributed
        self.swa_start = swa_start
        self.plot_interaction_e = plot_interaction_e

    @staticmethod
    def _parse_output_args(output_args: Union[str, Dict[str, bool]]) -> Dict[str, bool]:
        if isinstance(output_args, dict):
            return output_args
        if isinstance(output_args, str):
            try:
                parsed = json.loads(output_args)
                if isinstance(parsed, dict):
                    return {k: bool(v) for k, v in parsed.items()}
            except Exception:  # pylint: disable=broad-exception-caught
                logging.debug(
                    "Could not json-parse output_args string; defaulting to empty dict."
                )
        return {}

    def plot(
        self, model_epoch: Union[str, int], model: torch.nn.Module, rank: int
    ) -> None:
        # All ranks process data through model_inference
        train_valid_dict = model_inference(
            self.train_valid_data,
            model,
            self.output_args,
            self.device,
            self.distributed,
        )
        test_dict = model_inference(
            self.test_data,
            model,
            self.output_args,
            self.device,
            self.distributed,
        )

        # Only rank 0 creates and saves plots
        if rank != 0:
            return

        data = pd.DataFrame(
            results for results in parse_training_results(self.results_dir)
        )
        labels, quantities = error_type[self.table_type]

        model_epoch_int = _coerce_int(model_epoch)
        model_epoch_for_plot = (
            model_epoch_int if model_epoch_int is not None else model_epoch
        )

        for head in self.heads:
            fig = plt.figure(layout="constrained", figsize=(len(quantities) * 3, 6))
            fig.suptitle(
                f"Model loaded from epoch {model_epoch} ({head} head)", fontsize=16
            )

            subfigs = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.05)
            axsTop = subfigs[0].subplots(1, 2, sharey=False)
            axsBottom = subfigs[1].subplots(1, len(quantities), sharey=False)

            plot_epoch_dependence(axsTop, data, head, model_epoch_for_plot, labels)

            plot_inference_from_results(
                axsBottom,
                train_valid_dict,
                test_dict,
                head,
                quantities,
                plot_interaction_e=self.plot_interaction_e,
            )

            if self.swa_start is not None:
                for ax in axsTop:
                    ax.axvline(
                        self.swa_start,
                        color="black",
                        linestyle="dashed",
                        linewidth=1,
                        alpha=0.6,
                        label="Stage Two Starts",
                    )
                # Stage is determined by the loaded model epoch vs swa_start
                if model_epoch_int is not None and model_epoch_int >= int(
                    self.swa_start
                ):
                    stage = "stage_two"
                else:
                    stage = "stage_one"
            else:
                stage = "stage_one"

            axsTop[0].legend(loc="best")

            filename = f"{self.results_dir[:-4]}_{head}_{stage}.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)


def parse_training_results(path: str) -> List[dict]:
    results: List[dict] = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if isinstance(d, dict):
                    results.append(d)
            except json.JSONDecodeError:
                logging.debug("Skipping invalid JSONL line in %s: %s", path, line)
    return results


def plot_epoch_dependence(
    axes: np.ndarray,
    data: pd.DataFrame,
    head: str,
    model_epoch: Union[int, str],
    labels: List[Tuple[str, str]],
) -> None:
    if data.empty:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No training log data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.axis("off")
        return

    # Ensure epoch is numeric where possible
    if "epoch" in data.columns:
        data = data.copy()
        data["epoch"] = pd.to_numeric(data["epoch"], errors="coerce")

    # Validation rows
    if "mode" in data.columns:
        valid_df = data[data["mode"] == "eval"]
        train_df = data[data["mode"] == "opt"]
    else:
        valid_df = data
        train_df = data

    group_valid_keys = ["epoch"]
    if "mode" in valid_df.columns:
        group_valid_keys = ["mode", "epoch"]
    if "head" in valid_df.columns:
        group_valid_keys.append("head")

    valid_data = valid_df.groupby(group_valid_keys).agg(["mean", "std"]).reset_index()

    if "head" in valid_data.columns and head is not None:
        try:
            valid_data = valid_data[valid_data["head"] == head]
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    group_train_keys = ["epoch"]
    if "mode" in train_df.columns:
        group_train_keys = ["mode", "epoch"]
    train_data = train_df.groupby(group_train_keys).agg(["mean", "std"]).reset_index()

    # ---- Plot loss ----
    ax = axes[0]
    if ("loss" in _available_metric_names(train_data)) and (
        "loss" in _available_metric_names(valid_data)
    ):
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
        ax2.set_yscale("log")
    else:
        ax.text(
            0.5,
            0.5,
            "Loss not found in log",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    if model_epoch is not None:
        ax.axvline(
            model_epoch,
            color="black",
            linestyle="solid",
            linewidth=1,
            alpha=0.8,
            label="Loaded Model",
        )
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.5)

    # ---- Plot selected keys ----
    ax = axes[1]
    plotted = 0
    for key, axis_label in labels:
        resolved = _resolve_metric_key(valid_data, key)
        if resolved is None:
            logging.debug(
                "Plotting: metric '%s' not found for head '%s' (skipping).", key, head
            )
            continue

        color = colors[(plotted + 3) % len(colors)]
        if plotted == 0:
            main_ax = ax
        else:
            main_ax = ax.twinx()
            main_ax.spines.right.set_position(("outward", 60 * (plotted - 1)))

        try:
            y = valid_data[resolved]["mean"] * 1e3
        except Exception:  # pylint: disable=broad-exception-caught
            logging.debug(
                "Plotting: could not access valid_data[%s]['mean'] (skipping).",
                resolved,
            )
            continue

        main_ax.plot(
            valid_data["epoch"],
            y,
            color=color,
            label=axis_label,
            linewidth=1,
        )
        main_ax.set_yscale("log")
        main_ax.set_ylabel(axis_label, color=color)
        main_ax.tick_params(axis="y", colors=color)
        plotted += 1

    if plotted == 0:
        ax.text(
            0.5,
            0.5,
            "No requested metrics found in log",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    if model_epoch is not None:
        ax.axvline(
            model_epoch,
            color="black",
            linestyle="solid",
            linewidth=1,
            alpha=0.8,
            label="Loaded Model",
        )
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.5)


# INFERENCE=========


def plot_inference_from_results(
    axes: np.ndarray,
    train_valid_dict: dict,
    test_dict: dict,
    head: str,
    quantities: List[Tuple[str, str]],
    plot_interaction_e: bool = False,
) -> None:
    for ax, quantity in zip(axes, quantities):
        key, label = quantity

        legend_labels = {}

        # ---- Train/valid ----
        for name, result in train_valid_dict.items():
            if head not in name:
                continue

            if "train" in name:
                fixed_color = colors[1]
                marker = "x"
            else:
                fixed_color = colors[0]
                marker = "+"

            scatter = None

            if key == "energy" and "energy" in result:
                e_key = "energy" if not plot_interaction_e else "interaction_energy"
                if e_key in result:
                    scatter = ax.scatter(
                        result[e_key]["reference_per_atom"],
                        result[e_key]["predicted_per_atom"],
                        marker=marker,
                        color=fixed_color,
                        label=name,
                    )

            elif key == "force" and "forces" in result:
                scatter = ax.scatter(
                    result["forces"]["reference"],
                    result["forces"]["predicted"],
                    marker=marker,
                    color=fixed_color,
                    label=name,
                )

            elif key == "stress" and "stress" in result:
                scatter = ax.scatter(
                    result["stress"]["reference"],
                    result["stress"]["predicted"],
                    marker=marker,
                    color=fixed_color,
                    label=name,
                )

            elif key == "virials" and "virials" in result:
                scatter = ax.scatter(
                    result["virials"]["reference_per_atom"],
                    result["virials"]["predicted_per_atom"],
                    marker=marker,
                    color=fixed_color,
                    label=name,
                )

            elif key == "dipole" and "dipole" in result:
                scatter = ax.scatter(
                    result["dipole"]["reference_per_atom"],
                    result["dipole"]["predicted_per_atom"],
                    marker=marker,
                    color=fixed_color,
                    label=name,
                )

            elif key == "polarization" and "polarization" in result:
                scatter = ax.scatter(
                    result["polarization"]["reference"],
                    result["polarization"]["predicted"],
                    marker=marker,
                    color=fixed_color,
                    label=name,
                )

            elif key == "becs" and "becs" in result:
                scatter = ax.scatter(
                    result["becs"]["reference"],
                    result["becs"]["predicted"],
                    marker=marker,
                    color=fixed_color,
                    label=name,
                )

            elif key == "polarizability" and "polarizability" in result:
                scatter = ax.scatter(
                    result["polarizability"]["reference"],
                    result["polarizability"]["predicted"],
                    marker=marker,
                    color=fixed_color,
                    label=name,
                )

            if scatter is not None:
                legend_labels[name] = scatter

        # ---- Test ----
        fixed_color_test = colors[2]
        for _, result in test_dict.items():
            scatter = None

            if key == "energy" and "energy" in result:
                e_key = "energy" if not plot_interaction_e else "interaction_energy"
                if e_key in result:
                    scatter = ax.scatter(
                        result[e_key]["reference_per_atom"],
                        result[e_key]["predicted_per_atom"],
                        marker="o",
                        color=fixed_color_test,
                        label="Test",
                    )

            elif key == "force" and "forces" in result:
                scatter = ax.scatter(
                    result["forces"]["reference"],
                    result["forces"]["predicted"],
                    marker="o",
                    color=fixed_color_test,
                    label="Test",
                )

            elif key == "stress" and "stress" in result:
                scatter = ax.scatter(
                    result["stress"]["reference"],
                    result["stress"]["predicted"],
                    marker="o",
                    color=fixed_color_test,
                    label="Test",
                )

            elif key == "virials" and "virials" in result:
                scatter = ax.scatter(
                    result["virials"]["reference_per_atom"],
                    result["virials"]["predicted_per_atom"],
                    marker="o",
                    color=fixed_color_test,
                    label="Test",
                )

            elif key == "dipole" and "dipole" in result:
                scatter = ax.scatter(
                    result["dipole"]["reference_per_atom"],
                    result["dipole"]["predicted_per_atom"],
                    marker="o",
                    color=fixed_color_test,
                    label="Test",
                )

            elif key == "polarization" and "polarization" in result:
                scatter = ax.scatter(
                    result["polarization"]["reference"],
                    result["polarization"]["predicted"],
                    marker="o",
                    color=fixed_color_test,
                    label="Test",
                )

            elif key == "becs" and "becs" in result:
                scatter = ax.scatter(
                    result["becs"]["reference"],
                    result["becs"]["predicted"],
                    marker="o",
                    color=fixed_color_test,
                    label="Test",
                )

            elif key == "polarizability" and "polarizability" in result:
                scatter = ax.scatter(
                    result["polarizability"]["reference"],
                    result["polarizability"]["predicted"],
                    marker="o",
                    color=fixed_color_test,
                    label="Test",
                )

            if scatter is not None:
                legend_labels["Test"] = scatter

            # only plot one test set (your dict is usually single-entry anyway)
            break

        # Diagonal guide
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="black",
            alpha=0.7,
        )

        if legend_labels:
            ax.legend(
                handles=legend_labels.values(), labels=legend_labels.keys(), loc="best"
            )

        if key != "energy" or not plot_interaction_e:
            ax.set_xlabel(f"Reference {label}")
            ax.set_ylabel(f"MACE {label}")
        else:
            ax.set_xlabel(f"Reference Interaction {label}")
            ax.set_ylabel(f"MACE Interaction {label}")
        ax.grid(True, linestyle="--", alpha=0.5)


def model_inference(
    all_data_loaders: dict,
    model: torch.nn.Module,
    output_args: Dict[str, bool],
    device: str,
    distributed: bool = False,
):
    for param in model.parameters():
        param.requires_grad = False

    results_dict = {}

    for name in all_data_loaders:
        data_loader = all_data_loaders[name]
        logging.debug(f"Running inference on {name} dataset")
        scatter_metric = InferenceMetric().to(device)

        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            if (
                output_args.get("polarization", False)
                or output_args.get("becs", False)
                or output_args.get("polarizability", False)
            ):
                output = model(
                    batch_dict,
                    training=True,
                    compute_force=output_args.get("forces", False),
                    compute_virials=output_args.get("virials", False),
                    compute_stress=output_args.get("stress", False),
                    compute_polarization=output_args.get("polarization", False),
                    compute_becs=output_args.get("becs", False),
                    compute_polarizability=output_args.get("polarizability", False),
                )
            else:
                output = model(
                    batch_dict,
                    training=False,
                    compute_force=output_args.get("forces", False),
                    compute_virials=output_args.get("virials", False),
                    compute_stress=output_args.get("stress", False),
                )

            scatter_metric(batch, output)

        if distributed:
            torch.distributed.barrier()

        results = scatter_metric.compute()
        results_dict[name] = results
        scatter_metric.reset()

        del data_loader

    for param in model.parameters():
        param.requires_grad = True

    return results_dict


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


class InferenceMetric(Metric):
    """Metric class for collecting reference and predicted values for scatterplot visualization."""

    def __init__(self):
        super().__init__()
        # Raw values
        self.add_state("ref_energies", default=[], dist_reduce_fx="cat")
        self.add_state("ref_interaction_energies", default=[], dist_reduce_fx="cat")
        self.add_state("pred_energies", default=[], dist_reduce_fx="cat")
        self.add_state("pred_interaction_energies", default=[], dist_reduce_fx="cat")
        self.add_state("ref_forces", default=[], dist_reduce_fx="cat")
        self.add_state("pred_forces", default=[], dist_reduce_fx="cat")
        self.add_state("ref_stress", default=[], dist_reduce_fx="cat")
        self.add_state("pred_stress", default=[], dist_reduce_fx="cat")
        self.add_state("ref_virials", default=[], dist_reduce_fx="cat")
        self.add_state("pred_virials", default=[], dist_reduce_fx="cat")
        self.add_state("ref_dipole", default=[], dist_reduce_fx="cat")
        self.add_state("pred_dipole", default=[], dist_reduce_fx="cat")
        self.add_state("ref_polarization", default=[], dist_reduce_fx="cat")
        self.add_state("pred_polarization", default=[], dist_reduce_fx="cat")
        self.add_state("ref_becs", default=[], dist_reduce_fx="cat")
        self.add_state("pred_becs", default=[], dist_reduce_fx="cat")
        self.add_state("ref_polarizability", default=[], dist_reduce_fx="cat")
        self.add_state("pred_polarizability", default=[], dist_reduce_fx="cat")

        # Per-atom normalized values
        self.add_state("ref_energies_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "ref_interaction_energies_per_atom", default=[], dist_reduce_fx="cat"
        )
        self.add_state("pred_energies_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "pred_interaction_energies_per_atom", default=[], dist_reduce_fx="cat"
        )
        self.add_state("ref_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("pred_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("ref_dipole_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("pred_dipole_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("ref_polarization_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("pred_polarization_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("ref_polarizability_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("pred_polarizability_per_atom", default=[], dist_reduce_fx="cat")

        self.add_state("atom_counts", default=[], dist_reduce_fx="cat")

        # Counters
        self.add_state("n_energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_interaction_energy", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("n_forces", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_stress", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_virials", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_dipole", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_polarization", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("n_becs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_polarizability", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, batch, output):  # pylint: disable=arguments-differ
        atoms_per_config = batch.ptr[1:] - batch.ptr[:-1]
        self.atom_counts.append(atoms_per_config)

        # Energy
        if output.get("energy") is not None and batch.energy is not None:
            self.ref_energies.append(batch.energy)
            self.pred_energies.append(output["energy"])
            self.ref_energies_per_atom.append(batch.energy / atoms_per_config)
            self.pred_energies_per_atom.append(output["energy"] / atoms_per_config)

            self.n_energy += filter_nonzero_weight(
                batch, self.ref_energies, batch.weight, batch.energy_weight
            )
            filter_nonzero_weight(
                batch, self.pred_energies, batch.weight, batch.energy_weight
            )
            filter_nonzero_weight(
                batch, self.ref_energies_per_atom, batch.weight, batch.energy_weight
            )
            filter_nonzero_weight(
                batch, self.pred_energies_per_atom, batch.weight, batch.energy_weight
            )

        if output.get("interaction_energy") is not None and batch.energy is not None:
            E0s = output["energy"].to(torch.float64) - output["interaction_energy"].to(
                torch.float64
            )
            self.ref_interaction_energies.append(batch.energy - E0s)
            self.pred_interaction_energies.append(output["interaction_energy"])
            self.ref_interaction_energies_per_atom.append(
                (batch.energy - E0s) / atoms_per_config
            )
            self.pred_interaction_energies_per_atom.append(
                output["interaction_energy"] / atoms_per_config
            )

            self.n_interaction_energy += filter_nonzero_weight(
                batch, self.ref_interaction_energies, batch.weight, batch.energy_weight
            )
            filter_nonzero_weight(
                batch, self.pred_interaction_energies, batch.weight, batch.energy_weight
            )
            filter_nonzero_weight(
                batch,
                self.ref_interaction_energies_per_atom,
                batch.weight,
                batch.energy_weight,
            )
            filter_nonzero_weight(
                batch,
                self.pred_interaction_energies_per_atom,
                batch.weight,
                batch.energy_weight,
            )

        # Forces
        if output.get("forces") is not None and batch.forces is not None:
            self.ref_forces.append(batch.forces)
            self.pred_forces.append(output["forces"])

            self.n_forces += filter_nonzero_weight(
                batch,
                self.ref_forces,
                batch.weight,
                batch.forces_weight,
                spread_atoms=True,
            )
            filter_nonzero_weight(
                batch,
                self.pred_forces,
                batch.weight,
                batch.forces_weight,
                spread_atoms=True,
            )

        # Stress
        if output.get("stress") is not None and batch.stress is not None:
            self.ref_stress.append(batch.stress)
            self.pred_stress.append(output["stress"])

            self.n_stress += filter_nonzero_weight(
                batch, self.ref_stress, batch.weight, batch.stress_weight
            )
            filter_nonzero_weight(
                batch, self.pred_stress, batch.weight, batch.stress_weight
            )

        # Virials
        if output.get("virials") is not None and batch.virials is not None:
            self.ref_virials.append(batch.virials)
            self.pred_virials.append(output["virials"])
            atoms_per_config_3d = atoms_per_config.view(-1, 1, 1)
            self.ref_virials_per_atom.append(batch.virials / atoms_per_config_3d)
            self.pred_virials_per_atom.append(output["virials"] / atoms_per_config_3d)

            self.n_virials += filter_nonzero_weight(
                batch, self.ref_virials, batch.weight, batch.virials_weight
            )
            filter_nonzero_weight(
                batch, self.pred_virials, batch.weight, batch.virials_weight
            )
            filter_nonzero_weight(
                batch, self.ref_virials_per_atom, batch.weight, batch.virials_weight
            )
            filter_nonzero_weight(
                batch, self.pred_virials_per_atom, batch.weight, batch.virials_weight
            )

        # Dipole
        if output.get("dipole") is not None and batch.dipole is not None:
            self.ref_dipole.append(batch.dipole)
            self.pred_dipole.append(output["dipole"])
            atoms_per_config_3d = atoms_per_config.view(-1, 1)
            self.ref_dipole_per_atom.append(batch.dipole / atoms_per_config_3d)
            self.pred_dipole_per_atom.append(output["dipole"] / atoms_per_config_3d)

            self.n_dipole += filter_nonzero_weight(
                batch, self.ref_dipole, batch.weight, batch.dipole_weight, "config"
            )
            filter_nonzero_weight(
                batch, self.pred_dipole, batch.weight, batch.dipole_weight, "config"
            )
            filter_nonzero_weight(
                batch,
                self.ref_dipole_per_atom,
                batch.weight,
                batch.dipole_weight,
                spread_quantity_vector=False,
            )
            filter_nonzero_weight(
                batch,
                self.pred_dipole_per_atom,
                batch.weight,
                batch.dipole_weight,
                spread_quantity_vector=False,
            )

        # Polarization
        if output.get("polarization") is not None and batch.polarization is not None:
            polarization_difference, _ = fold_polarization(
                output["polarization"], batch.polarization, batch.cell
            )
            self.ref_polarization.append(batch.polarization)
            self.pred_polarization.append(polarization_difference + batch.polarization)
            atoms_per_config_3d = atoms_per_config.view(-1, 1)
            self.ref_polarization_per_atom.append(
                batch.polarization / atoms_per_config_3d
            )
            self.pred_polarization_per_atom.append(
                (polarization_difference + batch.polarization) / atoms_per_config_3d
            )

            self.n_polarization += filter_nonzero_weight(
                batch,
                self.ref_polarization,
                batch.weight,
                batch.polarization_weight,
                spread_quantity_vector=False,
            )
            filter_nonzero_weight(
                batch,
                self.pred_polarization,
                batch.weight,
                batch.polarization_weight,
                spread_quantity_vector=False,
            )
            filter_nonzero_weight(
                batch,
                self.ref_polarization_per_atom,
                batch.weight,
                batch.polarization_weight,
                spread_quantity_vector=False,
            )
            filter_nonzero_weight(
                batch,
                self.pred_polarization_per_atom,
                batch.weight,
                batch.polarization_weight,
                spread_quantity_vector=False,
            )

        # Born effective charges
        if output.get("becs") is not None and batch.becs is not None:
            self.ref_becs.append(batch.becs)
            self.pred_becs.append(output["becs"])

            self.n_becs += filter_nonzero_weight(
                batch,
                self.ref_becs,
                batch.weight,
                batch.becs_weight,
                spread_atoms=True,
                spread_quantity_vector=False,
            )
            filter_nonzero_weight(
                batch,
                self.pred_becs,
                batch.weight,
                batch.becs_weight,
                spread_atoms=True,
                spread_quantity_vector=False,
            )

        # Polarizability
        if (
            output.get("polarizability") is not None
            and batch.polarizability is not None
        ):
            self.ref_polarizability.append(batch.polarizability)
            self.pred_polarizability.append(output["polarizability"])
            atoms_per_config_3d = atoms_per_config.view(-1, 1, 1)
            self.ref_polarizability_per_atom.append(
                batch.polarizability / atoms_per_config_3d
            )
            self.pred_polarizability_per_atom.append(
                output["polarizability"] / atoms_per_config_3d
            )

            self.n_polarizability += filter_nonzero_weight(
                batch,
                self.ref_polarizability,
                batch.weight,
                batch.polarizability_weight,
                spread_quantity_vector=False,
            )
            filter_nonzero_weight(
                batch,
                self.pred_polarizability,
                batch.weight,
                batch.polarizability_weight,
                spread_quantity_vector=False,
            )
            filter_nonzero_weight(
                batch,
                self.ref_polarizability_per_atom,
                batch.weight,
                batch.polarizability_weight,
                spread_quantity_vector=False,
            )
            filter_nonzero_weight(
                batch,
                self.pred_polarizability_per_atom,
                batch.weight,
                batch.polarizability_weight,
                spread_quantity_vector=False,
            )

    def _process_data(self, ref_list, pred_list):
        if isinstance(ref_list, (list, tuple)):
            if len(ref_list) == 0:
                return None, None
            ref = torch.cat(ref_list).reshape(-1)
            pred = torch.cat(pred_list).reshape(-1)
        elif isinstance(ref_list, torch.Tensor):
            ref = ref_list.reshape(-1)
            pred = pred_list.reshape(-1)
        else:
            return None, None
        return to_numpy(ref), to_numpy(pred)

    def compute(self):
        results = {}

        if self.n_energy.item() > 0:
            ref_e, pred_e = self._process_data(self.ref_energies, self.pred_energies)
            ref_e_pa, pred_e_pa = self._process_data(
                self.ref_energies_per_atom, self.pred_energies_per_atom
            )
            results["energy"] = {
                "reference": ref_e,
                "predicted": pred_e,
                "reference_per_atom": ref_e_pa,
                "predicted_per_atom": pred_e_pa,
            }

        if self.n_interaction_energy.item() > 0:
            ref_ie, pred_ie = self._process_data(
                self.ref_interaction_energies, self.pred_interaction_energies
            )
            ref_ie_pa, pred_ie_pa = self._process_data(
                self.ref_interaction_energies_per_atom,
                self.pred_interaction_energies_per_atom,
            )
            results["interaction_energy"] = {
                "reference": ref_ie,
                "predicted": pred_ie,
                "reference_per_atom": ref_ie_pa,
                "predicted_per_atom": pred_ie_pa,
            }

        if self.n_forces.item() > 0:
            ref_f, pred_f = self._process_data(self.ref_forces, self.pred_forces)
            results["forces"] = {"reference": ref_f, "predicted": pred_f}

        if self.n_stress.item() > 0:
            ref_s, pred_s = self._process_data(self.ref_stress, self.pred_stress)
            results["stress"] = {"reference": ref_s, "predicted": pred_s}

        if self.n_virials.item() > 0:
            ref_v, pred_v = self._process_data(self.ref_virials, self.pred_virials)
            ref_v_pa, pred_v_pa = self._process_data(
                self.ref_virials_per_atom, self.pred_virials_per_atom
            )
            results["virials"] = {
                "reference": ref_v,
                "predicted": pred_v,
                "reference_per_atom": ref_v_pa,
                "predicted_per_atom": pred_v_pa,
            }

        if self.n_dipole.item() > 0:
            ref_d, pred_d = self._process_data(self.ref_dipole, self.pred_dipole)
            ref_d_pa, pred_d_pa = self._process_data(
                self.ref_dipole_per_atom, self.pred_dipole_per_atom
            )
            results["dipole"] = {
                "reference": ref_d,
                "predicted": pred_d,
                "reference_per_atom": ref_d_pa,
                "predicted_per_atom": pred_d_pa,
            }

        if self.n_polarization.item() > 0:
            ref_p, pred_p = self._process_data(
                self.ref_polarization, self.pred_polarization
            )
            ref_p_pa, pred_p_pa = self._process_data(
                self.ref_polarization_per_atom, self.pred_polarization_per_atom
            )
            results["polarization"] = {
                "reference": ref_p,
                "predicted": pred_p,
                "reference_per_atom": ref_p_pa,
                "predicted_per_atom": pred_p_pa,
            }

        if self.n_becs.item() > 0:
            ref_b, pred_b = self._process_data(self.ref_becs, self.pred_becs)
            results["becs"] = {"reference": ref_b, "predicted": pred_b}

        if self.n_polarizability.item() > 0:
            ref_a, pred_a = self._process_data(
                self.ref_polarizability, self.pred_polarizability
            )
            ref_a_pa, pred_a_pa = self._process_data(
                self.ref_polarizability_per_atom, self.pred_polarizability_per_atom
            )
            results["polarizability"] = {
                "reference": ref_a,
                "predicted": pred_a,
                "reference_per_atom": ref_a_pa,
                "predicted_per_atom": pred_a_pa,
            }

        return results
