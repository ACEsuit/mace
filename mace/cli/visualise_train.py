import json
import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed
from torchmetrics import Metric

from mace.tools.utils import filter_nonzero_weight

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

error_type = {
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
            ("rmse_polarizability_per_atom", "Relative ALPHA RMSE [%]"),  # check that
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
}


class TrainingPlotter:
    def __init__(
        self,
        results_dir: str,
        heads: List[str],
        table_type: str,
        train_valid_data: Dict,
        test_data: Dict,
        output_args: str,
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
        self.output_args = output_args
        self.device = device
        self.plot_frequency = plot_frequency
        self.distributed = distributed
        self.swa_start = swa_start
        self.plot_interaction_e = plot_interaction_e

    def plot(self, model_epoch: str, model: torch.nn.Module, rank: int) -> None:

        # All ranks process data through model_inference
        train_valid_dict = model_inference(
            self.train_valid_data,
            model,
            self.output_args,
            self.device,
            self.distributed,
        )
        test_dict = model_inference(
            self.test_data, model, self.output_args, self.device, self.distributed
        )

        # Only rank 0 creates and saves plots
        if rank != 0:
            return

        data = pd.DataFrame(
            results for results in parse_training_results(self.results_dir)
        )
        labels, quantities = error_type[self.table_type]

        for head in self.heads:
            fig = plt.figure(layout="constrained", figsize=(10, 6))
            fig.suptitle(
                f"Model loaded from epoch {model_epoch} ({head} head)", fontsize=16
            )

            subfigs = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.05)
            axsTop = subfigs[0].subplots(1, 2, sharey=False)
            axsBottom = subfigs[1].subplots(1, len(quantities), sharey=False)

            plot_epoch_dependence(axsTop, data, head, model_epoch, labels)

            # Use the pre-computed results for plotting
            plot_inference_from_results(
                axsBottom,
                train_valid_dict,
                test_dict,
                head,
                quantities,
                plot_interaction_e=self.plot_interaction_e,
            )

            if self.swa_start is not None:
                # Add vertical lines to both axes
                for ax in axsTop:
                    ax.axvline(
                        self.swa_start,
                        color="black",
                        linestyle="dashed",
                        linewidth=1,
                        alpha=0.6,
                        label="Stage Two Starts",
                    )
                stage = "stage_two" if self.swa_start < model_epoch else "stage_one"
            else:
                stage = "stage_one"
            axsTop[0].legend(loc="best")
            # Save the figure using the appropriate stage in the filename
            filename = f"{self.results_dir[:-4]}_{head}_{stage}.png"

            fig.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)


def parse_training_results(path: str) -> List[dict]:
    results = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())  # Ensure it's valid JSON
                results.append(d)
            except json.JSONDecodeError:
                print(
                    f"Skipping invalid line: {line.strip()}"
                )  # Handle non-JSON lines gracefully
    return results


def plot_epoch_dependence(
    axes: np.ndarray, data: pd.DataFrame, head: str, model_epoch: str, labels: List[str]
) -> None:

    valid_data = (
        data[data["mode"] == "eval"]
        .groupby(["mode", "epoch", "head"])
        .agg(["mean", "std"])
        .reset_index()
    )
    valid_data = valid_data[valid_data["head"] == head]
    train_data = (
        data[data["mode"] == "opt"]
        .groupby(["mode", "epoch"])
        .agg(["mean", "std"])
        .reset_index()
    )

    # ---- Plot loss ----
    ax = axes[0]
    ax.plot(
        train_data["epoch"], train_data["loss"]["mean"], color=colors[1], linewidth=1
    )
    ax.set_ylabel("Training Loss", color=colors[1])
    ax.set_yscale("log")

    ax2 = ax.twinx()
    ax2.plot(
        valid_data["epoch"], valid_data["loss"]["mean"], color=colors[0], linewidth=1
    )
    ax2.set_ylabel("Validation Loss", color=colors[0])
    ax2.set_yscale("log")

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
    twin_axes = []
    for i, label in enumerate(labels):
        color = colors[(i + 3)]
        key, axis_label = label
        if i == 0:
            main_ax = ax
        else:
            main_ax = ax.twinx()
            main_ax.spines.right.set_position(("outward", 60 * (i - 1)))
            twin_axes.append(main_ax)

        main_ax.plot(
            valid_data["epoch"],
            valid_data[key]["mean"] * 1e3,
            color=color,
            label=label,
            linewidth=1,
        )
        main_ax.set_yscale("log")
        main_ax.set_ylabel(axis_label, color=color)
        main_ax.tick_params(axis="y", colors=color)
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
    quantities: List[str],
    plot_interaction_e: bool = False,
) -> None:

    for ax, quantity in zip(axes, quantities):
        key, label = quantity

        # Store legend handles to avoid duplicates
        legend_labels = {}

        # Plot train/valid data (each entry keeps its own name)
        for name, result in train_valid_dict.items():
            if "train" in name:
                fixed_color_train_valid = colors[1]
                marker = "x"
            else:
                fixed_color_train_valid = colors[0]
                marker = "+"
            if head not in name:
                continue

            # Initialize scatter to None
            scatter = None

            if key == "energy" and "energy" in result:
                e_key = "energy" if not plot_interaction_e else "interaction_energy"
                scatter = ax.scatter(
                    result[e_key]["reference_per_atom"],
                    result[e_key]["predicted_per_atom"],
                    marker=marker,
                    color=fixed_color_train_valid,
                    label=name,
                )

            elif key == "force" and "forces" in result:
                scatter = ax.scatter(
                    result["forces"]["reference"],
                    result["forces"]["predicted"],
                    marker=marker,
                    color=fixed_color_train_valid,
                    label=name,
                )

            elif key == "stress" and "stress" in result:
                scatter = ax.scatter(
                    result["stress"]["reference"],
                    result["stress"]["predicted"],
                    marker=marker,
                    color=fixed_color_train_valid,
                    label=name,
                )

            elif key == "virials" and "virials" in result:
                scatter = ax.scatter(
                    result["virials"]["reference_per_atom"],
                    result["virials"]["predicted_per_atom"],
                    marker=marker,
                    color=fixed_color_train_valid,
                    label=name,
                )

            elif key == "dipole" and "dipole" in result:
                scatter = ax.scatter(
                    result["dipole"]["reference_per_atom"],
                    result["dipole"]["predicted_per_atom"],
                    marker=marker,
                    color=fixed_color_train_valid,
                    label=name,
                )

            # Add each train/valid dataset's name to the legend if scatter was assigned
            if scatter is not None:
                legend_labels[name] = scatter

        fixed_color_test = colors[2]  # Color for test dataset

        # Plot test data (single legend entry)
        for name, result in test_dict.items():
            # Initialize scatter to None to avoid possibly used before assignment
            scatter = None

            if key == "energy" and "energy" in result:
                e_key = "energy" if not plot_interaction_e else "interaction_energy"
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

            # Only add to legend_labels if scatter was assigned
            if scatter is not None:
                legend_labels["Test"] = scatter

        # Add diagonal line for guide
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="black",
            alpha=0.7,
        )

        # Set legend with unique entries (Test + individual train/valid names)
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
            output = model(
                batch_dict,
                training=False,
                compute_force=output_args.get("forces", False),
                compute_virials=output_args.get("virials", False),
                compute_stress=output_args.get("stress", False),
            )

            results = scatter_metric(batch, output)

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

        # Store atom counts for each configuration
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

    def update(self, batch, output):  # pylint: disable=arguments-differ
        """Update metric states with new batch data."""
        # Calculate number of atoms per configuration
        atoms_per_config = batch.ptr[1:] - batch.ptr[:-1]
        self.atom_counts.append(atoms_per_config)

        # Energy
        if output.get("energy") is not None and batch.energy is not None:
            self.ref_energies.append(batch.energy)
            self.pred_energies.append(output["energy"])
            # Per-atom normalization
            self.ref_energies_per_atom.append(batch.energy / atoms_per_config)
            self.pred_energies_per_atom.append(output["energy"] / atoms_per_config)

            self.n_energy += filter_nonzero_weight(
                batch,
                self.ref_energies,
                batch.weight,
                batch.energy_weight,
            )
            filter_nonzero_weight(
                batch,
                self.pred_energies,
                batch.weight,
                batch.energy_weight,
            )
            filter_nonzero_weight(
                batch,
                self.ref_energies_per_atom,
                batch.weight,
                batch.energy_weight,
            )
            filter_nonzero_weight(
                batch,
                self.pred_energies_per_atom,
                batch.weight,
                batch.energy_weight,
            )

        if output.get("interaction_energy") is not None and batch.energy is not None:
            E0s = output["energy"].to(torch.float64) - output["interaction_energy"].to(
                torch.float64
            )
            self.ref_interaction_energies.append(batch.energy - E0s)
            self.pred_interaction_energies.append(output["interaction_energy"])
            # Per-atom normalization
            self.ref_interaction_energies_per_atom.append(
                (batch.energy - E0s) / atoms_per_config
            )
            self.pred_interaction_energies_per_atom.append(
                output["interaction_energy"] / atoms_per_config
            )

            self.n_interaction_energy += filter_nonzero_weight(
                batch,
                self.ref_interaction_energies,
                batch.weight,
                batch.energy_weight,
            )
            filter_nonzero_weight(
                batch,
                self.pred_interaction_energies,
                batch.weight,
                batch.energy_weight,
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
                batch,
                self.ref_stress,
                batch.weight,
                batch.stress_weight,
            )
            filter_nonzero_weight(
                batch,
                self.pred_stress,
                batch.weight,
                batch.stress_weight,
            )

        # Virials
        if output.get("virials") is not None and batch.virials is not None:
            self.ref_virials.append(batch.virials)
            self.pred_virials.append(output["virials"])
            # Per-atom normalization
            atoms_per_config_3d = atoms_per_config.view(-1, 1, 1)
            self.ref_virials_per_atom.append(batch.virials / atoms_per_config_3d)
            self.pred_virials_per_atom.append(output["virials"] / atoms_per_config_3d)

            self.n_virials += filter_nonzero_weight(
                batch,
                self.ref_virials,
                batch.weight,
                batch.virials_weight,
            )
            filter_nonzero_weight(
                batch,
                self.pred_virials,
                batch.weight,
                batch.virials_weight,
            )
            filter_nonzero_weight(
                batch,
                self.ref_virials_per_atom,
                batch.weight,
                batch.virials_weight,
            )
            filter_nonzero_weight(
                batch,
                self.pred_virials_per_atom,
                batch.weight,
                batch.virials_weight,
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

    def _process_data(self, ref_list, pred_list):
        # Handle different possible states of ref_list and pred_list in distributed mode

        # Check if this is a list type object
        if isinstance(ref_list, (list, tuple)):
            if len(ref_list) == 0:
                return None, None
            ref = torch.cat(ref_list).reshape(-1)
            pred = torch.cat(pred_list).reshape(-1)
        # Handle case where ref_list is already a tensor (happens after reset in distributed mode)
        elif isinstance(ref_list, torch.Tensor):
            ref = ref_list.reshape(-1)
            pred = pred_list.reshape(-1)
        # Handle other possible types
        else:
            return None, None
        return to_numpy(ref), to_numpy(pred)

    def compute(self):
        """Compute final results for scatterplot."""
        results = {}

        # Process energies
        if self.n_energy:
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

        if self.n_interaction_energy:
            ref_interaction_e, pred_interaction_e = self._process_data(
                self.ref_interaction_energies, self.pred_interaction_energies
            )
            ref_interaction_e_pa, pred_interaction_e_pa = self._process_data(
                self.ref_interaction_energies_per_atom,
                self.pred_interaction_energies_per_atom,
            )
            results["interaction_energy"] = {
                "reference": ref_interaction_e,
                "predicted": pred_interaction_e,
                "reference_per_atom": ref_interaction_e_pa,
                "predicted_per_atom": pred_interaction_e_pa,
            }

        # Process forces
        if self.n_forces:
            ref_f, pred_f = self._process_data(self.ref_forces, self.pred_forces)
            results["forces"] = {
                "reference": ref_f,
                "predicted": pred_f,
            }

        # Process stress
        if self.n_stress:
            ref_s, pred_s = self._process_data(self.ref_stress, self.pred_stress)
            results["stress"] = {
                "reference": ref_s,
                "predicted": pred_s,
            }

        # Process virials
        if self.n_virials:
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

        # Process dipoles
        if self.n_dipole:
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
        return results
