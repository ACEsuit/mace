import argparse
import dataclasses
import glob
import json
import os
import re
from typing import List
import logging
from typing import Dict
import torch
from torchmetrics import Metric
import matplotlib.pyplot as plt
import numpy as np
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

#TODO: figure out device/distributed
def plot_training(
        model_epoch: str,
        swa_start: int,
        results_dir: str,
        heads: str,
        table_type: str,
        model: torch.nn.Module,
        train_valid_data: dict,
        test_data: dict,
        output_args: str,
        device: str,
        distributed: bool=False) -> None:

    error_type = {
    "TotalRMSE": ([
        ("rmse_e", "RMSE E [meV]"),
        ("rmse_f", "RMSE F [meV / A]")
    ], [("energy", "Energy per atom [meV]"), ("force", "Force [meV / A]")]),

    "PerAtomRMSE": ([
        ("rmse_e_per_atom", "RMSE E/atom [meV]"),
        ("rmse_f", "RMSE F [meV / A]")
    ], [("energy", "Energy per atom [meV]"), ("force", "Force [meV / A]")]),

    "PerAtomRMSEstressvirials": ([
        ("rmse_e_per_atom", "RMSE E/atom [meV]"),
        ("rmse_f", "RMSE F [meV / A]"),
        ("rmse_stress", "RMSE Stress [meV / A^3]")
    ], [("energy", "Energy per atom [meV]"), ("force", "Force [meV / A]"), ("stress", "Stress [meV / A^3]")]),

    "PerAtomMAEstressvirials": ([
        ("mae_e_per_atom", "MAE E/atom [meV]"),
        ("mae_f", "MAE F [meV / A]"),
        ("mae_stress", "MAE Stress [meV / A^3]")
    ], [("energy", "Energy per atom [meV]"), ("force", "Force [meV / A]"), ("stress", "Stress [meV / A^3]")]),

    "TotalMAE": ([
        ("mae_e", "MAE E [meV]"),
        ("mae_f", "MAE F [meV / A]")
    ], [("energy", "Energy per atom [meV]"), ("force", "Force [meV / A]")]),

    "PerAtomMAE": ([
        ("mae_e_per_atom", "MAE E/atom [meV]"),
        ("mae_f", "MAE F [meV / A]")
    ], [("energy", "Energy per atom [meV]"), ("force", "Force [meV / A]")]),

    "DipoleRMSE": ([
        ("rmse_mu_per_atom", "RMSE MU/atom [mDebye]"),
        ("rel_rmse_f", "Relative MU RMSE [%]")
    ], [("dipole", "Dipole per atom [mDebye]")]),

    "DipoleMAE": ([
        ("mae_mu", "MAE MU [mDebye]"),
        ("rel_mae_f", "Relative MU MAE [%]")
    ], [("dipole", "Dipole per atom [mDebye]")]),

    "EnergyDipoleRMSE": ([
        ("rmse_e_per_atom", "RMSE E/atom [meV]"),
        ("rmse_f", "RMSE F [meV / A]"),
        ("rmse_mu_per_atom", "RMSE MU/atom [mDebye]")
    ], [("energy", "Energy per atom [meV]"), ("force", "Force [meV / A]"), ("dipole", "Dipole per atom [mDebye]")]),
    }

           #EPOCH dependence
    data = pd.DataFrame(
            results
            for results in parse_training_results(results_dir)
    )
    labels,quantities = error_type[table_type]

    print(swa_start)


    for head in heads:
        
        fig = plt.figure(layout='constrained', figsize=(10, 6))
        subfigs = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.05)
        axsTop = subfigs[0].subplots(1, 2, sharey=False)
        axsBottom = subfigs[1].subplots(1,len(quantities),  sharey=False)
            

        plot_epoch_dependence(axsTop, data,  head, model_epoch, labels)
        
        # #TODO: on what to check model inference

        plot_inference(axsBottom, train_valid_data, test_data, head, quantities, model, output_args, device, distributed)

        if swa_start < model_epoch:
            axsTop[0].axvline(swa_start, color="black", linestyle="dashed", linewidth=1, alpha=0.6, label="Stage Two Starts")
            axsTop[0].legend(loc="best")
            axsTop[1].axvline(swa_start, color="black", linestyle="dashed", linewidth=1, alpha=0.6, label="Stage Two Starts")
            fig.savefig(f"{results_dir[:-4]}_{head}_stage_two.png", dpi=300, bbox_inches="tight")

        else:
            fig.savefig(f"{results_dir[:-4]}_{head}_stage_one.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


#LOADING training results

#TODO: check if this is the correct way to load for comittee models
def parse_training_results(path: str) -> List[dict]:
    results = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())  # Ensure it's valid JSON
                results.append(d)
            except json.JSONDecodeError:
                print(f"Skipping invalid line: {line.strip()}")  # Handle non-JSON lines gracefully
    return results



def plot_epoch_dependence( 
    axes: np.ndarray,
    data: pd.DataFrame,
    head: str, 
    model_epoch: str,
    labels: List[str])-> None:

    valid_data = data[data["mode"] == "eval"].groupby(["mode", "epoch", "head"]).agg(["mean", "std"]).reset_index()
    valid_data = valid_data[valid_data["head"] == head]
    train_data = data[data["mode"] == "opt"].groupby(["mode", "epoch"]).agg(["mean", "std"]).reset_index()


    # ---- Plot loss ----
    ax = axes[0]
    ax.plot(train_data["epoch"], train_data["loss"]["mean"], color=colors[1], label="Training", linewidth=1)
    ax.plot(valid_data["epoch"], valid_data["loss"]["mean"], color=colors[0], label="Validation", linewidth=1)

    ax.axvline(model_epoch, color="black", linestyle="solid", linewidth=1, alpha=0.8, label="Loaded Model")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
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
                main_ax.spines.right.set_position(("outward", 40 * (i - 1)))
                twin_axes.append(main_ax)

            main_ax.plot(valid_data["epoch"], valid_data[key]["mean"] * 1e3, color=color, label=label, linewidth=1)

            main_ax.set_ylabel(axis_label, color=color)
            main_ax.tick_params(axis="y", colors=color)
    ax.axvline(model_epoch, color="black", linestyle="solid", linewidth=1, alpha=0.8, label="Loaded Model")
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.5)

#INFERENCE=========

def plot_inference(
    axes: np.ndarray,
    train_valid_data: dict,
    test_data: dict,
    head: str,  
    quantities: List[str],
    model: torch.nn.Module,
    output_args: Dict[str, bool],
    device:str,
    distributed: bool) -> None:
    
    
    train_valid_dict = model_inference(train_valid_data, model, quantities, output_args, device, distributed)
    test_dict = model_inference(test_data, model, quantities, output_args, device, distributed)
    #TODO: add dipole per atom to InferenceMetric
    #TODO: turn values meV
    for ax, quantity in zip(axes,quantities):
        key, label = quantity

        fixed_color_test = colors[2]  # Color for test dataset


        # Store legend handles to avoid duplicates
        legend_labels = {}

        # Plot test data (single legend entry)
        for name, result in test_dict.items():

            if key == "energy" and "energy" in result:  
                scatter = ax.scatter(result["energy"]["reference_per_atom"], 
                                    result["energy"]["predicted_per_atom"],
                                    marker="o",
                                    size=1, 
                                    color=fixed_color_test, label="Test")
            
            elif key == "force" and "forces" in result:  
                scatter = ax.scatter(result["forces"]["reference"], 
                                    result["forces"]["predicted"], 
                                    marker="o",
                                    size=1, 
                                    color=fixed_color_test, label="Test")
            
            elif key == "stress" and "stress" in result:  
                scatter = ax.scatter(result["stress"]["reference"], 
                                    result["stress"]["predicted"], 
                                    marker="o",
                                    size=1, 
                                    color=fixed_color_test, label="Test")
            
            elif key == "virials" and "virials" in result:  
                scatter = ax.scatter(result["virials"]["reference_per_atom"], 
                                    result["virials"]["predicted_per_atom"], 
                                    marker="o",
                                    size=1, 
                                    color=fixed_color_test, label="Test")

            legend_labels["Test"] = scatter

        # Plot train/valid data (each entry keeps its own name)
        for name, result in train_valid_dict.items():
            print("Name:", name)
            if "train" in name:
                fixed_color_train_valid = colors[1]
                marker = "x"
            else:
                fixed_color_train_valid = colors[0]
                marker = "+"
            if head not in name:
                continue

            if key == "energy" and "energy" in result:  
                scatter = ax.scatter(result["energy"]["reference_per_atom"], 
                                    result["energy"]["predicted_per_atom"], 
                                    marker=marker,
                                    color=fixed_color_train_valid, label=name)

            elif key == "force" and "forces" in result:  
                scatter = ax.scatter(result["forces"]["reference"], 
                                    result["forces"]["predicted"], 
                                    marker=marker,
                                    color=fixed_color_train_valid, label=name)

            elif key == "stress" and "stress" in result:  
                scatter = ax.scatter(result["stress"]["reference"], 
                                    result["stress"]["predicted"], 
                                    marker=marker,
                                    color=fixed_color_train_valid, label=name)

            elif key == "virials" and "virials" in result:  
                scatter = ax.scatter(result["virials"]["reference_per_atom"], 
                                    result["virials"]["predicted_per_atom"], 
                                    marker=marker,
                                    color=fixed_color_train_valid, label=name)

            # Add each train/valid dataset's name to the legend
            legend_labels[name] = scatter

        # Set legend with unique entries (Test + individual train/valid names)
        ax.legend(handles=legend_labels.values(), labels=legend_labels.keys(), loc="best")
        ax.set_xlabel(f"Reference {label}")
        ax.set_ylabel(f"MACE {label}")
        ax.grid(True, linestyle="--", alpha=0.5)
        

def model_inference(
    all_data_loaders: dict,
    model: torch.nn.Module,
    quantities: List[str],
    output_args: Dict[str, bool],
    device: str,
    distributed: bool = False,
):
    #Disable gradients for evaluation

    for param in model.parameters():
        param.requires_grad = False
    scatter_metric = InferenceMetric().to(device)
    results_dict = {}
    for name in all_data_loaders: 
        data_loader = all_data_loaders[name]
        logging.info(f"Inference on {name} ...")
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

            scatter_metric.update(batch, output)
        results = scatter_metric.compute()
        results_dict[name] = results 
        scatter_metric.reset()
            
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
        self.add_state("pred_energies", default=[], dist_reduce_fx="cat")
        self.add_state("ref_forces", default=[], dist_reduce_fx="cat")
        self.add_state("pred_forces", default=[], dist_reduce_fx="cat")
        self.add_state("ref_stress", default=[], dist_reduce_fx="cat")
        self.add_state("pred_stress", default=[], dist_reduce_fx="cat")
        self.add_state("ref_virials", default=[], dist_reduce_fx="cat")
        self.add_state("pred_virials", default=[], dist_reduce_fx="cat")
        
        # Per-atom normalized values
        self.add_state("ref_energies_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("pred_energies_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("ref_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("pred_virials_per_atom", default=[], dist_reduce_fx="cat")
        
        # Store atom counts for each configuration
        self.add_state("atom_counts", default=[], dist_reduce_fx="cat")
        
        # Counters
        self.add_state("n_energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_forces", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_stress", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_virials", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch, output):
        """Update metric states with new batch data."""
        # Calculate number of atoms per configuration
        atoms_per_config = batch.ptr[1:] - batch.ptr[:-1]
        self.atom_counts.append(atoms_per_config)

        # Energy
        if output.get("energy") is not None and batch.energy is not None:
            self.n_energy += batch.num_graphs
            self.ref_energies.append(batch.energy)
            self.pred_energies.append(output["energy"])
            # Per-atom normalization
            self.ref_energies_per_atom.append(batch.energy / atoms_per_config)
            self.pred_energies_per_atom.append(output["energy"] / atoms_per_config)

        # Forces
        if output.get("forces") is not None and batch.forces is not None:
            self.n_forces += batch.forces.shape[0]
            self.ref_forces.append(batch.forces)
            self.pred_forces.append(output["forces"])

        # Stress
        if output.get("stress") is not None and batch.stress is not None:
            self.n_stress += batch.stress.shape[0]
            self.ref_stress.append(batch.stress)
            self.pred_stress.append(output["stress"])

        # Virials
        if output.get("virials") is not None and batch.virials is not None:
            self.n_virials += batch.virials.shape[0]
            self.ref_virials.append(batch.virials)
            self.pred_virials.append(output["virials"])
            # Per-atom normalization
            atoms_per_config_3d = atoms_per_config.view(-1, 1, 1)
            self.ref_virials_per_atom.append(batch.virials / atoms_per_config_3d)
            self.pred_virials_per_atom.append(output["virials"] / atoms_per_config_3d)

    def _process_data(self, ref_list, pred_list):
        """Helper to process and convert data to numpy arrays."""
        if not ref_list:
            return None, None
        ref = torch.cat(ref_list).reshape(-1)  # Flatten to 1D
        pred = torch.cat(pred_list).reshape(-1)  # Flatten to 1D
        return to_numpy(ref), to_numpy(pred)

    def compute(self):
        """Compute final results for scatterplot."""
        results = {}
        
        # Process energies
        if self.n_energy > 0:
            ref_e, pred_e = self._process_data(self.ref_energies, self.pred_energies)
            ref_e_pa, pred_e_pa = self._process_data(
                self.ref_energies_per_atom, self.pred_energies_per_atom
            )
            results["energy"] = {
                "reference": ref_e,
                "predicted": pred_e,
                "reference_per_atom": ref_e_pa,
                "predicted_per_atom": pred_e_pa,
                "n_points": to_numpy(self.n_energy).item()
            }
            
        # Process forces
        if self.n_forces > 0:
            ref_f, pred_f = self._process_data(self.ref_forces, self.pred_forces)
            results["forces"] = {
                "reference": ref_f,
                "predicted": pred_f,
                "n_points": to_numpy(self.n_forces).item()
            }
            
        # Process stress
        if self.n_stress > 0:
            ref_s, pred_s = self._process_data(self.ref_stress, self.pred_stress)
            results["stress"] = {
                "reference": ref_s,
                "predicted": pred_s,
                "n_points": to_numpy(self.n_stress).item()
            }
            
        # Process virials
        if self.n_virials > 0:
            ref_v, pred_v = self._process_data(self.ref_virials, self.pred_virials)
            ref_v_pa, pred_v_pa = self._process_data(
                self.ref_virials_per_atom, self.pred_virials_per_atom
            )
            results["virials"] = {
                "reference": ref_v,
                "predicted": pred_v,
                "reference_per_atom": ref_v_pa,
                "predicted_per_atom": pred_v_pa,
                "n_points": to_numpy(self.n_virials).item()
            }
            
        return results

