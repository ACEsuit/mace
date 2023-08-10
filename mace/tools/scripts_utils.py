###########################################################################################
# Training utils
# Authors: David Kovacs, Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

import torch
from prettytable import PrettyTable

from mace import data
from mace.data import AtomicData
from mace.tools import AtomicNumberTable, evaluate, torch_geometric


@dataclasses.dataclass
class SubsetCollection:
    train: data.Configurations
    valid: data.Configurations
    tests: List[Tuple[str, data.Configurations]]


def get_dataset_from_xyz(
    train_path: str,
    valid_path: str,
    valid_fraction: float,
    config_type_weights: Dict,
    test_path: str = None,
    seed: int = 1234,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipoles",
    charges_key: str = "charges",
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""
    atomic_energies_dict, all_train_configs = data.load_from_xyz(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        extract_atomic_energies=True,
    )
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )
    if valid_path is not None:
        _, valid_configs = data.load_from_xyz(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            extract_atomic_energies=False,
        )
        logging.info(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    else:
        logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction
        )
        train_configs, valid_configs = data.random_train_valid_split(
            all_train_configs, valid_fraction, seed
        )

    test_configs = []
    if test_path is not None:
        _, all_test_configs = data.load_from_xyz(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            extract_atomic_energies=False,
        )
        # create list of tuples (config_type, list(Atoms))
        test_configs = data.test_config_types(all_test_configs)
        logging.info(
            f"Loaded {len(all_test_configs)} test configurations from '{test_path}'"
        )
    return (
        SubsetCollection(train=train_configs, valid=valid_configs, tests=test_configs),
        atomic_energies_dict,
    )


class LRScheduler:
    def __init__(self, optimizer, args) -> None:
        self.scheduler = args.scheduler
        if args.scheduler == "ExponentialLR":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=args.lr_scheduler_gamma
            )
        elif args.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=args.lr_factor,
                patience=args.scheduler_patience,
            )
        else:
            raise RuntimeError(f"Unknown scheduler: '{args.scheduler}'")

    def step(self, metrics=None, epoch=None):  # pylint: disable=E1123
        if self.scheduler == "ExponentialLR":
            self.lr_scheduler.step(epoch=epoch)
        elif self.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler.step(metrics=metrics, epoch=epoch)

    def __getattr__(self, name):
        if name == "step":
            return self.step
        return getattr(self.lr_scheduler, name)


def create_error_table(
    table_type: str,
    all_collections: list,
    z_table: AtomicNumberTable,
    r_max: float,
    valid_batch_size: int,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    output_args: Dict[str, bool],
    log_wandb: bool,
    device: str,
) -> PrettyTable:
    if log_wandb:
        import wandb
    table = PrettyTable()
    if table_type == "TotalRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    elif table_type == "PerAtomRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    elif table_type == "PerAtomRMSEstressvirials":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE Stress (Virials) / meV / A (A^3)",
        ]
    elif table_type == "TotalMAE":
        table.field_names = [
            "config_type",
            "MAE E / meV",
            "MAE F / meV / A",
            "relative F MAE %",
        ]
    elif table_type == "PerAtomMAE":
        table.field_names = [
            "config_type",
            "MAE E / meV / atom",
            "MAE F / meV / A",
            "relative F MAE %",
        ]
    elif table_type == "DipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE MU / mDebye / atom",
            "relative MU RMSE %",
        ]
    elif table_type == "DipoleMAE":
        table.field_names = [
            "config_type",
            "MAE MU / mDebye / atom",
            "relative MU MAE %",
        ]
    elif table_type == "EnergyDipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "rel F RMSE %",
            "RMSE MU / mDebye / atom",
            "rel MU RMSE %",
        ]
    for name, subset in all_collections:
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
                for config in subset
            ],
            batch_size=valid_batch_size,
            shuffle=False,
            drop_last=False,
        )

        logging.info(f"Evaluating {name} ...")
        _, metrics = evaluate(
            model,
            loss_fn=loss_fn,
            data_loader=data_loader,
            output_args=output_args,
            device=device,
        )
        if log_wandb:
            wandb_log_dict = {
                name
                + "_final_rmse_e_per_atom": metrics["rmse_e_per_atom"]
                * 1e3,  # meV / atom
                name + "_final_rmse_f": metrics["rmse_f"] * 1e3,  # meV / A
                name + "_final_rel_rmse_f": metrics["rel_rmse_f"],
            }
            wandb.log(wandb_log_dict)
        if table_type == "TotalRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e'] * 1000:.1f}",
                    f"{metrics['rmse_f'] * 1000:.1f}",
                    f"{metrics['rel_rmse_f']:.2f}",
                ]
            )
        elif table_type == "PerAtomRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:.1f}",
                    f"{metrics['rmse_f'] * 1000:.1f}",
                    f"{metrics['rel_rmse_f']:.2f}",
                ]
            )
        elif (
            table_type == "PerAtomRMSEstressvirials"
            and metrics["rmse_stress"] is not None
        ):
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:.1f}",
                    f"{metrics['rmse_f'] * 1000:.1f}",
                    f"{metrics['rel_rmse_f']:.2f}",
                    f"{metrics['rmse_stress'] * 1000:.1f}",
                ]
            )
        elif (
            table_type == "PerAtomRMSEstressvirials"
            and metrics["rmse_virials"] is not None
        ):
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:.1f}",
                    f"{metrics['rmse_f'] * 1000:.1f}",
                    f"{metrics['rel_rmse_f']:.2f}",
                    f"{metrics['rmse_virials'] * 1000:.1f}",
                ]
            )
        elif table_type == "TotalMAE":
            table.add_row(
                [
                    name,
                    f"{metrics['mae_e'] * 1000:.1f}",
                    f"{metrics['mae_f'] * 1000:.1f}",
                    f"{metrics['rel_mae_f']:.2f}",
                ]
            )
        elif table_type == "PerAtomMAE":
            table.add_row(
                [
                    name,
                    f"{metrics['mae_e_per_atom'] * 1000:.1f}",
                    f"{metrics['mae_f'] * 1000:.1f}",
                    f"{metrics['rel_mae_f']:.2f}",
                ]
            )
        elif table_type == "DipoleRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_mu_per_atom'] * 1000:.2f}",
                    f"{metrics['rel_rmse_mu']:.1f}",
                ]
            )
        elif table_type == "DipoleMAE":
            table.add_row(
                [
                    name,
                    f"{metrics['mae_mu_per_atom'] * 1000:.2f}",
                    f"{metrics['rel_mae_mu']:.1f}",
                ]
            )
        elif table_type == "EnergyDipoleRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:.1f}",
                    f"{metrics['rmse_f'] * 1000:.1f}",
                    f"{metrics['rel_rmse_f']:.1f}",
                    f"{metrics['rmse_mu_per_atom'] * 1000:.1f}",
                    f"{metrics['rel_rmse_mu']:.1f}",
                ]
            )
    return table
