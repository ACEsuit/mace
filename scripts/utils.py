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
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""
    atomic_energies_dict, all_train_configs = data.load_from_xyz(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
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


def create_error_table(
    table_type: str,
    all_collections: list,
    z_table: AtomicNumberTable,
    r_max: float,
    valid_batch_size: int,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: str,
) -> PrettyTable:
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
            model, loss_fn=loss_fn, data_loader=data_loader, device=device
        )
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
    return table
