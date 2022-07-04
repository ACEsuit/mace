import json
import logging
import os
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Union

from prettytable import PrettyTable
import numpy as np
import torch

from mace import data
from .train import evaluate
from .torch_geometric import dataloader
from .torch_tools import to_numpy


def compute_mae(delta: np.ndarray) -> float:
    return np.mean(np.abs(delta)).item()

def compute_rel_mae(delta: np.ndarray, target_val: np.ndarray) -> float:
    return np.mean(np.abs(delta) / target_val).item() * 100


def compute_rmse(delta: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(delta))).item()

def compute_rel_rmse(delta: np.ndarray,  target_val: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(delta / target_val))).item() * 100


def compute_q95(delta: np.ndarray) -> float:
    return np.percentile(np.abs(delta), q=95)


def compute_c(delta: np.ndarray, eta: float) -> float:
    return np.mean(np.abs(delta) < eta).item()


def get_tag(name: str, seed: int) -> str:
    return f"{name}_run-{seed}"


def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (directory is not None) and (tag is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, tag + ".log")
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)


class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)


def get_optimizer(
    name: str,
    amsgrad: bool,
    learning_rate: float,
    weight_decay: float,
    parameters: Iterable[torch.Tensor],
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(
            parameters, lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay
        )

    if name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay
        )

    raise RuntimeError(f"Unknown optimizer '{name}'")

def create_error_table(table_type: str, all_collections: list, z_table:AtomicNumberTable, r_max: float, 
                        valid_batch_size: int, model: torch.nn.Module, loss_fn: torch.nn.Module, 
                        device:str) -> PrettyTable:
    table = PrettyTable()
    if table_type == "TotalRMSE":
        table.field_names = ["config_type", "RMSE E / meV", "RMSE F / meV / A", "relative F RMSE %"]
    elif table_type == "PerAtomRMSE":
        table.field_names = ["config_type", "RMSE E / meV \n/ per atom", "RMSE F / meV / A", "relative F RMSE %"]
    elif table_type == "TotalMAE":
        table.field_names = ["config_type", "MAE E / meV", "MAE F / meV / A", "relative F MAE %"]
    elif table_type == "PerAtomMAE":
        table.field_names = ["config_type", "MAE E / meV \n/ per atom", "MAE F / meV / A", "relative F MAE %"]
    for name, subset in all_collections:
        data_loader = dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
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
                [name, f"{metrics['rmse_e'] * 1000:.1f}", f"{metrics['rmse_f'] * 1000:.1f}", f"{metrics['rel_rmse_f']:.2f}"]
            )
        elif table_type == "PerAtomRMSE":
            table.add_row(
                [name, f"{metrics['rmse_e_per_atom'] * 1000:.1f}", f"{metrics['rmse_f'] * 1000:.1f}", f"{metrics['rel_rmse_f']:.2f}"]
                )
        elif table_type == "TotalMAE":
            table.add_row(
                [name, f"{metrics['mae_e'] * 1000:.1f}", f"{metrics['mae_f'] * 1000:.1f}", f"{metrics['rel_mae_f']:.2f}"]
            )
        elif table_type == "PerAtomMAE":
            table.add_row(
                [name, f"{metrics['mae_e_per_atom'] * 1000:.1f}", f"{metrics['mae_f'] * 1000:.1f}", f"{metrics['rel_mae_f']:.2f}"]
                )
    return table

class UniversalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return to_numpy(o)
        return json.JSONEncoder.default(self, o)


class MetricsLogger:
    def __init__(self, directory: str, tag: str) -> None:
        self.directory = directory
        self.filename = tag + ".txt"
        self.path = os.path.join(self.directory, self.filename)

    def log(self, d: Dict[str, Any]) -> None:
        logging.debug(f"Saving info: {self.path}")
        os.makedirs(name=self.directory, exist_ok=True)
        with open(self.path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(d, cls=UniversalEncoder))
            f.write("\n")
