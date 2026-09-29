"""A small periodic model and structure, built as a run builds them."""

from __future__ import annotations

import numpy as np
from ase.build import bulk
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.data import collate_training
from mace_torch.data.batch import GraphDataset
from mace_torch.train.model_stage import build_model

NUMBERS = [6, 14]


def engine(backend: str, seed: int = 3, device: str = "cpu"):
    """A two-layer, correlation-three model on ``backend``."""
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"seed": seed},
            "model": {
                "backend": backend,
                "observables": ["energy", "forces", "stress"],
                "r_max": 3.0,
                "num_channels": 4,
                "max_ell": 2,
                "num_interactions": 2,
                "correlation": 3,
                "hidden_irreps": "0e+1o",
            },
        }
    )
    built, _ = build_model(
        config,
        DEFAULT_CATALOGUE,
        z_table=AtomicNumberTable(NUMBERS),
        heads=("default",),
        e0s=ResolvedE0s({"default": {6: -1.0, 14: -2.0}}),
        statistics=DatasetStatistics(avg_num_neighbors=8.0, std=1.0),
    )
    return built.to(device)


def graph(device: str = "cpu") -> dict:
    atoms = bulk("SiC", "zincblende", a=4.35).repeat(2)
    generator = np.random.default_rng(0)
    atoms.positions += generator.normal(scale=0.05, size=atoms.positions.shape)
    item = Configuration(
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        cell=np.asarray(atoms.cell),
        pbc=(True, True, True),
    )
    table = AtomicNumberTable(NUMBERS)
    dataset = GraphDataset([item], cutoff=3.0, z_table=table, targets=())
    collated = collate_training([dataset[0]], z_table=table, float_dtype="float64")
    return dict(collated.to(device).graph)
