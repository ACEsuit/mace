"""A small periodic model and structure, for comparing one backend with another.

A module of its own, with a name no other package's tests use, so both the
cueq and the oeq tests can build the same model without one importing the
other's skip conditions.
"""

from __future__ import annotations

import numpy as np
from ase.build import bulk
from mace_core.config.precision import PrecisionConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.data import collate_training
from mace_torch.data.batch import GraphDataset
from mace_torch.train.model_stage import build_model


def engine(
    backend: str,
    seed: int = 3,
    device: str = "cpu",
    heads: tuple[str, ...] = ("a",),
    readout: dict | None = None,
    observables: tuple[str, ...] = ("energy", "forces", "stress"),
    precision: PrecisionConfig | None = None,
):
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"seed": seed},
            "data": {"heads": {name: {"train_file": "x.xyz"} for name in heads}},
            "model": {
                "backend": backend,
                "observables": list(observables),
                **({"readout": readout} if readout else {}),
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
        z_table=AtomicNumberTable([6, 14]),
        heads=heads,
        e0s=ResolvedE0s({name: {6: -1.0, 14: -2.0} for name in heads}),
        statistics=DatasetStatistics(avg_num_neighbors=8.0, std=1.0),
        **({"precision": precision} if precision is not None else {}),
    )
    return built.to(device)


def graph(device: str = "cpu"):
    atoms = bulk("SiC", "zincblende", a=4.35).repeat(2)
    generator = np.random.default_rng(0)
    atoms.positions += generator.normal(scale=0.05, size=atoms.positions.shape)
    item = Configuration(
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        cell=np.asarray(atoms.cell),
        pbc=(True, True, True),
    )
    table = AtomicNumberTable([6, 14])
    dataset = GraphDataset([item], cutoff=3.0, z_table=table, targets=())
    collated = collate_training([dataset[0]], z_table=table, float_dtype="float64")
    return dict(collated.to(device).graph)
