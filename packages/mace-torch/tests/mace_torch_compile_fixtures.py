"""A deployed energy model and water clusters, for the compiled-path suites."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from mace_core.config.precision import PrecisionConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import Precision
from mace_core.metadata import ConfigRecord, ModelMetadata, Provenance
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.deploy.loader import DeployedModel
from mace_torch.train.model_stage import build_model

NUMBERS = [1, 8]
E0S = {"default": {1: -13.6, 8: -2040.0}}
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def deployed(
    precision: Precision | PrecisionConfig = "float64", seed: int = 0
) -> DeployedModel:
    """The model, at a dtype name with float64 energies, or at a configuration."""
    if not isinstance(precision, PrecisionConfig):
        precision = PrecisionConfig(model=precision, accumulate="float64")
    config = ResolvedConfig.model_validate(
        {
            "model": {
                "model": "scale_shift",
                "observables": ["energy"],
                "r_max": 4.0,
                "num_channels": 4,
                "max_ell": 1,
                "hidden_irreps": "0e+1o",
            },
            "runtime": {"seed": seed},
        }
    )
    engine, outputs = build_model(
        config,
        DEFAULT_CATALOGUE,
        z_table=AtomicNumberTable(NUMBERS),
        heads=("default",),
        e0s=ResolvedE0s(E0S),
        statistics=DatasetStatistics(avg_num_neighbors=3.0),
        precision=precision,
    )
    return DeployedModel(
        engine=engine.eval(),
        config=config,
        metadata=ModelMetadata(
            config=ConfigRecord(),
            provenance=Provenance(code_version="0", git_commit=None),
        ),
        z_table=AtomicNumberTable(NUMBERS),
        heads=("default",),
        e0s=E0S,
        outputs=outputs,
        path=Path(f"water-{seed}"),
    )


def cluster(molecules: int, periodic: bool = False, seed: int = 0) -> Atoms:
    """``molecules`` waters, jittered and close enough to interact."""
    generator = np.random.default_rng(seed)
    positions = np.concatenate(
        [
            WATER
            + np.array([2.8 * index, 0.7 * (index % 2), 0.0])
            + generator.normal(scale=0.05, size=WATER.shape)
            for index in range(molecules)
        ]
    )
    atoms = Atoms("OHH" * molecules, positions=positions)
    if periodic:
        atoms.cell = np.diag([2.8 * molecules, 6.0, 6.0])
        atoms.pbc = True
    return atoms
