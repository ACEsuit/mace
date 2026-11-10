"""The only tutorial module allowed to import the production stack."""

import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from mace_core.data import KeySpecification, configuration_from_atoms
from mace_torch.backends.harmonics import spherical_harmonics
from mace_torch.deploy.loader import load_deployed
from mace_torch.graph import build_batch
from mace_torch.neighbors import resolve_neighbor_backend

fixture = json.loads(Path(__file__).parents[1].joinpath("fixture.json").read_text())
loaded = load_deployed(Path(__file__).parent / "fixtures/oracle.json")
atoms = Atoms(
    numbers=fixture["graph"]["numbers"], positions=fixture["graph"]["positions"]
)
configuration = configuration_from_atoms(
    atoms, KeySpecification.from_defaults(), head_name=loaded.heads[0]
)
graph = build_batch(
    [configuration],
    atomic_number_table=loaded.z_table,
    cutoff=loaded.config.model.r_max,
    heads=loaded.heads,
    neighbor_backend=resolve_neighbor_backend(),
    dtype=torch.float64,
)
output = loaded.engine(graph, compute=(), training=False)
directions = np.random.default_rng(53).normal(size=(20, 3))
print(
    json.dumps(
        {
            "energy": output.total_energy.detach().tolist(),
            "harmonics": spherical_harmonics(
                torch.tensor(directions, dtype=torch.float64), 3
            ).tolist(),
        }
    )
)
