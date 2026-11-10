"""Native local/ghost physics against the unchanged libtorch export golden.

The old TorchScript ABI is retired. Native export/communication is exercised by
parity/test_mliap.py; this test retains the original cluster's numerical oracle.
"""

import json

import numpy as np
import torch
from ase import Atoms
from mace_core.data import KeySpecification, configuration_from_atoms
from mace_torch.deploy.loader import load_deployed
from mace_torch.graph import build_batch
from mace_torch.neighbors import resolve_neighbor_backend

from tests.golden.harness import FP64_CPU_REFERENCE
from tests.retirement.anchors import ARTIFACTS, GOLDEN


def test_native_cluster_matches_frozen_lammps_export():
    reference = json.loads(
        (GOLDEN / "references/lammps_export_libtorch_fp64.json").read_text()
    )
    recorded = reference["input"]
    loaded = load_deployed(ARTIFACTS / "tiny_scaleshift.json")
    species = np.asarray(loaded.z_table.zs)
    numbers = species[np.asarray(recorded["node_attrs"]).argmax(axis=1)]
    atoms = Atoms(numbers=numbers, positions=recorded["positions"], pbc=False)
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
    graph["positions"].requires_grad_(True)
    output = loaded.engine(graph, compute=(), training=True)
    node_energy = output.get("node_energies")
    assert node_energy is not None
    energy = (
        node_energy * torch.tensor(recorded["local_or_ghost"], dtype=torch.float64)
    ).sum()
    forces = -torch.autograd.grad(energy, graph["positions"])[0]
    image_of = np.asarray(recorded["image_of"], dtype=int)
    folded = np.zeros((int(image_of.max()) + 1, 3))
    np.add.at(folded, image_of, forces.detach().numpy())
    actual = {
        "node_energy": node_energy.detach().numpy(),
        "total_energy_local": energy.detach().numpy(),
        "folded_forces": folded,
    }
    for name, value in actual.items():
        np.testing.assert_allclose(
            value,
            reference["outputs"][name],
            atol=FP64_CPU_REFERENCE.atol,
            rtol=FP64_CPU_REFERENCE.rtol,
            err_msg=name,
        )
