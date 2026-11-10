"""ELEC-3 fixtures and a test-only bridge to the pinned external oracle.

No runtime converter is shipped: the bridge only supplies frozen test weights.
The external packages load in a child process with their required MACE version.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.config.model import ModelConfig
from mace_core.config.precision import PrecisionConfig
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import widest
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.backends import resolve_backend
from mace_torch.graph import build_batch
from mace_torch.models import EnergyReadout, ScaleShiftSpec
from mace_torch.models.electrostatics import LocalChargeModel
from mace_torch.models.registry import registered_architectures
from mace_torch.neighbors import resolve_neighbor_backend
from mace_torch.physics import DerivativeEngine
from scipy.spatial.transform import Rotation

from tests.parity.anchors import load_anchor_weights

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "molecular": "realspace",
    "molecular_rotated": "realspace",
    "full_periodic": "pbc",
    "z_slab": "slab",
    "partial": "mixed_periodic",
    "per_structure": "auto",
}
ENERGY = DEFAULT_CATALOGUE.observable("energy")


def configuration(kind, degree, **options):
    return ModelConfig(
        model=kind,
        local_sources={
            "kind": kind,
            "multipole_max_l": degree,
            **({"formal_charges": {1: 1.0, 8: -2.0}} if kind == "lsc_mace" else {}),
            **options,
        },
        num_channels=2,
        num_interactions=2,
        max_L=1,
        max_ell=1,
        correlation=2,
        radial_mlp=(8,),
        radial_basis={"kind": "bessel", "num_basis": 4},
        scaling="none",
        readout={"mlp_irreps": "4x0e"},
        interaction_first="plain",
    )


def build_model(
    kind, degree, profile, dtype=torch.float64, backend="reference", **options
):
    precision = str(dtype).removeprefix("torch.")
    head = EnergyReadout(
        ResolvedE0s({"default": {1: -0.2, 8: -0.7}}),
        ["default"],
        AtomicNumberTable([1, 8]),
        ScaleShiftSpec("none", (1.0,), (0.0,)),
        PrecisionConfig(model=precision),
        widen_node_energy=False,
    )
    config = configuration(kind, degree, **options)
    return registered_architectures()[kind].model(
        resolve_backend(backend),
        config,
        [1, 8],
        [ENERGY],
        head,
        settings=LocalChargeModel.make_solver_spec(
            config,
            ElectrostaticsConfig(
                enabled=True,
                periodicity_profile="molecular"
                if profile == "molecular_rotated"
                else profile,
                slab_normal=2 if profile == "z_slab" else None,
            ),
            precision=widest(precision, head.accumulate),
        ),
        precision=precision,
    )


def configurations(profile):
    positions = np.array(
        [[0.5, 0.4, 0.3], [1.45, 0.4, 0.3], [0.26, 1.33, 0.3], [3.1, 0.8, 0.5]]
    )
    numbers = np.array([8, 1, 1, 1])
    fields = {"external_field": np.array([0.01, -0.02, 0.005])}
    molecule = Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=np.zeros((3, 3)),
        pbc=(False, False, False),
        properties=fields,
    )
    rotation = Rotation.from_rotvec([0.3, -0.2, 0.4]).as_matrix()
    rotated = Configuration(
        atomic_numbers=numbers,
        positions=positions @ rotation.T,
        cell=np.zeros((3, 3)),
        pbc=(False, False, False),
        properties={"external_field": fields["external_field"] @ rotation.T},
    )
    crystal = Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=np.array([[5.2, 0.2, 0], [0, 5.4, 0.1], [0.2, 0, 5.6]]),
        pbc=(True, True, True),
        properties=fields,
    )
    slab = Configuration(
        atomic_numbers=numbers,
        positions=positions + np.array([0, 0, 4.0]),
        cell=np.diag([5.2, 5.4, 12.0]),
        pbc=(True, True, False),
        properties=fields,
    )
    return {
        "molecular": [molecule],
        "molecular_rotated": [rotated],
        "full_periodic": [crystal],
        "z_slab": [slab],
        "partial": [molecule, crystal, slab],
        "per_structure": [molecule, crystal, slab],
    }[profile]


def graph_for(profile, dtype=torch.float64):
    graph = build_batch(
        configurations(profile),
        atomic_number_table=AtomicNumberTable([1, 8]),
        cutoff=5.0,
        neighbor_backend=resolve_neighbor_backend(),
        dtype=dtype,
    )
    graph["charges"] = graph["node_attrs"] @ torch.tensor([1.0, -2.0], dtype=dtype)
    return graph


def outputs(model, graph):
    result = DerivativeEngine(model, ENERGY)(
        graph, compute=("forces", "stress", "virials"), training=True
    )
    return {
        "energy": result.total_energy,
        "node_energy": result.node_energies,
        "forces": result.forces,
        "stress": result.stress,
        "virials": result.virials,
        "dipole": result.dipole,
        "density_coefficients": result.extras["density_coefficients"],
    }


def anchor_path(kind, degree):
    return ROOT / "tests/golden/models" / f"elec3_{kind}_l{degree}.safetensors"


def reference_path(kind):
    return ROOT / "tests/golden/references" / f"elec3_{kind}.json"


def load_model(kind, degree, profile, dtype=torch.float64, backend="reference"):
    model = build_model(kind, degree, profile, dtype, backend)
    load_anchor_weights(anchor_path(kind, degree), model)
    return model
