"""The charge-aware model against the live frozen one, in one process.

A small legacy ``PolarMACE`` is built with random weights, converted, and both
are evaluated on the same structures. Energy, forces, the dipole, the charges
and spins, the density and the Fukui functions are compared directly.

**The stress is compared with a finite difference, not with the frozen tree.**
The frozen tree differentiates the long-range energy with the reciprocal cell
and the volume held at the unstrained cell's, so its stress misses their part;
develop corrected that after the tree was frozen. What is pinned here is what
develop pins: the stress is the derivative of the energy the model computed,
and that energy is the one compared with the frozen tree above.

Each structure is evaluated under the profile the frozen tree's dispatch
reaches for it on its own: an open molecule is summed in real space, a crystal
and a slab in reciprocal space, the slab with its dipole correction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_core.neighbors import get_neighborhood
from mace_core.observables import ObservableSpec
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models import EnergyOutputHead, ScaleShiftSpec
from mace_torch.models.electrostatics import PolarModel
from mace_torch.physics import DerivativeEngine

from tests.parity.fm00_convert import build_config, energy_constants_to_canonical
from tests.parity.polar_convert import (
    polar_backbone_config,
    polar_settings,
    transfer_polar_weights,
)

pytest.importorskip("graph_longrange")

pytestmark = pytest.mark.polar

ENERGY = ObservableSpec(
    name="energy",
    irreps="0e",
    per_atom=False,
    units="eV",
    derivatives=[
        {"wrt": "pos", "name": "forces", "sign": -1, "units": "eV/A"},
        {"wrt": "strain", "name": "stress", "sign": 1, "units": "eV/A^3"},
    ],
)

#: The fp64 row of the golden tolerance table. Stated here because this package
#: cannot import that harness, and a change to it is its own reviewed PR.
TOLERANCE = 1e-6

WATER = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]])

#: Name, then the structure, the profile its evaluation takes, and its charge,
#: multiplicity and applied field.
CASES = {
    "neutral molecule": (
        Atoms("OHH", positions=WATER + 5.0),
        ("molecular", None),
        (0.0, 1.0, (0.0, 0.0, 0.0)),
    ),
    "charged doublet in a field": (
        Atoms(
            "OHHOH", positions=np.vstack([WATER, WATER[:2] + np.array([2.6, 0.3, 0.4])])
        ),
        ("molecular", None),
        (-1.0, 2.0, (0.02, -0.01, 0.03)),
    ),
    "crystal": (
        Atoms(
            "OHHOHH",
            positions=np.vstack([WATER + 0.5, WATER + np.array([2.7, 2.5, 2.4])]),
            cell=[[5.2, 0.2, 0.0], [0.0, 5.0, 0.3], [0.1, 0.0, 5.4]],
            pbc=True,
        ),
        ("full_periodic", None),
        (0.0, 1.0, (0.0, 0.0, 0.0)),
    ),
    "slab": (
        Atoms(
            "OHHOHH",
            positions=np.vstack(
                [WATER + np.array([0.5, 0.5, 4.0]), WATER + np.array([2.5, 2.4, 5.1])]
            ),
            cell=[[5.0, 0.0, 0.0], [0.3, 5.0, 0.0], [0.0, 0.0, 14.0]],
            pbc=(True, True, False),
        ),
        ("z_slab", 2),
        (0.0, 1.0, (0.0, 0.0, 0.0)),
    ),
}


def legacy_polar(agnostic=True):
    from e3nn import o3

    from mace.modules import interaction_classes
    from mace.modules.extensions import PolarMACE as LegacyPolarMACE

    torch.manual_seed(7)
    model = LegacyPolarMACE(
        r_max=4.0,
        num_bessel=6,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=torch.tensor([-1.3, -4.7]),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
        correlation=2,
        gate=torch.nn.functional.silu,
        radial_MLP=[64, 64, 64],
        radial_type="bessel",
        kspace_cutoff_factor=1.0,
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.0,
        field_feature_max_l=1,
        field_feature_widths=[1.0, 1.5],
        field_feature_norms=[2.0, 3.0, 0.5, 0.7],
        num_recursion_steps=2,
        include_electrostatic_self_interaction=True,
        add_local_electron_energy=True,
        heads=["Default"],
        atomic_inter_scale=[1.1],
        atomic_inter_shift=[0.2],
        fixedpoint_update_config={
            "type": "AgnosticEmbeddedOneBodyVariableUpdate",
            "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
            "nonlinearity_cls": "MLPNonLinearity",
        },
        field_readout_config={"type": "OneBodyMLPFieldReadout"},
        use_agnostic_product=agnostic,
    )
    # Every weight away from zero, including the biases the frozen tree
    # starts at zero, so a bias that is dropped or misplaced shows.
    with torch.no_grad():
        generator = torch.Generator().manual_seed(11)
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.copy_(0.1 * torch.randn(parameter.shape, generator=generator))
    return model.eval()


def convert(legacy, profile, normal):
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    config = polar_backbone_config(legacy, build_config)
    values, scale, shift = energy_constants_to_canonical(legacy)
    head = EnergyOutputHead(
        ResolvedE0s(values),
        ["default"],
        AtomicNumberTable(numbers),
        ScaleShiftSpec("std", scale, shift),
        PrecisionConfig(),
    )
    model = PolarModel(
        ReferenceBackend(),
        observables=[ENERGY],
        energy_head=head,
        settings=polar_settings(
            legacy, periodicity_profile=profile, slab_normal=normal
        ),
        **config,
    )
    transfer_polar_weights(legacy, model, config["correlation"])
    return model, config


def with_inputs(atoms, charge, spin, field):
    atoms = atoms.copy()
    atoms.info.update(charge=charge, spin=spin, external_field=np.asarray(field))
    return atoms


def legacy_batch(legacy, atoms):
    from mace import data as legacy_data
    from mace.tools import AtomicNumberTable as LegacyTable
    from mace.tools import torch_geometric

    keyspec = legacy_data.KeySpecification(
        info_keys={
            "total_spin": "spin",
            "total_charge": "charge",
            "external_field": "external_field",
        },
    )
    configuration = legacy_data.config_from_atoms(
        atoms, key_specification=keyspec, head_name="Default"
    )
    item = legacy_data.AtomicData.from_config(
        configuration,
        z_table=LegacyTable([int(z) for z in legacy.atomic_numbers.tolist()]),
        cutoff=float(legacy.r_max),
        heads=legacy.heads,
    )
    loader = torch_geometric.DataLoader(dataset=[item], batch_size=1, shuffle=False)
    return next(iter(loader)).to_dict()


def v1_graph(atoms, numbers, cutoff):
    positions = atoms.get_positions()
    periodic = tuple(bool(flag) for flag in atoms.pbc)
    cell = np.array(atoms.cell) if any(periodic) else None
    neighborhood = get_neighborhood(positions, cutoff, periodic, cell)
    index = {z: i for i, z in enumerate(numbers)}
    return {
        "positions": torch.tensor(positions),
        "atomic_numbers": torch.tensor(atoms.get_atomic_numbers()),
        "element_index": torch.tensor(
            [index[int(z)] for z in atoms.get_atomic_numbers()]
        ),
        "edge_index": torch.tensor(neighborhood.edge_index),
        "shifts": torch.tensor(neighborhood.shifts),
        "unit_shifts": torch.tensor(neighborhood.unit_shifts),
        "cell": torch.tensor(np.asarray(neighborhood.cell, dtype=float)).reshape(
            1, 3, 3
        ),
        "pbc": torch.tensor([list(periodic)]),
        "batch": torch.zeros(len(atoms), dtype=torch.long),
        "num_graphs": 1,
        "head": torch.zeros(1, dtype=torch.long),
        "total_charge": torch.tensor([float(atoms.info["charge"])]),
        "total_spin": torch.tensor([float(atoms.info["spin"])]),
        "external_field": torch.tensor([atoms.info["external_field"]], dtype=float),
    }


def evaluate(name, agnostic=True):
    atoms, (profile, normal), inputs = CASES[name]
    atoms = with_inputs(atoms, *inputs)
    legacy = legacy_polar(agnostic)
    model, config = convert(legacy, profile, normal)
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    reference = legacy(legacy_batch(legacy, atoms), training=False, compute_force=True)
    graph = v1_graph(atoms, numbers, config["cutoff"])
    result = DerivativeEngine(model, ENERGY)(graph, compute=("forces", "stress"))
    return atoms, model, graph, reference, result


def gap(a, b) -> float:
    return float(
        (torch.as_tensor(a).detach() - torch.as_tensor(b).detach()).abs().max()
    )


@pytest.mark.parametrize("agnostic", [True, False], ids=["agnostic", "per_element"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_converted_model_matches_the_live_legacy_model(
    fp64, isolated, name, agnostic
):
    """Both product bases: the published models share one set of weights
    across elements, and the frozen tree's command line defaults to one per
    element."""
    _, _, _, reference, result = evaluate(name, agnostic)
    extras = result.extras
    compared = {
        "energy": (reference["energy"], result.total_energy),
        "forces": (reference["forces"], result.forces),
        "interaction_energy": (
            reference["interaction_energy"],
            extras["interaction_energy"],
        ),
        "electrostatic_energy": (
            reference["electrostatic_energy"],
            extras["electrostatic_energy"],
        ),
        "electron_energy": (reference["electron_energy"], extras["electron_energy"]),
        "dipole": (reference["dipole"], result.dipole),
        "total_charge": (reference["total_charge"], extras["total_charge"]),
        "charges": (reference["charges"], extras["charges"]),
        "spins": (reference["spins"], extras["spins"]),
        "density_coefficients": (
            reference["density_coefficients"],
            extras["density_coefficients"],
        ),
        "spin_charge_density": (
            reference["spin_charge_density"],
            extras["spin_charge_density"],
        ),
        "fukui_functions": (reference["fukui_functions"], extras["fukui_functions"]),
        "node_energies": (reference["node_energy"], result.node_energies),
    }
    gaps = {key: gap(*pair) for key, pair in compared.items()}
    assert max(gaps.values()) < TOLERANCE, gaps
    # Far below the gate, not just inside it: both sides compute the same
    # arithmetic in a different order, so the gap is rounding.
    assert max(gaps.values()) < 1e-10, gaps


@pytest.mark.parametrize("name", ["crystal", "slab"])
def test_the_stress_is_the_derivative_of_the_energy_compared(fp64, isolated, name):
    _atoms, model, graph, _, result = evaluate(name)
    engine = DerivativeEngine(model, ENERGY)
    cell = graph["cell"].view(3, 3)
    volume = float(torch.linalg.det(cell).abs())
    step = 1e-5
    numerical = torch.zeros(3, 3, dtype=torch.float64)
    for i in range(3):
        for j in range(3):
            energies = []
            for sign in (1.0, -1.0):
                strain = torch.zeros(3, 3, dtype=torch.float64)
                strain[i, j] += sign * step / 2
                strain[j, i] += sign * step / 2
                deform = torch.eye(3, dtype=torch.float64) + strain
                shifted = dict(graph)
                shifted["positions"] = graph["positions"] @ deform
                shifted["cell"] = (cell @ deform).reshape(1, 3, 3)
                shifted["shifts"] = graph["unit_shifts"].to(torch.float64) @ (
                    cell @ deform
                )
                energies.append(float(engine(shifted, compute=()).total_energy))
            numerical[i, j] = (energies[0] - energies[1]) / (2 * step) / volume
    torch.testing.assert_close(
        result.stress.detach().view(3, 3), numerical, rtol=0, atol=1e-8
    )
    assert float(numerical.abs().max()) > 1e-4, "no stress to compare"
