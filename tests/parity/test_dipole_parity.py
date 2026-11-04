"""The dipole and dielectric models against the live frozen ones, in one process.

Two legacy models are converted and evaluated on the same structures as the
originals: the committed ``AtomicDipolesMACE`` anchor, and a small
``AtomicDielectricMACE`` built here with random weights, once with a readout
middle that carries every irrep and once with the scalar-only middle the
command line defaults to. The dipole, the per-atom dipoles, the charges and
both forms of the polarizability are compared, and so are the position
derivatives of the dipole and the polarizability.

The frozen dipole model has no derivative route of its own: its forward takes
no ``compute_dielectric_derivatives``. Its ``dmu_dr`` is taken here with the
frozen tree's own function, over its own output, which is what that route
would return.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from mace_core.neighbors import get_neighborhood
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models.dipoles import DipoleModel, DipoleSettings
from mace_torch.neighbors import resolve_neighbor_backend
from mace_torch.physics import DerivativeEngine

from tests.parity.dipole_convert import dipole_config, transfer_dipole_weights
from tests.parity.fm00_convert import model_arguments

#: The fp64 row of the golden tolerance table, as in the other parity tests.
TOLERANCE = 1e-6

ANCHOR = (
    Path(__file__).resolve().parents[1] / "golden" / "models" / "tiny_dipoles.model"
)

WATER = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]])

#: The structures, each with the fixed charges the dipole model reads and the
#: total charge the dielectric model fixes its predicted ones to.
CASES = {
    "water": (
        Atoms("OHH", positions=WATER + 1.0),
        [-0.8, 0.4, 0.4],
        0.0,
    ),
    "methanol": (
        Atoms(
            "COHHHH",
            positions=[
                [0.0, 0.0, 0.0],
                [1.42, 0.0, 0.0],
                [1.75, 0.9, 0.1],
                [-0.36, 1.03, 0.05],
                [-0.36, -0.52, 0.89],
                [-0.36, -0.5, -0.9],
            ],
        ),
        [-0.1, -0.6, 0.4, 0.1, 0.1, 0.1],
        0.0,
    ),
    "charged pair": (
        Atoms("OHHO", positions=np.vstack([WATER, [[2.4, 0.4, 0.3]]])),
        [-0.9, 0.5, 0.5, -1.1],
        -1.0,
    ),
}


def legacy_dielectric(mlp_irreps: str, atomic_numbers=(1, 6, 8), gate="silu"):
    from e3nn import o3

    from mace.modules import gate_dict, interaction_classes
    from mace.modules.models import AtomicDielectricMACE

    torch.manual_seed(5)
    model = AtomicDielectricMACE(
        r_max=3.5,
        num_bessel=6,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=len(atomic_numbers),
        hidden_irreps=o3.Irreps("4x0e + 4x1o + 4x2e"),
        MLP_irreps=o3.Irreps(mlp_irreps),
        avg_num_neighbors=3.0,
        atomic_numbers=list(atomic_numbers),
        correlation=2,
        gate=gate_dict[gate],
        radial_MLP=[64, 64, 64],
        use_polarizability=True,
    )
    return model.eval()


def legacy_anchor():
    return torch.load(ANCHOR, map_location="cpu", weights_only=False).eval()


def convert(legacy, family: str):
    config = dipole_config(legacy)
    settings = (
        DipoleSettings(charges="fixed")
        if family == "dipole"
        else DipoleSettings(charges="predicted", polarizability=True)
    )
    model = DipoleModel(
        ReferenceBackend(), settings=settings, **model_arguments(config)
    )
    transfer_dipole_weights(legacy, model, family)
    return model, config


def legacy_batch(legacy, atoms, charges, total_charge):
    from mace import data as legacy_data
    from mace.tools import AtomicNumberTable as LegacyTable
    from mace.tools import torch_geometric

    atoms = atoms.copy()
    atoms.arrays["REF_charges"] = np.asarray(charges, dtype=float)
    atoms.info["charge"] = total_charge
    keyspec = legacy_data.KeySpecification(
        info_keys={"total_charge": "charge"},
        arrays_keys={"charges": "REF_charges"},
    )
    configuration = legacy_data.config_from_atoms(atoms, key_specification=keyspec)
    item = legacy_data.AtomicData.from_config(
        configuration,
        z_table=LegacyTable([int(z) for z in legacy.atomic_numbers.tolist()]),
        cutoff=float(legacy.r_max),
    )
    loader = torch_geometric.DataLoader(dataset=[item], batch_size=1, shuffle=False)
    return next(iter(loader)).to_dict()


def v1_graph(atoms, numbers, cutoff, charges, total_charge):
    positions = atoms.get_positions()
    neighborhood = get_neighborhood(
        positions,
        cutoff,
        (False, False, False),
        None,
        backend=resolve_neighbor_backend(),
    )
    index = {z: i for i, z in enumerate(numbers)}
    return {
        "positions": torch.tensor(positions),
        "node_atomic_numbers": torch.tensor(atoms.get_atomic_numbers()),
        "node_attrs": torch.nn.functional.one_hot(
            torch.tensor([index[int(z)] for z in atoms.get_atomic_numbers()]),
            num_classes=len(numbers),
        ).to(torch.get_default_dtype()),
        "edge_index": torch.tensor(neighborhood.edge_index),
        "shifts": torch.tensor(neighborhood.shifts),
        "unit_shifts": torch.tensor(neighborhood.unit_shifts),
        "cell": torch.tensor(np.asarray(neighborhood.cell, dtype=float)).reshape(
            1, 3, 3
        ),
        "pbc": torch.tensor([[False, False, False]]),
        "batch": torch.zeros(len(atoms), dtype=torch.long),
        "ptr": torch.cat(
            (
                torch.zeros(1, dtype=torch.long),
                torch.bincount(torch.zeros(len(atoms), dtype=torch.long)).cumsum(0),
            )
        ),
        "head": torch.zeros(1, dtype=torch.long),
        "charges": torch.tensor(charges, dtype=torch.float64),
        "total_charge": torch.tensor([float(total_charge)]),
    }


def engine(model):
    responses = [DEFAULT_CATALOGUE.observable(n) for n in ("dipole", "polarizability")]
    return DerivativeEngine(model, None, responses=responses)


def gap(a, b) -> float:
    return float(
        (torch.as_tensor(a).detach() - torch.as_tensor(b).detach()).abs().max()
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_dipole_anchor_matches_the_live_legacy_model(fp64, name):
    from mace.modules.utils import compute_dielectric_gradients

    atoms, charges, total_charge = CASES[name]
    legacy = legacy_anchor()
    model, config = convert(legacy, "dipole")
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    batch = legacy_batch(legacy, atoms, charges, total_charge)
    reference = legacy(batch, training=False)
    expected_dmu_dr = compute_dielectric_gradients(
        reference["dipole"], batch["positions"]
    )
    graph = v1_graph(atoms, numbers, config["cutoff"], charges, total_charge)
    result = engine(model)(graph, compute=("dmu_dr",))

    assert gap(result.dipole, reference["dipole"]) < 1e-12
    assert gap(result.extras["atomic_dipoles"], reference["atomic_dipoles"]) < 1e-12
    assert gap(result.extras["dmu_dr"], expected_dmu_dr) < 1e-12
    # The charges reach the dipole: without them the two would differ.
    assert gap(reference["dipole"], reference["atomic_dipoles"].sum(0)) > TOLERANCE


@pytest.mark.parametrize("mlp_irreps", ["8x0e + 8x1o + 8x2e", "8x0e"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_dielectric_model_matches_the_live_legacy_model(fp64, name, mlp_irreps):
    atoms, charges, total_charge = CASES[name]
    legacy = legacy_dielectric(mlp_irreps)
    model, config = convert(legacy, "dielectric")
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    batch = legacy_batch(legacy, atoms, charges, total_charge)
    reference = legacy(batch, training=False, compute_dielectric_derivatives=True)
    graph = v1_graph(atoms, numbers, config["cutoff"], charges, total_charge)
    result = engine(model)(graph, compute=("dmu_dr", "dalpha_dr"))

    assert gap(result.dipole, reference["dipole"]) < 1e-12
    for key in ("atomic_dipoles", "charges", "polarizability", "polarizability_sh"):
        assert gap(result.extras[key], reference[key]) < 1e-12, key
    for key in ("dmu_dr", "dalpha_dr"):
        assert gap(result.extras[key], reference[key]) < 1e-12, key
    # The predicted charges are fixed to the structure's total.
    assert abs(float(result.extras["charges"].sum()) - total_charge) < 1e-12
    assert float(reference["polarizability_sh"][:, 1:].abs().max()) > TOLERANCE


@pytest.mark.parametrize("family", ["dipole", "dielectric"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_training_gradients_follow_the_weight_conversion(
    isolated, fp64, family, device
):
    """Shared readout gradients obey the conversion Jacobian, including gates."""
    import copy

    from tests.golden.harness import FP64_CPU_REFERENCE

    legacy = (
        legacy_anchor() if family == "dipole" else legacy_dielectric("8x0e+8x1o+8x2e")
    )
    model, config = convert(legacy, family)
    atoms, charges, total = CASES["charged pair"]
    batch = {
        key: value.to(device)
        for key, value in legacy_batch(legacy, atoms, charges, total).items()
    }
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    graph = {
        key: value.to(device)
        for key, value in v1_graph(
            atoms, numbers, config["cutoff"], charges, total
        ).items()
    }
    expected = legacy.to(device)(batch, training=True)
    actual = engine(model.to(device))(graph, compute=(), training=True)
    names = ["dipole", *(["polarizability"] if family == "dielectric" else [])]
    predictions = {"dipole": actual.dipole, **actual.extras}
    for name in names:
        torch.testing.assert_close(
            predictions[name],
            expected[name],
            atol=FP64_CPU_REFERENCE.atol,
            rtol=FP64_CPU_REFERENCE.rtol,
        )
    sum(expected[name].square().sum() for name in names).backward()
    sum(predictions[name].square().sum() for name in names).backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in legacy.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.parameters()
    )
    for seed in (12, 43):
        tangent = copy.deepcopy(legacy).cpu()
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in tangent.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
        mapped, _ = convert(tangent, family)
        # The scalar-only final layer of the dipole anchor is unreachable.
        # An absent autograd gradient contributes zero to the directional derivative.
        oracle = sum(
            (parameter.grad * direction.to(device)).sum()
            for parameter, direction in zip(
                legacy.parameters(), tangent.parameters(), strict=True
            )
            if parameter.grad is not None
        )
        gradient = sum(
            (parameter.grad * direction.to(device)).sum()
            for parameter, direction in zip(
                model.parameters(), mapped.parameters(), strict=True
            )
            if parameter.grad is not None
        )
        torch.testing.assert_close(
            gradient, oracle, atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol
        )
