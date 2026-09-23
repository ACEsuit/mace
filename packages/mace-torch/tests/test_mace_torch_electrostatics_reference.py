"""The in-tree reference long-range solver.

Its numbers are pinned against the original ``graph_longrange``, the code it
was taken from, through values that code wrote with ``e3nn`` beside it
(``make_electrostatics_reference.py``). The rest is what the solver promises
the models built on it: a second derivative for force training, the cell the
graph builder chose in each of its three regimes, a stress that is the
derivative of the energy it computed, and capabilities that say what it does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.data.configuration import Configuration
from mace_core.electrostatics import ElectrostaticsSolverDescriptor
from mace_core.elements import AtomicNumberTable
from mace_torch.data import collate_training
from mace_torch.data.graphs import graph_from_configuration
from mace_torch.electrostatics import (
    ReferenceSolver,
    build_long_range,
    reciprocal_cell_and_volume,
)
from mace_torch.electrostatics.reference.energy import GTOElectrostaticEnergy
from mace_torch.electrostatics.reference.gto_utils import gto_basis_kspace_cutoff
from mace_torch.electrostatics.reference.kspace import compute_k_vectors_flat

REFERENCE = json.loads(
    (Path(__file__).with_name("electrostatics_reference.json")).read_text()
)

#: The profile whose op is the reference's evaluation in each recorded mode.
PROFILE_OF_MODE = {
    "pbc": ("full_periodic", None),
    "slab": ("z_slab", 2),
    "realspace": ("molecular", None),
    "mixed_periodic": ("partial", None),
}

TABLE = AtomicNumberTable([1, 8])
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def descriptor(profile: Any, slab_normal: int | None = None, **settings):
    return ElectrostaticsSolverDescriptor(
        profile,
        multipole_max_l=settings.pop("max_l", 1),
        kspace_cutoff=settings.pop(
            "kspace_cutoff", 1.5 * gto_basis_kspace_cutoff([1.0], 1)
        ),
        smearing_width=settings.pop("sigma", 1.0),
        slab_normal=slab_normal,
        **settings,
    )


def graph_of(case: dict) -> dict:
    return {
        "positions": torch.tensor(case["positions"]),
        "batch": torch.tensor(case["batch"]),
        "cell": torch.tensor(case["cell"], dtype=torch.float64),
        "pbc": torch.tensor(case["pbc"]),
    }


def graph_from_builder(positions, cell, pbc, cutoff: float = 5.0) -> dict:
    configuration = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.asarray(positions, dtype=float),
        cell=np.asarray(cell, dtype=float),
        pbc=pbc,
    )
    item = graph_from_configuration(configuration, cutoff=cutoff, z_table=TABLE)
    return dict(collate_training([(item, {}, {})], z_table=TABLE).graph)


#: A neutral water's multipoles, charges then dipoles, in the component order.
MULTIPOLES = torch.tensor(
    [[-0.8, 0.1, 0.0, 0.05], [0.4, 0.0, 0.02, 0.0], [0.4, 0.01, 0.0, 0.0]],
    dtype=torch.float64,
)


# ---------------------------------------------------------------------------
# Against the original
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize("name", sorted(REFERENCE["cases"]))
def test_the_in_tree_solver_reproduces_the_original(name):
    """Energy and its three gradients, within the rounding of two orders of
    summation. Measured at 3.6e-15 on values of order one to thirty."""
    case = REFERENCE["cases"][name]
    positions = torch.tensor(case["positions"], requires_grad=True)
    feats = torch.tensor(case["source_feats"], requires_grad=True)
    cell = torch.tensor(case["cell"], dtype=torch.float64, requires_grad=True)
    solver = GTOElectrostaticEnergy(
        density_max_l=REFERENCE["max_l"],
        density_smearing_width=REFERENCE["sigma"],
        kspace_cutoff=case["kspace_cutoff"],
        pbc_handling=case["mode"],
    )
    rcell, volume = reciprocal_cell_and_volume(cell)
    k_vectors, k_norm2, k_batch, k0_mask = compute_k_vectors_flat(
        case["kspace_cutoff"], cell, rcell
    )
    energy = solver(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_batch,
        k0_mask=k0_mask,
        source_feats=feats,
        node_positions=positions,
        batch=torch.tensor(case["batch"]),
        volume=volume,
        pbc=torch.tensor(case["pbc"]),
    )
    gradients = torch.autograd.grad(
        energy.sum(), [positions, feats, cell], allow_unused=True
    )
    torch.testing.assert_close(
        energy.detach(), torch.tensor(case["energy"]), rtol=1e-13, atol=1e-13
    )
    for key, value, leaf in zip(
        ("d_positions", "d_source_feats", "d_cell"),
        gradients,
        (positions, feats, cell),
        strict=True,
    ):
        got = torch.zeros_like(leaf) if value is None else value
        torch.testing.assert_close(got, torch.tensor(case[key]), rtol=1e-12, atol=1e-13)


@fp64_only
@pytest.mark.parametrize("name", sorted(REFERENCE["cases"]))
def test_each_profile_s_op_is_the_original_s_evaluation(name):
    case = REFERENCE["cases"][name]
    profile, normal = PROFILE_OF_MODE[case["mode"]]
    op = build_long_range(
        descriptor(profile, normal, kspace_cutoff=case["kspace_cutoff"])
    )
    energy = op(graph_of(case), torch.tensor(case["source_feats"]))
    torch.testing.assert_close(
        energy.detach(), torch.tensor(case["energy"]), rtol=1e-13, atol=1e-13
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize("profile", ["full_periodic", "z_slab", "molecular", "partial"])
def test_the_solve_is_differentiable_twice(profile):
    """What force training needs: the gradient of a force against the
    multipoles and the geometry, checked against finite differences."""
    case = REFERENCE["cases"][
        {"full_periodic": "pbc", "z_slab": "slab", "molecular": "realspace"}.get(
            profile, "mixed_periodic"
        )
    ]
    op = build_long_range(
        descriptor(profile, 2 if profile == "z_slab" else None),
        trains_derivatives=True,
    )
    graph = graph_of(case)

    def energy(positions, feats, cell):
        return op({**graph, "positions": positions, "cell": cell}, feats).sum()

    leaves = (
        graph["positions"].clone().requires_grad_(True),
        torch.tensor(case["source_feats"]).requires_grad_(True),
        graph["cell"].clone().requires_grad_(True),
    )
    assert torch.autograd.gradgradcheck(energy, leaves, eps=1e-6, atol=1e-5)


@fp64_only
def test_the_stress_is_the_derivative_of_the_energy_it_computed():
    """The reciprocal cell and the volume are derived from the cell the solve
    is handed, so a strain of that cell reaches both. Taken from the cell
    before the strain, as the frozen tree took them, the stress misses their
    part and is wrong by tens of percent in a small box."""
    op = build_long_range(descriptor("full_periodic"))
    graph = graph_from_builder(WATER + 0.5, np.eye(3) * 6.0, (True, True, True))

    def strained(strain: torch.Tensor) -> torch.Tensor:
        symmetric = 0.5 * (strain + strain.T)
        deform = torch.eye(3, dtype=torch.float64) + symmetric
        return op(
            {
                **graph,
                "positions": graph["positions"] @ deform,
                "cell": graph["cell"].view(3, 3) @ deform,
            },
            MULTIPOLES,
        ).sum()

    strain = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    (analytic,) = torch.autograd.grad(strained(strain), [strain])
    step = 1e-5
    numerical = torch.zeros(3, 3, dtype=torch.float64)
    for i in range(3):
        for j in range(3):
            plus = torch.zeros(3, 3, dtype=torch.float64)
            plus[i, j] = step
            numerical[i, j] = (strained(plus) - strained(-plus)) / (2 * step)
    torch.testing.assert_close(analytic, numerical, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# The cell the graph carries
# ---------------------------------------------------------------------------


def test_an_open_molecule_reaches_the_solve_in_the_box_around_it():
    """The regime the long-range models depend on: the box the graph builder
    made, with a finite volume, rather than a degenerate or identity cell."""
    graph = graph_from_builder(WATER + 40.0, np.zeros((3, 3)), (False, False, False))
    cell = graph["cell"].view(3, 3)
    rcell, volume = reciprocal_cell_and_volume(cell)
    extent = WATER.max(axis=0) - WATER.min(axis=0)
    np.testing.assert_allclose(np.diag(cell.numpy()), extent + 2 * 5.0 + 1.0)
    assert float(volume) > 0 and torch.isfinite(rcell).all()


@fp64_only
def test_a_molecule_in_its_box_converges_on_the_open_molecule():
    """Among periodic structures a molecule is summed in k-space in its box,
    and the monopole correction divides by that box's volume. With the right
    volume the result tends to the exact open-boundary sum as the box grows;
    a wrong one would leave it off by a constant."""
    charges = MULTIPOLES.clone()
    charges[:, 1:] = 0.0
    exact = build_long_range(descriptor("molecular"))
    boxed = build_long_range(descriptor("partial"))
    gaps = []
    for cutoff in (5.0, 10.0, 20.0):
        graph = graph_from_builder(WATER, np.zeros((3, 3)), (False,) * 3, cutoff)
        gaps.append(float(abs(boxed(graph, charges) - exact(graph, charges))))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-5


def test_a_slab_reaches_the_solve_with_its_physical_cell():
    """The search box's vacuum row is not the slab's, and its volume is not
    the one the stress divides by."""
    cell = np.diag([6.0, 6.0, 20.0])
    graph = graph_from_builder(WATER + 2.0, cell, (True, True, False))
    np.testing.assert_array_equal(graph["cell"].view(3, 3).numpy(), cell)


def test_a_slab_with_no_vacuum_gets_a_finite_volume_and_reciprocal_cell():
    graph = graph_from_builder(
        WATER + 2.0, np.diag([6.0, 6.0, 0.0]), (True, True, False)
    )
    rcell, volume = reciprocal_cell_and_volume(graph["cell"])
    assert float(volume) > 0 and torch.isfinite(rcell).all()


def test_a_cell_with_no_volume_has_a_zero_reciprocal_cell():
    rcell, volume = reciprocal_cell_and_volume(torch.zeros(1, 3, 3))
    assert float(volume) == 0 and not rcell.any()


# ---------------------------------------------------------------------------
# What it declares
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["full_periodic", "molecular", "partial"])
def test_the_reference_supports_every_profile(profile):
    assert ReferenceSolver.capabilities.supports(descriptor(profile))


def test_the_reference_slab_correction_acts_along_z_only():
    """The original corrects along z and nothing else, so another normal is a
    solve it declines rather than one it quietly does along z."""
    capabilities = ReferenceSolver.capabilities
    assert capabilities.supports(descriptor("z_slab", 2))
    assert not capabilities.supports(descriptor("z_slab", 0))


def test_an_option_the_reference_does_not_know_is_declined():
    unknown = descriptor("full_periodic", external_field_flags=frozenset({"field"}))
    assert not ReferenceSolver.capabilities.supports(unknown)


def test_the_reference_is_twice_differentiable_and_bit_for_bit_itself():
    assert ReferenceSolver.capabilities.supports_double_backward
    assert ReferenceSolver.capabilities.bit_parity
