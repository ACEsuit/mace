"""The charge-aware model on its own terms, with no legacy code in sight.

Its numbers against the frozen tree are pinned in ``tests/parity``. What is
pinned here is what the model promises regardless of any oracle: the density
carries the charge and spin it was given, force training has a second
derivative to follow, the long-range ops are resolved once from the solver the
model names, and a periodic energy does not depend on how the cell is turned.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.electrostatics import (
    ENTRY_POINT_GROUPS,
    SolverCapabilities,
    UnsupportedSolveError,
)
from mace_core.electrostatics import registry as registry_module
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.electrostatics import ReferenceSolver
from mace_torch.kernels.initialization import initialize_model_weights
from mace_torch.models import EnergyOutputHead, ScaleShiftSpec
from mace_torch.models.electrostatics import PolarModel, PolarSettings
from mace_torch_engine_fixtures import ENERGY, build_graph, crystal, molecule
from scipy.spatial.transform import Rotation

WITH_STRESS = ENERGY.model_copy(
    update={
        "derivatives": (
            *ENERGY.derivatives,
            {"wrt": "strain", "name": "stress", "sign": 1, "units": "eV/A^3"},
        )
    }
)


def build_model(profile="molecular", steps=2, **overrides) -> PolarModel:
    head = EnergyOutputHead(
        ResolvedE0s({"default": {1: -13.6, 8: -2040.0}}),
        ["default"],
        AtomicNumberTable([1, 8]),
        ScaleShiftSpec("std", (1.0,), (0.0,)),
        PrecisionConfig(model="float64"),
    )
    model = PolarModel(
        ReferenceBackend(),
        atomic_numbers=[1, 8],
        observables=[WITH_STRESS],
        energy_head=head,
        settings=PolarSettings(
            multipole_max_l=1,
            feature_max_l=1,
            feature_widths=(1.0, 1.5),
            num_recursion_steps=steps,
            kspace_cutoff_factor=1.0,
            add_local_electron_energy=True,
            periodicity_profile=profile,
            fukui_hidden=4,
        ),
        num_layers=2,
        num_features=2,
        lmax=1,
        hidden_irreps="0e+1o",
        num_radial=4,
        cutoff=5.0,
        correlation=2,
        readout_hidden=4,
        element_agnostic_product=True,
        **overrides,
    )
    initialize_model_weights(model, seed=3)
    return model


def polar_graph(positions, numbers, cell=None, pbc=(False,) * 3, charge=0.0, spin=1.0):
    graph = build_graph(positions, numbers, cell, pbc)
    graph["total_charge"] = torch.tensor([charge])
    graph["total_spin"] = torch.tensor([spin])
    graph["external_field"] = torch.tensor([[0.01, -0.02, 0.005]])
    return graph


# ---------------------------------------------------------------------------
# What the density carries
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize(("charge", "spin"), [(0.0, 1.0), (-1.0, 2.0), (1.0, 3.0)])
def test_the_density_carries_the_charge_and_spin_it_was_given(charge, spin):
    positions, numbers = molecule()
    result = build_model()(polar_graph(positions, numbers, charge=charge, spin=spin))
    extras = result.extras
    torch.testing.assert_close(
        extras["total_charge"], torch.tensor([charge]), rtol=0, atol=1e-12
    )
    alpha, beta = extras["spin_charge_density"][:, :, 0].sum(dim=0)
    torch.testing.assert_close(alpha - beta, torch.tensor(spin - 1), rtol=0, atol=1e-12)
    torch.testing.assert_close(
        extras["fukui_functions"].sum(dim=0),
        torch.ones(2, dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


@fp64_only
def test_the_energy_is_the_sum_of_its_parts():
    positions, numbers = molecule()
    graph = polar_graph(positions, numbers)
    result = build_model()(graph)
    extras = result.extras
    field = graph["external_field"]
    assert result.total_energy is not None and result.dipole is not None
    parts = (
        extras["interaction_energy"]
        + torch.tensor([-2040.0 - 3 * 13.6])
        + extras["electrostatic_energy"]
        + extras["electron_energy"]
        + (field * result.dipole).sum(dim=-1)
    )
    torch.testing.assert_close(result.total_energy, parts, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize("profile", ["molecular", "full_periodic"])
def test_forces_and_stress_are_differentiable_twice(profile):
    """What force and stress training need, through the electrostatic energy
    and every step of the recursion, against finite differences."""
    if profile == "molecular":
        positions, numbers = molecule()
        cell, pbc = None, (False,) * 3
    else:
        positions, numbers, cell = crystal()
        pbc = (True,) * 3
    model = build_model(profile, steps=1)
    graph = polar_graph(positions, numbers, cell, pbc)

    def energy(positions, cell):
        strained = dict(graph)
        strained["positions"] = positions
        strained["cell"] = cell.reshape(1, 3, 3)
        strained["shifts"] = graph["unit_shifts"].to(cell.dtype) @ cell
        return model(strained).total_energy.sum()

    leaves = (
        graph["positions"].clone().requires_grad_(True),
        graph["cell"].view(3, 3).clone().requires_grad_(True),
    )
    assert torch.autograd.gradgradcheck(energy, leaves, eps=1e-6, atol=1e-5)


class _InferenceOnly:
    name = "fast"
    capabilities = SolverCapabilities(
        ops=frozenset({"long_range_energy", "long_range_features"}),
        supports_double_backward=False,
    )


class _EntryPoint:
    def __init__(self, name, target):
        self.name, self._target = name, target

    def load(self):
        return self._target


@pytest.fixture(name="registered")
def fixture_registered(monkeypatch):
    points = [
        _EntryPoint("reference", ReferenceSolver),
        _EntryPoint("fast", _InferenceOnly),
    ]

    def entry_points(group: str):
        assert group == ENTRY_POINT_GROUPS["torch"]
        return points

    monkeypatch.setattr(registry_module, "entry_points", entry_points)


@fp64_only
def test_a_solver_that_cannot_differentiate_twice_is_refused_for_force_training(
    registered,
):
    with pytest.raises(UnsupportedSolveError, match="differentiated twice"):
        build_model(solver="fast", trains_derivatives=True)


@fp64_only
def test_the_ops_come_from_the_named_solver_and_are_held(registered):
    model = build_model()
    assert model.solver == "reference"
    assert model.projection.solver == model.coulomb.solver == "reference"


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


def _rotated_energies(profile, positions, numbers, cell, pbc):
    model = build_model(profile)
    energies = []
    for seed in range(4):
        rotation = Rotation.random(random_state=seed).as_matrix()
        turned_cell = None if cell is None else cell @ rotation.T
        graph = polar_graph(positions @ rotation.T, numbers, turned_cell, pbc)
        graph["external_field"] = torch.zeros(1, 3, dtype=torch.float64)
        energies.append(float(model(graph).total_energy))
    return np.array(energies)


@fp64_only
def test_a_periodic_energy_does_not_depend_on_how_the_cell_is_turned():
    positions, numbers, cell = crystal()
    energies = _rotated_energies("full_periodic", positions, numbers, cell, (True,) * 3)
    np.testing.assert_allclose(energies, energies[0], rtol=0, atol=1e-10)


@fp64_only
@pytest.mark.xfail(
    strict=True,
    reason=(
        "the finite-difference real-space method displaces charges along the "
        "laboratory axes, so an open molecule's energy depends on its "
        "orientation; every published model was trained with it"
    ),
)
def test_an_open_molecule_energy_does_not_depend_on_its_orientation():
    positions, numbers = molecule()
    energies = _rotated_energies("molecular", positions, numbers, None, (False,) * 3)
    np.testing.assert_allclose(energies, energies[0], rtol=0, atol=1e-10)
