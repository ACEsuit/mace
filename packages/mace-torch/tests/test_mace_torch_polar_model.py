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
from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.config.model import ModelConfig
from mace_core.config.precision import PrecisionConfig
from mace_core.electrostatics import (
    ENTRY_POINT_GROUPS,
    SolverCapabilities,
    UnsupportedSolveError,
)
from mace_core.electrostatics import registry as registry_module
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.electrostatics import ReferenceSolver
from mace_torch.kernels.initialization import initialize_model_weights
from mace_torch.models import EnergyOutputHead, ScaleShiftSpec
from mace_torch.models.electrostatics import PolarModel
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


def build_model(
    profile="molecular",
    steps=2,
    backend=None,
    head_precision: PrecisionConfig | None = None,
    supports_float64=True,
    **overrides,
) -> PolarModel:
    head = EnergyOutputHead(
        ResolvedE0s({"default": {1: -13.6, 8: -2040.0}}),
        ["default"],
        AtomicNumberTable([1, 8]),
        ScaleShiftSpec("std", (1.0,), (0.0,)),
        head_precision or PrecisionConfig(model="float64"),
        supports_float64=supports_float64,
    )
    model = PolarModel(
        backend or ReferenceBackend(),
        config=ModelConfig(
            interaction_first="plain",
            num_interactions=2,
            num_channels=2,
            max_ell=1,
            hidden_irreps="0e+1o",
            radial_basis={"kind": "bessel", "num_basis": 4},
            r_max=5.0,
            correlation=2,
            readout={"mlp_irreps": "4x0e"},
            use_agnostic_product=True,
            polar={
                "multipole_max_l": 1,
                "feature_max_l": 1,
                "feature_widths": (1.0, 1.5),
                "num_recursion_steps": steps,
                "add_local_electron_energy": True,
            },
        ),
        atomic_numbers=[1, 8],
        observables=[WITH_STRESS],
        energy_head=head,
        electrostatics=ElectrostaticsConfig(
            enabled=True,
            solver=overrides.pop("solver", "reference"),
            periodicity_profile=profile,
            kspace_cutoff_factor=1.0,
        ),
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


def float32_molecule(device="cpu", charge=0.0, spin=1.0):
    graph = polar_graph(*molecule(), charge=charge, spin=spin)
    for name, value in graph.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.float32 if value.is_floating_point() else value.dtype
            graph[name] = value.to(device=device, dtype=dtype)
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


@pytest.mark.parametrize(
    ("precision", "accumulate", "supports_float64", "expected"),
    [
        ("float32", "float64", True, torch.float64),
        ("float32", "float64", False, torch.float32),
        ("float64", "float32", True, torch.float64),
    ],
)
def test_charge_reductions_use_the_resolved_precision_floor(
    precision, accumulate, supports_float64, expected
):
    model = build_model(
        head_precision=PrecisionConfig(model=precision, accumulate=accumulate),
        supports_float64=supports_float64,
        precision=precision,
    )
    assert model.total_dtype is expected
    assert getattr(torch, model.descriptor.precision) is expected


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_float32_fukui_projection_conserves_representable_spin_totals(device):
    model = build_model(
        head_precision=PrecisionConfig.float32(), precision="float32"
    ).to(device)
    # The second neutral dimer carries two unpaired electrons. All four
    # target sums survive a single final rounding of the monopoles.
    batch = torch.tensor([0, 0, 1, 1], device=device)
    density = torch.tensor(
        [[100000, 300000], [-50000, 100000], [12, 20], [-2, 4]],
        dtype=torch.float32,
        device=device,
    )
    fukui = density.new_tensor([[1, 1], [2, 2], [3, 2], [1, 3]])
    target = torch.zeros_like(fukui)
    target[2:] = density.new_tensor([1, -1])
    multipoles = torch.stack([density, 2 * density], dim=-1)
    corrected, weights = model._fix_totals(multipoles, fukui, target, batch, 2)
    assert corrected.dtype == weights.dtype == torch.float32
    torch.testing.assert_close(corrected[:, :, 1], multipoles[:, :, 1], atol=0, rtol=0)
    expected = density.new_tensor(
        [
            [250000 / 3, 500000 / 3],
            [-250000 / 3, -500000 / 3],
            [5.25, 10],
            [-4.25, -11],
        ],
    )
    torch.testing.assert_close(corrected[:, :, 0], expected, atol=0, rtol=0)
    for start in (0, 2):
        charges = corrected[start : start + 2, :, 0]
        torch.testing.assert_close(charges.sum(0), target[start], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32], indirect=True)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_reported_total_charge_sums_the_emitted_monopoles_in_the_floor(device):
    model = build_model(
        steps=0, head_precision=PrecisionConfig.float32(), precision="float32"
    ).to(device)
    graph = float32_molecule(device, charge=1, spin=2)
    # A unit charge between two opposite, exactly representable monopoles.
    charges = torch.tensor([2**24, 1, -(2**24), 0], device=device, dtype=torch.float32)

    def known_monopoles(_module, _args, out):
        return torch.nn.functional.pad(charges[:, None], (0, out.shape[-1] - 1))

    model.source_maps[0].register_forward_hook(known_monopoles)
    model.source_maps[1].register_forward_hook(
        lambda _m, _args, out: torch.zeros_like(out)
    )
    result = model(graph).extras
    torch.testing.assert_close(result["charges"], charges, atol=0, rtol=0)
    torch.testing.assert_close(
        result["total_charge"], charges.new_ones(1), atol=0, rtol=0
    )


@pytest.mark.parametrize("scale", [1.0, 1e10, 1e19])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_polar_layer_norm_accumulates_without_overflow(scale, device):
    from mace_torch.nn.field import LayerNormMLP

    from tests.golden.harness import FP32

    norm = LayerNormMLP([64, 64, 1], accumulate="float64").float().to(device).net[1]
    assert isinstance(norm, torch.nn.LayerNorm)
    base = torch.randn(
        (2, 64), dtype=torch.float32, generator=torch.Generator().manual_seed(179)
    )

    def evaluate(dtype, candidate):
        value = base.to(device=device, dtype=dtype).requires_grad_()
        inputs = value * scale
        output = (
            norm(inputs)
            if candidate
            else torch.nn.functional.layer_norm(
                inputs,
                norm.normalized_shape,
                norm.weight.double(),
                norm.bias.double(),
                norm.eps,
            )
        )
        direction = torch.linspace(
            -0.7, 0.8, output.numel(), dtype=dtype, device=device
        ).reshape(output.shape)
        gradient = torch.autograd.grad(
            (output * direction).sum(), value, create_graph=True
        )[0]
        hessian = torch.autograd.grad((gradient * direction.flip(-1)).sum(), value)[0]
        return output, gradient, hessian

    expected = evaluate(torch.float64, False)
    actual = evaluate(torch.float32, True)
    for value, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(
            value.double(), reference, atol=FP32.atol, rtol=FP32.rtol
        )
    assert norm.weight.dtype == torch.float32


def test_float32_polar_electrostatics_honours_the_reduction_floor():
    from mace_torch.electrostatics import long_range_geometry

    from tests.golden.harness import FP32

    model = build_model(
        head_precision=PrecisionConfig(model="float32"), precision="float32"
    )
    control = build_model(precision="float64")
    graph = float32_molecule()
    source = torch.randn(
        (len(graph["positions"]), 4),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(11),
    )
    cache = model.projection.prepare(long_range_geometry(graph, model.descriptor))
    expected_cache = control.projection.prepare(
        long_range_geometry(graph, control.descriptor)
    )
    actual = model.projection(cache, source)
    expected = control.projection(expected_cache, source.double())
    assert actual.dtype == torch.float32
    assert model.descriptor.precision == "float64"
    torch.testing.assert_close(
        actual.double(), expected, atol=FP32.atol, rtol=FP32.rtol
    )
    shifted = torch.nextafter(source, torch.full_like(source, float("inf")))
    residual = (
        model.projection(cache, shifted)
        - actual
        - model.projection(cache, shifted - source)
    )
    torch.testing.assert_close(
        residual, torch.zeros_like(residual), atol=FP32.atol, rtol=FP32.rtol
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_float32_force_training_reaches_the_backbone_and_density_updates(device):
    model = build_model(
        head_precision=PrecisionConfig(model="float32"), precision="float32"
    ).to(device)
    graph = float32_molecule(device)
    graph["positions"].requires_grad_()
    energy = model(graph).total_energy
    assert energy is not None
    forces = -torch.autograd.grad(energy.sum(), graph["positions"], create_graph=True)[
        0
    ]
    forces.square().mean().backward()
    for module in (model.backbone, model.source_maps, model.updates):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients)
