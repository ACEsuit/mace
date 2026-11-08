"""ELEC-3: local sources, conservative transfers and physical derivatives."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
import torch
from conftest import TOLERANCES, fp64_only
from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.config.model import ModelConfig
from mace_core.config.precision import PrecisionConfig
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import Precision
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.kernels import initialize_model_weights
from mace_torch.models import EnergyOutputHead, ScaleShiftSpec
from mace_torch.models.electrostatics import LocalChargeModel
from mace_torch.models.registry import registered_models
from mace_torch.physics import DerivativeEngine
from mace_torch.serialization import canonical_state, load_canonical_state
from mace_torch_engine_fixtures import ENERGY, build_graph, molecule
from scipy.spatial.transform import Rotation
from test_mace_torch_polar_training import fixture_registered as fixture_registered

FAMILIES = ("lc_mace", "lsc_mace")
ENERGY_STRESS = ENERGY.model_copy(
    update={
        "derivatives": (
            *ENERGY.derivatives,
            {"wrt": "strain", "name": "stress", "sign": 1, "units": "eV/Å^3"},
        )
    }
)


@fp64_only
def test_registration_refuses_an_electrostatic_family_without_a_solver(monkeypatch):
    from mace_torch.models import registry
    from mace_torch.models.base import MACEModel

    monkeypatch.setattr(registry, "MODEL_REGISTRY", {})

    class MissingSolver(MACEModel):
        requires_electrostatics = True

    with pytest.raises(TypeError, match="must define solver_descriptor"):
        registry.register_model("missing_solver")(MissingSolver)
    assert "missing_solver" not in registry.MODEL_REGISTRY


def model_for(
    kind,
    *,
    degree=1,
    profile="molecular",
    per_atom=False,
    backend=None,
    solver="reference",
    trains_derivatives=False,
    options=None,
):
    precision = cast(Precision, str(torch.get_default_dtype()).removeprefix("torch."))
    sources = {"kind": kind, "multipole_max_l": degree}
    if kind == "lsc_mace":
        sources.update(
            {"formal_charge_input": "oxidation_state"}
            if per_atom
            else {"formal_charges": {1: 1.0, 8: -2.0}}
        )
    sources.update(options or {})
    config = ModelConfig(
        model=kind,
        local_sources=sources,
        num_channels=2,
        num_interactions=2,
        max_L=max(1, degree),
        max_ell=max(1, degree),
        correlation=2,
        radial_mlp=(8,),
        radial_basis={"kind": "bessel", "num_basis": 4},
        scaling="none",
        readout={"mlp_irreps": "4x0e"},
        interaction_first="plain",
    )
    head = EnergyOutputHead(
        ResolvedE0s({"default": {1: 0.0, 8: 0.0}}),
        ["default"],
        AtomicNumberTable([1, 8]),
        ScaleShiftSpec("none", (1.0,), (0.0,)),
        PrecisionConfig(model=precision),
        widen_node_energy=False,
    )
    model = cast(type[LocalChargeModel], registered_models()[kind])(
        backend or ReferenceBackend(),
        config,
        [1, 8],
        [ENERGY_STRESS],
        head,
        settings=LocalChargeModel.solver_descriptor(
            config,
            ElectrostaticsConfig(
                enabled=True,
                periodicity_profile=profile,
                slab_normal=2 if profile == "z_slab" else None,
            ),
            precision="float64",
        ),
        precision=precision,
        solver=solver,
        trains_derivatives=trains_derivatives,
    )
    initialize_model_weights(model, 3)
    return model


def graph_for(profile="molecular", *, shift=None, rotation=None, permutation=None):
    positions, numbers = molecule()
    cell = np.eye(3) * 9
    pbc = (
        (False, False, False)
        if profile == "molecular"
        else ((True, True, False) if profile == "z_slab" else (True, True, True))
    )
    if rotation is not None:
        positions = positions @ rotation.T
        cell = cell @ rotation.T
    if shift is not None:
        positions += shift
    if permutation is not None:
        positions, numbers = positions[permutation], np.array(numbers)[permutation]
    graph = build_graph(positions, numbers, cell, pbc)
    graph = {
        key: value.to(torch.get_default_dtype())
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for key, value in graph.items()
    }
    graph["external_field"] = torch.zeros((1, 3))
    graph["oxidation_state"] = torch.tensor([-2.0 if z == 8 else 1.0 for z in numbers])
    return graph


@pytest.mark.parametrize("kind", FAMILIES)
@pytest.mark.parametrize("degree", [0, 1])
def test_registry_energy_decomposition_and_checkpoint(kind, degree):
    model = model_for(kind, degree=degree)
    graph = graph_for()
    graph["external_field"] = torch.tensor([[0.02, -0.01, 0.03]])
    result = model(graph)
    local = model.outputs(graph, list(model.backbone(graph).layers), None)
    torch.testing.assert_close(
        result.total_energy,
        local.total_energy
        + result.extras["electrostatic_energy"]
        + result.extras["field_energy"],
    )
    torch.testing.assert_close(result.node_energies, local.node_energies)
    rebuilt = model_for(kind, degree=degree)
    load_canonical_state(rebuilt, canonical_state(model))
    torch.testing.assert_close(
        rebuilt(graph).total_energy, result.total_energy, atol=0, rtol=0
    )
    assert (
        ModelConfig.model_validate_json(model.config.model_dump_json()) == model.config
    )


@pytest.mark.parametrize("kind", FAMILIES)
@pytest.mark.parametrize("degree", [0, 1])
def test_translation_permutation_and_rotation(kind, degree):
    model = model_for(kind, degree=degree)
    # Reductions retain the configured accumulation dtype. Their tolerance
    # follows the model/geometry precision, not the dtype of the final scalar.
    atol, rtol = TOLERANCES[torch.get_default_dtype()]
    graph = graph_for()
    result = model(graph)
    translated = model(graph_for(shift=np.array([1.0, -0.3, 0.2])))
    torch.testing.assert_close(translated.total_energy, result.total_energy)
    torch.testing.assert_close(
        translated.dipole,
        result.dipole
        + result.extras["total_charge"][:, None]
        * torch.tensor([[1.0, -0.3, 0.2]], dtype=result.dipole.dtype),
    )
    permutation = np.array([3, 1, 0, 2])
    permuted = model(graph_for(permutation=permutation))
    torch.testing.assert_close(permuted.total_energy, result.total_energy)
    torch.testing.assert_close(
        permuted.extras["charges"], result.extras["charges"][permutation]
    )
    rotation = Rotation.from_rotvec([0.3, -0.2, 0.4]).as_matrix()
    rotated = model(graph_for(rotation=rotation))
    if degree == 0:
        torch.testing.assert_close(
            rotated.total_energy, result.total_energy, atol=atol, rtol=rtol
        )
    # With dipoles, the reference's axis-displaced finite-difference solver
    # is not exactly rotation invariant. Rotated energies/forces are compared
    # to that same oracle in test_elec3_goldens; the sources must be equivariant.
    torch.testing.assert_close(
        rotated.extras["charges"], result.extras["charges"], atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        rotated.dipole,
        result.dipole @ torch.tensor(rotation, dtype=result.dipole.dtype).T,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("degree", [0, 1])
def test_lsc_conserves_charge_per_component_and_reads_declared_inputs(degree):
    model = model_for("lsc_mace", degree=degree, per_atom=True)
    graph = graph_for()
    # Two disconnected fragments plus an isolated node; each keeps its formal total.
    keep = (graph["edge_index"][0] < 3) & (graph["edge_index"][1] < 3)
    for key in ("edge_index", "shifts", "unit_shifts"):
        graph[key] = graph[key][:, keep] if key == "edge_index" else graph[key][keep]
    output = model(graph)
    charges = output.extras["charges"]
    torch.testing.assert_close(charges[:3].sum(), graph["oxidation_state"][:3].sum())
    torch.testing.assert_close(charges[3], graph["oxidation_state"][3])
    assert any(
        spec.name == "oxidation_state" and spec.per_atom for spec in model.input_specs
    )
    assert model.config.feature_storages["oxidation_state"] == "atom"


@pytest.mark.parametrize("kind", FAMILIES)
@pytest.mark.parametrize("profile", ["molecular", "full_periodic", "z_slab"])
@fp64_only
def test_force_and_stress_finite_difference_and_double_backward(kind, profile):
    model = model_for(kind, profile=profile)
    graph = graph_for(profile)
    engine = DerivativeEngine(model, ENERGY_STRESS)
    output = engine(graph, compute=("forces", "stress"), training=True)
    step = 1.0e-5

    def energy(displacement=0.0, strain=0.0):
        changed = dict(graph)
        change = torch.zeros_like(graph["positions"])
        change[1, 0] = displacement
        deformation = torch.eye(3)
        deformation[0, 0] += strain
        changed["positions"] = (graph["positions"] + change) @ deformation
        changed["cell"] = graph["cell"] @ deformation
        changed["shifts"] = graph["shifts"] @ deformation
        return model(changed).total_energy.sum()

    force = -(energy(displacement=step) - energy(displacement=-step)) / (2 * step)
    torch.testing.assert_close(force, output.forces[1, 0], atol=1.0e-6, rtol=0)
    if profile != "molecular":
        stress = (
            (energy(strain=step) - energy(strain=-step))
            / (2 * step)
            / torch.linalg.det(graph["cell"]).abs().sum()
        )
        torch.testing.assert_close(stress, output.stress[0, 0, 0], atol=1.0e-6, rtol=0)
    loss = output.forces.square().sum() + output.stress.square().sum()
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert any(
        p.grad is not None and p.grad.count_nonzero()
        for p in model.source_maps.parameters()
    )


@fp64_only
@pytest.mark.parametrize("kind", FAMILIES)
def test_gradgradcheck_positions(kind):
    model = model_for(kind, degree=0)
    graph = graph_for()

    def energy(positions):
        return model(dict(graph, positions=positions)).total_energy

    positions = graph["positions"].clone().requires_grad_(True)
    assert torch.autograd.gradgradcheck(energy, (positions,), fast_mode=True)


@fp64_only
def test_lsc_periodic_polarization_tracks_formal_charge_transport():
    model = model_for("lsc_mace", profile="full_periodic")
    graph = graph_for("full_periodic")
    original = model(graph)
    # Unwrap a charged atom by a lattice vector; preserve physical bonds.
    moved = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in graph.items()
    }
    lattice = graph["cell"][0, 0]
    moved["positions"][1] += lattice
    sender, receiver = graph["edge_index"]
    moved["shifts"] += (sender == 1)[:, None] * lattice - (receiver == 1)[
        :, None
    ] * lattice
    result = model(moved)
    torch.testing.assert_close(
        result.extras["charges"], original.extras["charges"], atol=1.0e-12, rtol=1.0e-12
    )
    torch.testing.assert_close(
        result.total_energy, original.total_energy, atol=1.0e-6, rtol=0
    )
    torch.testing.assert_close(
        result.dipole - original.dipole, lattice[None], atol=1.0e-12, rtol=1.0e-12
    )
    moved["external_field"] = torch.tensor([[0.03, -0.01, 0.02]])
    graph["external_field"] = moved["external_field"]
    torch.testing.assert_close(
        model(moved).total_energy - model(graph).total_energy,
        (lattice * moved["external_field"][0]).sum().reshape(1),
        atol=1.0e-6,
        rtol=0,
    )


@fp64_only
@pytest.mark.parametrize("kind", FAMILIES)
@pytest.mark.parametrize("solver", ["reference", "accelerated"])
def test_training_optimizer_members_and_deployment_roundtrip(
    tmp_path, kind, solver, registered
):
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.deploy.loader import load_deployed
    from mace_torch.train import run_data_stage, run_model_stage, run_train_stage
    from mace_torch.train.optimizers import build_optimizer
    from test_mace_torch_full_batch import configuration

    document = configuration(tmp_path, max_num_epochs=1).model_dump(mode="json")
    document["model"].update(
        model=kind,
        local_sources={
            "kind": kind,
            "multipole_max_l": 1,
            **(
                {"formal_charge_input": "oxidation_state"} if kind == "lsc_mace" else {}
            ),
        },
    )
    document["electrostatics"].update(
        enabled=True, periodicity_profile="molecular", solver=solver
    )
    if kind == "lsc_mace":
        from ase.io import read, write

        path = document["data"]["heads"]["default"]["train_file"]
        atoms = read(path, index=":")
        for item in atoms:
            item.arrays["oxidation"] = np.where(item.numbers == 8, -2.0, 1.0)
        write(path, atoms, format="extxyz")
        document["data"]["embedding_specs"] = {
            "oxidation_state": {"per": "atom", "key": "oxidation"}
        }
    config = ResolvedConfig.model_validate(document)
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    parameters = {
        id(parameter)
        for parameter in built.model.parameters()
        if parameter.requires_grad
    }
    optimized = [
        id(parameter)
        for group in build_optimizer(built.model, config.training).param_groups
        for parameter in group["params"]
    ]
    assert set(optimized) == parameters and len(optimized) == len(parameters)
    local_model = cast(LocalChargeModel, built.model.get_submodule("backbone"))
    original = [p.detach().clone() for p in local_model.source_maps.parameters()]
    trained = run_train_stage(config, built, checkpoint_path=tmp_path / "local")
    assert np.isfinite(trained.history[-1].train_loss)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            original, local_model.source_maps.parameters(), strict=True
        )
    )
    deployed = load_deployed(tmp_path / "local.json")
    assert deployed.metadata.config.resolved["electrostatics"]["solver"] == solver
    assert deployed.metadata.electrostatics is not None
    assert deployed.metadata.electrostatics.solver == solver
    assert deployed.metadata.electrostatics.bit_parity is (solver == "reference")
    from mace_core.electrostatics import SolverSubstitutionError

    substitute = "accelerated" if solver == "reference" else "reference"
    with pytest.raises(SolverSubstitutionError):
        load_deployed(tmp_path / "local.json", solver=substitute)
    graph = next(data.train_loader.batches(0, drop_last=False)).graph
    expected = deployed.compute(graph)
    # Loading the same canonical checkpoint twice preserves all values exactly.
    restored = load_deployed(tmp_path / "local.json").compute(graph)
    for name in ("total_energy", "forces", "dipole"):
        torch.testing.assert_close(
            getattr(restored, name), getattr(expected, name), atol=0, rtol=0
        )


@fp64_only
@pytest.mark.parametrize("kind", FAMILIES)
def test_planned_forward_compiles_without_graph_breaks(kind):
    from mace_torch.electrostatics import LONG_RANGE_PLAN, long_range_plan

    model = model_for(kind, degree=0, profile="full_periodic")
    graph = graph_for("full_periodic")
    graph[LONG_RANGE_PLAN] = long_range_plan(graph, model.descriptor, dense_pairs=True)
    expected = model(graph)
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    actual = compiled(graph)
    torch.testing.assert_close(actual.total_energy, expected.total_energy)
    torch.testing.assert_close(actual.dipole, expected.dipole)


@pytest.mark.parametrize(
    "settings",
    [
        {"kind": "lsc_mace"},
        {
            "kind": "lsc_mace",
            "formal_charges": {1: 1},
            "formal_charge_input": "oxidation",
        },
        {"kind": "lsc_mace", "formal_charge_input": "charges"},
        {"kind": "lc_mace", "multipole_max_l": 2},
        {"kind": "lc_mace", "unused_option": True},
    ],
)
def test_invalid_source_settings_fail_before_build(settings):
    with pytest.raises(ValueError):
        ModelConfig(model=settings["kind"], local_sources=settings)


@fp64_only
@pytest.mark.parametrize("kind", FAMILIES)
def test_solver_is_built_once_and_sees_the_graph_cell(kind, monkeypatch):
    from mace_core.electrostatics import registry as solver_registry

    model = model_for(kind, profile="partial")

    def unexpected(*args, **kwargs):
        raise AssertionError("solver lookup during forward")

    monkeypatch.setattr(solver_registry, "get_solver", unexpected)
    seen = []
    original = model.coulomb.forward

    def record(graph, density, geometry=None):
        seen.append(graph["cell"].clone())
        return original(graph, density, geometry)

    monkeypatch.setattr(model.coulomb, "forward", record)
    molecule_graph = graph_for()
    slab_graph = graph_for("z_slab")
    model(molecule_graph)
    model(slab_graph)
    torch.testing.assert_close(seen[0], molecule_graph["cell"])
    torch.testing.assert_close(seen[1], slab_graph["cell"])
    assert torch.linalg.det(seen[0]).abs().item() > 0
    assert torch.equal(seen[1][0], torch.eye(3) * 9)


@fp64_only
@pytest.mark.parametrize("kind", FAMILIES)
def test_inference_solver_is_rejected_for_force_training(kind, monkeypatch):
    from mace_core.electrostatics import UnsupportedSolveError
    from mace_core.electrostatics import registry as solver_registry
    from mace_torch.electrostatics import ReferenceSolver
    from test_mace_torch_polar_training import _EntryPoint

    class InferenceOnly(ReferenceSolver):
        name = "inference-only"
        capabilities = replace(
            ReferenceSolver.capabilities, supports_double_backward=False
        )

    monkeypatch.setattr(
        solver_registry,
        "entry_points",
        lambda **kwargs: [_EntryPoint("inference-only", InferenceOnly)],
    )
    with pytest.raises(UnsupportedSolveError, match="differentiated twice"):
        model_for(kind, solver="inference-only", trains_derivatives=True)
    assert torch.isfinite(
        model_for(kind, solver="inference-only")(graph_for()).total_energy
    ).all()


@fp64_only
@pytest.mark.parametrize("kind", FAMILIES)
def test_no_neighbors_is_a_finite_model(kind):
    model = model_for(kind)
    graph = graph_for()
    graph["edge_index"] = graph["edge_index"][:, :0]
    graph["shifts"] = graph["shifts"][:0]
    graph["unit_shifts"] = graph["unit_shifts"][:0]
    result = DerivativeEngine(model, ENERGY_STRESS)(
        graph, compute=("forces",), training=True
    )
    assert torch.isfinite(result.total_energy).all()
    assert torch.isfinite(result.forces).all()
    if kind == "lsc_mace":
        torch.testing.assert_close(result.extras["charges"], graph["oxidation_state"])
