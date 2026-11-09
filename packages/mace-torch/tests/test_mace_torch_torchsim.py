"""TorchSim state translation, shared padding and static compilation."""

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.graph import FixedPadding
from mace_torch.calculators import MACECalculator
from mace_torch.deploy.bundle import export_lammps
from mace_torch.train.checkpoint import write_model
from test_mace_torch_calculator import water
from test_mace_torch_cli_commands import artifact as artifact
from test_mace_torch_polar_training import built, polar_configuration

from tests.golden.harness import FP32, FP64_CPU_REFERENCE

pytestmark = pytest.mark.torchsim


@pytest.fixture(autouse=True)
def require_torchsim():
    if importlib.util.find_spec("torch_sim") is None:
        if "torchsim" in os.environ.get("MACE_REQUIRE_CAPS", "").split(","):
            pytest.fail("torchsim is required but not installed")
        pytest.skip("requires torchsim (Python >= 3.12)")


def close(actual, expected, dtype=torch.float64):
    row = FP32 if dtype == torch.float32 else FP64_CPU_REFERENCE
    if torch.is_tensor(actual):
        actual = actual.detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=row.atol, rtol=row.rtol)


def state_of(atoms, dtype=torch.float64):
    from torch_sim.io import atoms_to_state

    return atoms_to_state(atoms, device=torch.device("cpu"), dtype=dtype)


def adapter(path, **kwargs):
    from mace_torch.deploy.torchsim import MaceTorchSimModel

    return MaceTorchSimModel(path, **kwargs)


def compare_ase(path, state, result):
    from torch_sim.io import state_to_atoms

    offset = 0
    for index, atoms in enumerate(state_to_atoms(state)):
        atoms.calc = MACECalculator(path)
        close(result["energy"][index], atoms.get_potential_energy())
        close(result["forces"][offset : offset + len(atoms)], atoms.get_forces())
        if "stress" in result:
            expected = (
                atoms.get_stress(voigt=False) if atoms.pbc.any() else np.zeros((3, 3))
            )
            close(result["stress"][index], expected)
        offset += len(atoms)
    assert all(not value.requires_grad for value in result.values())


@pytest.mark.parametrize("compute_stress", [True, False])
def test_energy_batched_state_and_triclinic_cell(artifact, dtype, compute_stress):
    atoms = water(True)
    atoms.set_cell([[4, 0, 0], [0.4, 4.3, 0], [0.2, 0.3, 4.5]])
    other = atoms.copy()
    other.positions[2, 0] += 0.2
    state = state_of([atoms, other], dtype)
    model = adapter(
        artifact,
        compute_stress=compute_stress,
        padding=FixedPadding(nodes_budget=8, edges_budget=256, graphs_budget=4),
    )
    with torch.inference_mode():
        result = model(state)
    assert ("stress" in result) == compute_stress
    compare_ase(artifact, state, result)


@fp64_only
def test_one_compile_for_changing_real_atoms_and_neighbors(artifact):
    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()
    model = adapter(
        artifact,
        compile_mode="default",
        compile_backend=counter,
        padding=FixedPadding(nodes_budget=8, edges_budget=128, graphs_budget=2),
    )
    for n in (3, 2, 3, 1):
        atoms = water(False)[:n]
        state = state_of(atoms)
        result = model(state)
        compare_ase(artifact, state, result)
    assert counter.frame_count == 1


@fp64_only
def test_displacement_survives_shared_padding_and_is_optional(artifact):
    model = adapter(
        artifact, padding=FixedPadding(nodes_budget=8, edges_budget=64, graphs_budget=2)
    )
    state = state_of(water(True))
    graph, _ = model._graph(state)
    assert "displacement" not in graph
    displacement = torch.tensor([[[0.01, 0.002, 0], [0.002, -0.01, 0], [0, 0, 0.005]]])
    state.system_extras["displacement"] = displacement
    graph, _ = model._graph(state)
    torch.testing.assert_close(graph["displacement"][:1], displacement, atol=0, rtol=0)
    torch.testing.assert_close(
        graph["displacement"][1:], torch.zeros(1, 3, 3), atol=0, rtol=0
    )
    eager = adapter(artifact)(state)
    padded = model(state)
    for name in eager:
        close(padded[name], eager[name].numpy())
    assert not torch.equal(padded["energy"], model(state_of(water(True)))["energy"])


@pytest.fixture
def polar_artifact(tmp_path, request):
    config = polar_configuration(
        tmp_path, profile=getattr(request, "param", "molecular")
    )
    _, model = built(config)
    write_model(tmp_path / "polar", model.engine, model.metadata)
    return tmp_path / "polar.json"


@fp64_only
@pytest.mark.parametrize(
    "systems,extras", [(1, False), (1, True), (2, False), (2, True)]
)
def test_polar_defaults_fields_and_batched_states(polar_artifact, systems, extras):
    state = state_of([water(False) for _ in range(systems)])
    if extras:
        state.system_extras["total_charge"] = torch.linspace(0.2, 0.4, systems)
        state.system_extras["total_spin"] = torch.ones(systems)
        state.system_extras["external_field"] = torch.full((systems, 3), 0.01)
    model = adapter(
        polar_artifact,
        padding=FixedPadding(nodes_budget=8, edges_budget=128, graphs_budget=4),
    )
    result = model(state)
    eager = adapter(polar_artifact)(state)
    assert result["charges"].shape == (3 * systems,)
    assert result["dipole"].shape == (systems, 3)
    assert result["density_coefficients"].shape == (3 * systems, 4)
    for name in eager:
        close(result[name], eager[name].numpy())
    charge = state.total_charge if extras else torch.zeros(systems)
    close(result["charges"].reshape(systems, 3).sum(1), charge.numpy())
    if extras:
        assert not torch.equal(
            result["energy"],
            model(state_of([water(False) for _ in range(systems)]))["energy"],
        )


@fp64_only
def test_polar_zero_field_matches_missing_and_alias_is_explicit(polar_artifact):
    state = state_of(water(False))
    model = adapter(polar_artifact, input_names={"external_field": "external_E_field"})
    expected = model(state)
    state.system_extras["external_E_field"] = torch.zeros(1, 3)
    actual = model(state)
    for name in expected:
        close(actual[name], expected[name].numpy())


@fp64_only
@pytest.mark.parametrize(
    "polar_artifact", ["molecular", "full_periodic"], indirect=True
)
def test_polar_one_compile_for_changing_real_atoms(polar_artifact):
    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()
    model = adapter(
        polar_artifact,
        compile_mode="default",
        compile_backend=counter,
        padding=FixedPadding(nodes_budget=4, edges_budget=64, graphs_budget=2),
    )
    periodic = (
        model.deployed.config.electrostatics.periodicity_profile == "full_periodic"
    )
    for n in (3, 2, 3):
        state = state_of(water(periodic)[:n])
        actual = model(state)
        expected = adapter(polar_artifact)(state)
        for name in expected:
            close(actual[name], expected[name].numpy())
    assert counter.frame_count == 1


@fp64_only
def test_real_torchsim_trajectory_from_bundle_matches_ase(artifact, monkeypatch):
    from torch_sim.integrators.nve import nve_init, nve_step

    def refuse_script(*args, **kwargs):
        raise AssertionError("TorchSim deployment must not use serialized code")

    for name in ("script", "trace", "load", "save"):
        monkeypatch.setattr(torch.jit, name, refuse_script)
    bundle = export_lammps(artifact, halo_exchange=False)
    model = adapter(bundle)
    state = nve_init(state_of(water(True)), model, kT=0.01)
    for _ in range(5):
        result = model(state)
        compare_ase(artifact, state, result)
        state = nve_step(state, model, dt=0.001)


@fp64_only
def test_cold_import_and_trajectory_never_use_torchscript(artifact):
    code = """
import sys
import torch
def forbidden(*args, **kwargs):
    raise AssertionError("TorchScript is forbidden in V1 TorchSim deployment")
for name in ("script", "trace", "load", "save"):
    setattr(torch.jit, name, forbidden)
from ase import Atoms
from torch_sim.io import atoms_to_state
from torch_sim.integrators.nve import nve_init, nve_step
from mace_torch.deploy.torchsim import MaceTorchSimModel
model = MaceTorchSimModel(sys.argv[1])
atoms = Atoms(
    "OH2", positions=[[0, 0, 0], [.96, 0, 0], [-.24, .93, 0]], cell=[4]*3, pbc=True
)
state = atoms_to_state(atoms, device=torch.device("cpu"), dtype=torch.float64)
state = nve_init(state, model, kT=.01)
nve_step(state, model, dt=.001)
assert "mace" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(artifact)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@fp64_only
def test_declared_backend_runs_are_built_after_padding(artifact, tmp_path, monkeypatch):
    from mace_torch_neighbor_fixtures import register
    from mace_torch_span_fixtures import ReferenceSpan, SpanBackend

    register(
        tmp_path,
        monkeypatch,
        "test_span",
        "mace_torch_span_fixtures:SpanBackend",
        group="mace.kernel_backends.torch",
    )
    monkeypatch.setattr(
        SpanBackend, "edge_features", frozenset({"sender_runs", "receiver_runs"})
    )
    called = []
    original = ReferenceSpan.forward

    def checked(
        self, nodes, edges, sender, receiver, num_nodes, member_weights, edge_runs
    ):
        called.append(num_nodes)
        assert num_nodes == 8 and len(sender) == 64
        for side, index in (("sender_runs", sender), ("receiver_runs", receiver)):
            runs = getattr(edges, side)
            assert len(runs.offsets) == 9
            assert runs.offsets[-1] == 64
            assert torch.equal(runs.order, torch.argsort(index, stable=True))
        return original(
            self, nodes, edges, sender, receiver, num_nodes, member_weights, edge_runs
        )

    monkeypatch.setattr(ReferenceSpan, "forward", checked)
    model = adapter(
        artifact,
        backend="test_span",
        padding=FixedPadding(nodes_budget=8, edges_budget=64, graphs_budget=2),
    )
    state = state_of(water(False))
    compare_ase(artifact, state, model(state))
    assert called


@fp64_only
def test_force_flags_and_invalid_state_inputs(artifact):
    model = adapter(artifact, compute_forces=False, compute_stress=True)
    state = state_of(water(True))
    assert "forces" not in model(state)
    with pytest.raises(ValueError, match="undeclared"):
        adapter(artifact, input_names={"pos": "not_read"})
    state.system_extras["magmom"] = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="leading rows"):
        model(state)


@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.parametrize("family", ["energy", "polar"])
def test_cuda_compiled_steps_with_cueq(artifact, polar_artifact, dtype, family):
    from mace_core.config.precision import PrecisionConfig
    from mace_torch_neighbor_fixtures import require_cuda

    require_cuda()
    path = artifact if family == "energy" else polar_artifact
    precision = PrecisionConfig(
        model="float32" if dtype == torch.float32 else "float64"
    )
    settings = dict(
        device="cuda",
        backend="cueq",
        precision=precision,
        padding=FixedPadding(nodes_budget=4, edges_budget=64, graphs_budget=2),
    )
    eager = adapter(path, **settings)
    compiled = adapter(path, compile_mode="reduce-overhead", **settings)
    for n in (3, 2, 3):
        state = state_of(water(False)[:n], dtype).to(device=eager.device)
        expected = eager(state)
        actual = compiled(state)
        torch.cuda.synchronize()
        for name in expected:
            row = FP32 if dtype == torch.float32 else FP64_CPU_REFERENCE
            torch.testing.assert_close(
                actual[name], expected[name], atol=row.atol, rtol=row.rtol
            )


@fp64_only
def test_declared_atomic_moments_and_magnetic_diagnostics(tmp_path):
    from test_mace_torch_magnetic_model import configuration, iron_cluster

    _, model = built(configuration(tmp_path))
    path = write_model(tmp_path / "magnetic", model.engine, model.metadata)
    runtime = adapter(
        path,
        compute_stress=False,
        padding=FixedPadding(nodes_budget=8, edges_budget=64, graphs_budget=2),
    )
    atoms = iron_cluster()
    state = state_of(atoms)
    state.atom_extras["magmom"] = torch.tensor(atoms.arrays["REF_magmom"])
    for scale in (1.0, 0.5):
        state.atom_extras["magmom"] *= scale
        atoms.arrays["REF_magmom"] = state.magmom.numpy()
        atoms.calc = MACECalculator(path)
        result = runtime(state)
        close(result["energy"][0], atoms.get_potential_energy())
        close(result["forces"], atoms.get_forces())
        assert result["node_interaction_energy"].shape[0] == len(atoms)


@fp64_only
def test_joint_dipole_preserves_supplied_fixed_charges(tmp_path):
    from ase import Atoms, units
    from mace_core.metadata import (
        ConfigRecord,
        E0Details,
        HeadSummary,
        ModelMetadata,
        Provenance,
    )
    from test_mace_torch_joint_dipole import build

    engine, config = build()
    metadata = ModelMetadata(
        elements=[1],
        config=ConfigRecord.from_config(config),
        provenance=Provenance(versions={"mace-torch-v1": "test"}),
        heads={
            "default": HeadSummary(e0=E0Details(source="explicit", values={"H": 0.0}))
        },
    )
    path = write_model(tmp_path / "joint", engine, metadata)
    atoms = Atoms("H2", positions=[[0.2, 0.3, -0.1], [0.9, -0.1, 0.2]])
    state = state_of(atoms)
    runtime = adapter(path, compute_forces=False, compute_stress=False)
    without_charges = runtime(state)["dipole"]
    state.atom_extras["charges"] = torch.tensor([-0.3, 0.7], dtype=torch.float64)
    graph, _ = runtime._graph(state)
    close(graph["charges"], state.charges)
    actual = runtime(state)["dipole"]
    # The catalogue reports dipoles in Debye; positions and charges use Å and e.
    expected = (state.positions * state.charges[:, None]).sum(0)[None] / units.Debye
    close(actual - without_charges, expected)
