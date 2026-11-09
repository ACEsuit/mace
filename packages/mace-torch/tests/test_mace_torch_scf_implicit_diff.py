"""QEq: implicit constrained derivatives, physical invariances and solver contracts."""

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
import torch
from conftest import TOLERANCES, fp64_only
from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.config.model import ModelConfig
from mace_core.electrostatics import UnsupportedSolveError
from mace_core.electrostatics import registry as solver_registry
from mace_core.kernels.precision import Precision
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.electrostatics import (
    LONG_RANGE_PLAN,
    ReferenceSolver,
    long_range_plan,
)
from mace_torch.kernels import initialize_model_weights
from mace_torch.models.electrostatics import QEqModel
from mace_torch.physics import DerivativeEngine
from mace_torch.physics.charge_equilibration import (
    ChargeEquilibration,
    constrained_charges,
)
from scipy.spatial.transform import Rotation
from test_mace_torch_electrostatics_families import ENERGY_STRESS, graph_for, model_for
from test_mace_torch_polar_training import _EntryPoint


def qeq_model(
    profile="molecular", *, solver="reference", trains_derivatives=False, **options
):
    local = model_for("lc_mace", degree=0, profile=profile)
    document = local.config.model_dump(mode="json")
    document.update(model="qeq", local_sources=None, qeq=options)
    config = ModelConfig.model_validate(document)
    precision = cast(Precision, str(torch.get_default_dtype()).removeprefix("torch."))
    model = QEqModel(
        ReferenceBackend(),
        config,
        [1, 8],
        [ENERGY_STRESS],
        local.outputs.energy_readout,
        settings=QEqModel.make_solver_spec(
            config,
            ElectrostaticsConfig(
                enabled=True,
                periodicity_profile=profile,
                slab_normal=2 if profile == "z_slab" else None,
            ),
            "float64",
        ),
        precision=precision,
        solver=solver,
        trains_derivatives=trains_derivatives,
    )
    initialize_model_weights(model, 7)
    return model


def qeq_graph(profile="molecular", **kwargs):
    graph = graph_for(profile, **kwargs)
    graph["total_charge"] = torch.tensor([0.6])
    graph["input_chi"] = torch.tensor([-0.4, 0.1, 0.2, 0.3])
    graph["input_hardness"] = torch.tensor([1.8, 2.0, 2.2, 2.4])
    return graph


@fp64_only
def test_linear_solve_implicit_gradient_matches_unrolled_and_finite_differences():
    matrix = torch.tensor(
        [[4.0, 0.2, 0.1], [0.2, 3.0, 0.3], [0.1, 0.3, 2.0]]
    ).requires_grad_()
    chi = torch.tensor([0.2, -0.3, 0.4], requires_grad=True)
    total = torch.tensor(0.7, requires_grad=True)
    actual = constrained_charges(matrix, chi, total)
    # Projected gradient descent is an independent unrolled reference on the
    # same fixed-charge subspace; the production solve does not unroll it.
    unrolled = total.expand(3) / 3
    for _ in range(120):
        gradient = matrix @ unrolled + chi
        unrolled = unrolled - 0.2 * (gradient - gradient.mean())
    torch.testing.assert_close(actual, unrolled, atol=1e-12, rtol=1e-12)
    probe = torch.tensor([0.2, -0.5, 0.8])
    gradients = torch.autograd.grad(
        (actual * probe).sum(), (matrix, chi, total), create_graph=True
    )
    reference = torch.autograd.grad((unrolled * probe).sum(), (matrix, chi, total))
    for found, expected in zip(gradients, reference, strict=True):
        torch.testing.assert_close(found, expected, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(
        constrained_charges, (matrix, chi, total), fast_mode=True
    )
    assert torch.autograd.gradgradcheck(
        constrained_charges, (matrix, chi, total), fast_mode=True
    )
    with pytest.raises(torch.linalg.LinAlgError):
        constrained_charges(torch.zeros(2, 2), torch.zeros(2), total)


@fp64_only
@pytest.mark.parametrize("profile", ["molecular", "full_periodic", "z_slab"])
def test_qeq_force_stress_and_force_training_gradients(profile):
    model = qeq_model(profile, train_hardness=True)
    graph = qeq_graph(profile)
    result = DerivativeEngine(model, ENERGY_STRESS)(
        graph, compute=("forces", "stress"), training=True
    )
    step = 1e-5

    def energy(displacement=0.0, strain=0.0):
        changed = dict(graph)
        delta = torch.zeros_like(graph["positions"])
        delta[1, 0] = displacement
        deformation = torch.eye(3)
        deformation[0, 0] += strain
        changed["positions"] = (graph["positions"] + delta) @ deformation
        changed["cell"] = graph["cell"] @ deformation
        changed["shifts"] = graph["shifts"] @ deformation
        return model(changed).total_energy.sum()

    force = -(energy(displacement=step) - energy(displacement=-step)) / (2 * step)
    torch.testing.assert_close(force, result.forces[1, 0], atol=1e-6, rtol=0)
    if profile != "molecular":
        stress = (
            (energy(strain=step) - energy(strain=-step))
            / (2 * step)
            / torch.linalg.det(graph["cell"]).abs().sum()
        )
        torch.testing.assert_close(stress, result.stress[0, 0, 0], atol=1e-6, rtol=0)
    else:
        assert torch.count_nonzero(result.stress) == 0
    (result.forces.square().sum() + result.stress.square().sum()).backward()
    for head in model.source_maps.values():
        gradients = [p.grad for p in head.parameters() if p.requires_grad]
        assert gradients and all(
            g is not None and torch.isfinite(g).all() for g in gradients
        )
        assert any(g.count_nonzero() for g in gradients)


@fp64_only
def test_position_and_total_charge_have_double_backward():
    model = qeq_model(train_hardness=True)
    graph = qeq_graph()

    def energy(positions, total):
        return model(dict(graph, positions=positions, total_charge=total)).total_energy

    assert torch.autograd.gradgradcheck(
        energy,
        (
            graph["positions"].clone().requires_grad_(),
            graph["total_charge"].clone().requires_grad_(),
        ),
        fast_mode=True,
    )


@pytest.mark.parametrize("train_hardness", [False, True])
def test_conservation_rotation_translation_and_permutation(train_hardness):
    model = qeq_model(train_hardness=train_hardness)
    graph = qeq_graph()
    result = model(graph)
    atol, rtol = TOLERANCES[torch.get_default_dtype()]
    torch.testing.assert_close(
        result.extras["total_charge"],
        graph["total_charge"].to(result.total_energy),
        atol=atol,
        rtol=rtol,
    )
    shift = np.array([0.4, -0.2, 0.3])
    shifted = model(qeq_graph(shift=shift))
    torch.testing.assert_close(
        shifted.total_energy, result.total_energy, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        shifted.dipole,
        result.dipole
        + result.extras["total_charge"][:, None]
        * torch.tensor(shift, dtype=result.dipole.dtype),
        atol=atol,
        rtol=rtol,
    )
    rotation = Rotation.from_rotvec([0.3, -0.2, 0.4]).as_matrix()
    rotated = model(qeq_graph(rotation=rotation))
    torch.testing.assert_close(
        rotated.total_energy, result.total_energy, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        rotated.dipole,
        result.dipole @ torch.tensor(rotation, dtype=result.dipole.dtype).T,
        atol=atol,
        rtol=rtol,
    )
    permutation = np.array([3, 1, 0, 2])
    permuted = model(qeq_graph(permutation=permutation))
    torch.testing.assert_close(
        permuted.extras["charges"],
        result.extras["charges"][permutation],
        atol=atol,
        rtol=rtol,
    )


@fp64_only
def test_applied_field_changes_charge_and_tracks_origin_work():
    model = qeq_model()
    graph = qeq_graph()
    original = model(graph)
    field = torch.tensor([[0.1, -0.03, 0.02]])
    graph["external_field"] = field
    result = model(graph)
    assert not torch.allclose(result.extras["charges"], original.extras["charges"])
    shift = np.array([0.4, -0.2, 0.3])
    shifted = qeq_graph(shift=shift)
    shifted["external_field"] = field
    changed = model(shifted)
    torch.testing.assert_close(
        changed.extras["charges"], result.extras["charges"], atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        changed.total_energy - result.total_energy,
        graph["total_charge"] * (torch.tensor(shift) * field).sum(),
        atol=1e-12,
        rtol=1e-12,
    )


@fp64_only
@pytest.mark.parametrize("profile", ["molecular", "full_periodic", "z_slab"])
def test_planned_qeq_compiles_without_graph_breaks(profile):
    model = qeq_model(profile, train_hardness=True)
    graph = qeq_graph(profile)
    graph[LONG_RANGE_PLAN] = long_range_plan(graph, model.solver_spec, dense_pairs=True)
    expected = model(graph)
    actual = torch.compile(model, backend="aot_eager", fullgraph=True)(graph)
    torch.testing.assert_close(actual.total_energy, expected.total_energy)
    torch.testing.assert_close(actual.extras["charges"], expected.extras["charges"])


@fp64_only
def test_fused_span_is_selected_once_and_inference_solver_is_refused_for_training(
    monkeypatch,
):
    calls = []

    class Fused(ReferenceSolver):
        name = "fused-qeq"

        def make_scf_solve(self, solver_spec):
            assert solver_spec.scf_spec.method == "qeq"
            calls.append("build")
            projection = self.long_range_features(solver_spec)

            class Span(ChargeEquilibration):
                def forward(self, *args, **kwargs):
                    calls.append("solve")
                    return super().forward(*args, **kwargs)

            return Span(projection)

    class InferenceOnly(Fused):
        name = "inference-qeq"
        capabilities = replace(
            ReferenceSolver.capabilities, supports_double_backward=False
        )

    monkeypatch.setattr(
        solver_registry,
        "entry_points",
        lambda **kwargs: [
            _EntryPoint("reference", ReferenceSolver),
            _EntryPoint("fused-qeq", Fused),
            _EntryPoint("inference-qeq", InferenceOnly),
        ],
    )
    model = qeq_model(solver="fused-qeq", trains_derivatives=True)
    with pytest.raises(UnsupportedSolveError, match="differentiated twice"):
        qeq_model(solver="inference-qeq", trains_derivatives=True)
    inference = qeq_model(solver="inference-qeq")

    def unexpected(**kwargs):
        raise AssertionError("solver resolved during forward")

    monkeypatch.setattr(solver_registry, "entry_points", unexpected)
    graph = qeq_graph()
    torch.testing.assert_close(model(graph).total_energy, inference(graph).total_energy)
    assert calls == ["build", "build", "solve", "solve"]


@pytest.mark.parametrize(
    "options",
    [
        {"kind": "other"},
        {"charge_method": "other"},
        {"electronegativity_input": "positions"},
        {"hardness_input": "total_charge"},
        {"electronegativity_input": "same", "hardness_input": "same"},
        {"default_hardness": float("nan")},
    ],
)
def test_invalid_qeq_options_are_rejected(options):
    with pytest.raises(ValueError):
        ModelConfig(model="qeq", qeq=options)


@fp64_only
@pytest.mark.parametrize("solver", ["reference", "qeq-nonbit"])
def test_qeq_training_inputs_optimizer_and_checkpoint(tmp_path, monkeypatch, solver):
    from ase.io import read, write
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.electrostatics import SolverSubstitutionError
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.deploy.loader import load_deployed
    from mace_torch.train import run_data_stage, run_model_stage, run_train_stage
    from mace_torch.train.optimizers import build_optimizer
    from test_mace_torch_full_batch import configuration

    class Nonbit(ReferenceSolver):
        name = "qeq-nonbit"
        capabilities = replace(ReferenceSolver.capabilities, bit_parity=False)

    monkeypatch.setattr(
        solver_registry,
        "entry_points",
        lambda **kwargs: [
            _EntryPoint("reference", ReferenceSolver),
            _EntryPoint("qeq-nonbit", Nonbit),
        ],
    )
    document = configuration(tmp_path, max_num_epochs=1).model_dump(mode="json")
    document["model"].update(
        model="qeq",
        qeq={
            "train_hardness": True,
            "electronegativity_input": "input_chi",
            "hardness_input": "input_hardness",
        },
    )
    document["electrostatics"].update(
        enabled=True, periodicity_profile="molecular", solver=solver
    )
    path = document["data"]["heads"]["default"]["train_file"]
    structures = read(path, index=":")
    for item in structures:
        item.arrays["chi"] = np.linspace(-0.2, 0.3, len(item))
        item.arrays["hardness"] = np.linspace(1.8, 2.4, len(item))
        item.info["total_charge"] = 0.7
    write(path, structures, format="extxyz")
    document["data"]["embedding_specs"] = {
        "input_chi": {"per": "atom", "key": "chi"},
        "input_hardness": {"per": "atom", "key": "hardness"},
    }
    config = ResolvedConfig.model_validate(document)
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    parameters = {id(p) for p in built.engine.parameters() if p.requires_grad}
    optimized = [
        id(p)
        for group in build_optimizer(built.engine, config.training).param_groups
        for p in group["params"]
    ]
    assert set(optimized) == parameters and len(optimized) == len(parameters)
    model = cast(QEqModel, built.engine.get_submodule("model"))
    before = {
        name: [p.detach().clone() for p in head.parameters()]
        for name, head in model.source_maps.items()
    }
    trained = run_train_stage(config, built, checkpoint_path=tmp_path / "qeq")
    assert np.isfinite(trained.history[-1].train_loss)
    for name, head in model.source_maps.items():
        assert any(
            not torch.equal(a, b)
            for a, b in zip(before[name], head.parameters(), strict=True)
        )
    deployed = load_deployed(tmp_path / "qeq.json")
    record = deployed.metadata.electrostatics
    assert record is not None
    assert record.solver == solver and record.bit_parity is (solver == "reference")
    assert record.solver_spec["scf_spec"]["method"] == "qeq"
    with pytest.raises(SolverSubstitutionError):
        load_deployed(
            tmp_path / "qeq.json",
            solver="qeq-nonbit" if solver == "reference" else "reference",
        )
    graph = next(data.train_loader.batches(0, drop_last=False)).graph
    assert torch.all(graph["total_charge"] == 0.7)
    expected = deployed.compute(graph)
    restored = load_deployed(tmp_path / "qeq.json").compute(graph)
    for name in ("total_energy", "forces", "dipole"):
        torch.testing.assert_close(
            getattr(restored, name), getattr(expected, name), atol=0, rtol=0
        )


@fp64_only
@pytest.mark.parametrize("charge_method", ["energy", "nonperiodic"])
def test_isolated_atom_charge_and_finite_second_derivatives(charge_method):
    from mace_torch_engine_fixtures import build_graph

    graph = build_graph(np.zeros((1, 3)), [1], np.eye(3) * 9, (False, False, False))
    graph.update(total_charge=torch.tensor([0.7]), external_field=torch.zeros(1, 3))
    model = qeq_model(charge_method=charge_method, train_hardness=True)
    result = DerivativeEngine(model, ENERGY_STRESS)(
        graph, compute=("forces", "stress"), training=True
    )
    torch.testing.assert_close(result.extras["charges"], graph["total_charge"])
    assert (
        torch.isfinite(result.total_energy).all()
        and torch.isfinite(result.forces).all()
    )
    (result.total_energy.square().sum() + result.forces.square().sum()).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@fp64_only
def test_qeq_compiled_solve_reuses_graph_when_structure_sizes_change():
    from mace_torch.electrostatics import long_range_geometry

    model = qeq_model()
    traces = []

    def backend(module, inputs):
        traces.append(module)
        return module.forward

    compiled = torch.compile(model.charge_solve, backend=backend, fullgraph=True)
    for sizes in ((2, 2), (1, 3)):
        graph = qeq_graph()
        graph.update(
            batch=torch.tensor([0] * sizes[0] + [1] * sizes[1]),
            ptr=torch.tensor([0, sizes[0], 4]),
            cell=graph["cell"].repeat(2, 1, 1),
            pbc=graph["pbc"].repeat(2, 1),
            structure_sizes=sizes,
            structure_periodic=(False, False),
            structure_padding=(False, False),
            total_charge=torch.tensor([0.6, -0.2]),
        )
        graph[LONG_RANGE_PLAN] = long_range_plan(
            graph, model.solver_spec, dense_pairs=True
        )
        geometry = long_range_geometry(graph, model.solver_spec)
        args = (
            graph,
            graph["input_chi"],
            graph["input_hardness"],
            graph["total_charge"],
            geometry,
        )
        actual = compiled(*args)
        torch.testing.assert_close(actual, model.charge_solve(*args))
        torch.testing.assert_close(
            torch.stack((actual[: sizes[0]].sum(), actual[sizes[0] :].sum())),
            graph["total_charge"],
        )
    assert len(traces) == 1
