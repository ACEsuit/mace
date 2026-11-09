"""Affine SCF roots, implicit derivatives, symmetry and training contracts."""

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
from mace_torch.electrostatics import LONG_RANGE_PLAN, ReferenceSolver, long_range_plan
from mace_torch.kernels import initialize_model_weights
from mace_torch.models.electrostatics import FixedPointSCFModel
from mace_torch.physics import DerivativeEngine
from mace_torch.physics.affine_fixed_point import AffineFixedPoint
from scipy.spatial.transform import Rotation
from test_mace_torch_electrostatics_families import ENERGY_STRESS, graph_for, model_for
from test_mace_torch_polar_training import _EntryPoint


def fixed_model(
    profile="molecular",
    *,
    solver="reference",
    trains_derivatives=False,
    readout_gate="silu",
    **options,
):
    local = model_for("lc_mace", degree=0, profile=profile)
    document = local.config.model_dump(mode="json")
    document.update(
        model="fixed_point_scf", local_sources=None, fixed_point_scf=options
    )
    if options.get("update") == "many_body" or options.get(
        "electron_energy", ""
    ).startswith("many_body"):
        document["max_ell"] = 2
    document["readout"]["gate"] = readout_gate
    config = ModelConfig.model_validate(document)
    model = FixedPointSCFModel(
        ReferenceBackend(),
        config,
        [1, 8],
        [ENERGY_STRESS],
        local.outputs.energy_readout,
        settings=FixedPointSCFModel.make_solver_spec(
            config,
            ElectrostaticsConfig(
                enabled=True,
                periodicity_profile=profile,
                slab_normal=2 if profile == "z_slab" else None,
            ),
            "float64",
        ),
        precision=cast(
            Precision, str(torch.get_default_dtype()).removeprefix("torch.")
        ),
        solver=solver,
        trains_derivatives=trains_derivatives,
    )
    initialize_model_weights(model, 7)
    return model


def fixed_graph(profile="molecular", **kwargs):
    graph = graph_for(profile, **kwargs)
    graph.update(total_charge=torch.tensor([0.06]), fermi_level=torch.tensor([0.2]))
    return graph


@fp64_only
@pytest.mark.parametrize("constant_charge", [True, False])
def test_implicit_derivatives_against_unrolled_and_finite_differences(constant_charge):
    response = torch.tensor([[0.05, 0.02], [0.01, -0.03]], requires_grad=True)
    offset = torch.tensor([[0.2], [-0.1]], requires_grad=True)
    fermi_response = torch.tensor([[0.1], [0.15]], requires_grad=True)
    total = torch.tensor([0.4], requires_grad=True)
    fermi = torch.tensor([0.2], requires_grad=True)
    membership = torch.ones(2, 1)
    solver = AffineFixedPoint(constant_charge)

    def solve(response, offset, fermi_response, total, fermi):
        independent = offset if constant_charge else offset + fermi_response * fermi
        return torch.cat(
            [
                part.reshape(-1)
                for part in solver(
                    response, independent, fermi_response, membership, total, fermi
                )
            ]
        )

    actual = solve(response, offset, fermi_response, total, fermi)
    density = offset.reshape(-1)
    mu = fermi
    for _ in range(80):
        independent = offset.reshape(-1) + response @ density
        if constant_charge:
            mu = (total - independent.sum()) / fermi_response.sum()
        density = independent + fermi_response.reshape(-1) * mu
    expected = torch.cat((density, mu))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    args = (response, offset, fermi_response, total, fermi)
    probe = torch.tensor([0.3, -0.2, 0.5])
    before = torch.autograd.grad((actual * probe).sum(), args, allow_unused=True)
    after = torch.autograd.grad((expected * probe).sum(), args, allow_unused=True)
    for first, second in zip(before, after, strict=True):
        if first is None:
            assert second is None
        else:
            torch.testing.assert_close(first, second, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(solve, args, fast_mode=True)
    assert torch.autograd.gradgradcheck(solve, args, fast_mode=True)


@fp64_only
@pytest.mark.parametrize("profile", ["molecular", "full_periodic", "z_slab"])
@pytest.mark.parametrize("constant_charge", [True, False])
def test_force_stress_and_force_training(profile, constant_charge):
    model = fixed_model(
        profile,
        constant_charge=constant_charge,
        multipole_max_l=1,
        feature_max_l=1,
        local_source="linear",
        electron_energy="quadratic_field",
    )
    graph = fixed_graph(profile)
    output = DerivativeEngine(model, ENERGY_STRESS)(
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
    torch.testing.assert_close(force, output.forces[1, 0], atol=1e-6, rtol=0)
    if profile != "molecular":
        stress = (
            (energy(strain=step) - energy(strain=-step))
            / (2 * step)
            / torch.linalg.det(graph["cell"]).abs().sum()
        )
        torch.testing.assert_close(stress, output.stress[0, 0, 0], atol=1e-6, rtol=0)
    else:
        assert output.stress.count_nonzero() == 0
    loss = output.forces.square().sum() + output.stress.square().sum()
    loss.backward()
    for module in (
        model.source_maps,
        model.layer_mixer,
        model.updates,
        model.electron_energy,
    ):
        grads = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.count_nonzero() for g in grads)
    if constant_charge:
        torch.testing.assert_close(output.extras["total_charge"], graph["total_charge"])
    else:
        torch.testing.assert_close(output.extras["fermi_level"], graph["fermi_level"])


@pytest.mark.parametrize("constant_charge", [True, False])
@pytest.mark.parametrize("profile,degree", [("molecular", 0), ("full_periodic", 1)])
def test_rotation_inversion_permutation_and_origin(constant_charge, profile, degree):
    model = fixed_model(
        profile,
        constant_charge=constant_charge,
        multipole_max_l=degree,
        feature_max_l=degree,
        local_source="linear",
    )
    graph = fixed_graph(profile)
    result = model(graph)
    atol, rtol = TOLERANCES[torch.get_default_dtype()]
    for rotation in (Rotation.from_rotvec([0.3, -0.2, 0.4]).as_matrix(), -np.eye(3)):
        rotated = model(fixed_graph(profile, rotation=rotation))
        torch.testing.assert_close(
            rotated.total_energy, result.total_energy, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            rotated.dipole,
            result.dipole @ torch.tensor(rotation, dtype=result.dipole.dtype).T,
            atol=atol,
            rtol=rtol,
        )
    permutation = np.array([2, 0, 3, 1])
    permuted = model(fixed_graph(profile, permutation=permutation))
    torch.testing.assert_close(
        permuted.extras["density_coefficients"],
        result.extras["density_coefficients"][permutation],
        atol=atol,
        rtol=rtol,
    )
    shift = np.array([0.2, -0.1, 0.3])
    moved = model(fixed_graph(profile, shift=shift))
    torch.testing.assert_close(
        moved.total_energy, result.total_energy, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        moved.dipole,
        result.dipole
        + result.extras["total_charge"][:, None]
        * torch.tensor(shift, dtype=result.dipole.dtype),
        atol=atol,
        rtol=rtol,
    )


@fp64_only
@pytest.mark.parametrize("profile", ["molecular", "full_periodic", "z_slab"])
def test_planned_forward_compiles_with_fullgraph(profile):
    model = fixed_model(profile)
    graph = fixed_graph(profile)
    graph[LONG_RANGE_PLAN] = long_range_plan(graph, model.solver_spec, dense_pairs=True)
    expected = model(graph)
    actual = torch.compile(model, backend="aot_eager", fullgraph=True)(graph)
    torch.testing.assert_close(actual.total_energy, expected.total_energy)
    torch.testing.assert_close(
        actual.extras["density_coefficients"], expected.extras["density_coefficients"]
    )


@fp64_only
def test_fused_solver_binds_once_and_checks_double_backward(monkeypatch):
    calls = []

    class Fused(ReferenceSolver):
        name = "fused-fixed-point"

        def make_scf_solve(self, solver_spec):
            assert solver_spec.scf_spec.method == "affine_fixed_charge"
            calls.append("build")

            class Span(AffineFixedPoint):
                def forward(self, *args, **kwargs):
                    calls.append("solve")
                    return super().forward(*args, **kwargs)

            return Span(True)

    class InferenceOnly(Fused):
        name = "inference-fixed-point"
        capabilities = replace(
            ReferenceSolver.capabilities, supports_double_backward=False
        )

    monkeypatch.setattr(
        solver_registry,
        "entry_points",
        lambda **kwargs: [
            _EntryPoint("reference", ReferenceSolver),
            _EntryPoint(Fused.name, Fused),
            _EntryPoint(InferenceOnly.name, InferenceOnly),
        ],
    )
    model = fixed_model(solver=Fused.name, trains_derivatives=True)
    with pytest.raises(UnsupportedSolveError, match="differentiated twice"):
        fixed_model(solver=InferenceOnly.name, trains_derivatives=True)

    def unexpected(**kwargs):
        raise AssertionError("solver resolved in forward")

    monkeypatch.setattr(solver_registry, "entry_points", unexpected)
    model(fixed_graph())
    model(fixed_graph())
    assert calls == ["build", "solve", "solve"]


@pytest.mark.parametrize(
    "options",
    [
        {"update_nonlinearity": "unknown"},
        {"potential_embedding": "unknown"},
        {"feature_widths": []},
        {"feature_norms": [1, 2]},
        {"density_scales": [1, 2]},
        {"multipole_width": float("nan")},
        {"kind": "other"},
        {"electron_energy": "many_body"},
    ],
)
def test_unsupported_or_invalid_architectures_are_explicit(options):
    with pytest.raises(ValueError):
        ModelConfig(model="fixed_point_scf", fixed_point_scf=options)


@fp64_only
@pytest.mark.parametrize("constant_charge", [True, False])
def test_singular_equations_fail_instead_of_returning_an_iterate(constant_charge):
    with pytest.raises(torch.linalg.LinAlgError):
        AffineFixedPoint(constant_charge)(
            torch.eye(2),
            torch.ones(2, 1),
            torch.zeros(2, 1),
            torch.ones(2, 1),
            torch.ones(1),
            torch.zeros(1),
        )


@fp64_only
@pytest.mark.parametrize("constant_charge", [True, False])
@pytest.mark.parametrize("solver", ["reference", "fixedpoint-nonbit"])
def test_training_optimizer_and_solver_preserving_checkpoint(
    tmp_path,
    monkeypatch,
    constant_charge,
    solver,
    options=None,
    learning_rate=0.01,
):
    from ase.io import read, write
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.electrostatics import SolverSubstitutionError
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.deploy.loader import load_deployed
    from mace_torch.train import run_data_stage, run_model_stage, run_train_stage
    from mace_torch.train.optimizers import build_optimizer
    from test_mace_torch_full_batch import configuration

    class Nonbit(ReferenceSolver):
        name = "fixedpoint-nonbit"
        capabilities = replace(ReferenceSolver.capabilities, bit_parity=False)

    monkeypatch.setattr(
        solver_registry,
        "entry_points",
        lambda **kwargs: [
            _EntryPoint("reference", ReferenceSolver),
            _EntryPoint(Nonbit.name, Nonbit),
        ],
    )
    document = configuration(tmp_path, max_num_epochs=1, lr=learning_rate).model_dump(
        mode="json"
    )
    document["model"].update(
        model="fixed_point_scf",
        fixed_point_scf={"constant_charge": constant_charge, **(options or {})},
    )
    document["electrostatics"].update(
        enabled=True, periodicity_profile="molecular", solver=solver
    )
    path = document["data"]["heads"]["default"]["train_file"]
    structures = read(path, index=":")
    for item in structures:
        item.info["total_charge"] = 0.06
        item.info[
            "chemical_potential" if solver == "fixedpoint-nonbit" else "fermi_level"
        ] = 0.2
    write(path, structures, format="extxyz")
    if not constant_charge and solver == "fixedpoint-nonbit":
        document["data"]["embedding_specs"] = {
            "fermi_level": {"per": "graph", "key": "chemical_potential"}
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
    model = cast(FixedPointSCFModel, built.engine.get_submodule("model"))
    before = [p.detach().clone() for p in model.updates.parameters()]
    trained = run_train_stage(config, built, checkpoint_path=tmp_path / "fixedpoint")
    assert np.isfinite(trained.history[-1].train_loss)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, model.updates.parameters(), strict=True)
    )
    deployed = load_deployed(tmp_path / "fixedpoint.json")
    record = deployed.metadata.electrostatics
    assert record is not None and record.solver == solver
    assert record.bit_parity is (solver == "reference")
    assert record.solver_spec["scf_spec"]["method"] == (
        ("affine_fixed_charge" if constant_charge else "affine_fixed_fermi")
        if model.settings.affine_response
        else ("nonlinear_fixed_charge" if constant_charge else "nonlinear_fixed_fermi")
    )
    with pytest.raises(SolverSubstitutionError):
        load_deployed(
            tmp_path / "fixedpoint.json",
            solver="fixedpoint-nonbit" if solver == "reference" else "reference",
        )
    graph = next(data.train_loader.batches(0, drop_last=False)).graph
    assert torch.all(
        graph["total_charge" if constant_charge else "fermi_level"]
        == (0.06 if constant_charge else 0.2)
    )
    first = deployed.compute(graph)
    second = load_deployed(tmp_path / "fixedpoint.json").compute(graph)
    for name in ("total_energy", "forces", "dipole"):
        torch.testing.assert_close(
            getattr(first, name), getattr(second, name), atol=0, rtol=0
        )


@fp64_only
@pytest.mark.parametrize("constant_charge", [True, False])
def test_position_and_constraint_double_backward(constant_charge):
    model = fixed_model(constant_charge=constant_charge)
    graph = fixed_graph()
    key = "total_charge" if constant_charge else "fermi_level"

    def energy(positions, constraint):
        return model(dict(graph, positions=positions, **{key: constraint})).total_energy

    assert torch.autograd.gradgradcheck(
        energy,
        (
            graph["positions"].clone().requires_grad_(),
            graph[key].clone().requires_grad_(),
        ),
        fast_mode=True,
    )


@fp64_only
def test_compiled_charge_constraints_follow_changing_atom_membership():
    solver = AffineFixedPoint(True)
    traces = []

    def backend(module, inputs):
        traces.append(module)
        return module.forward

    compiled = torch.compile(solver, backend=backend, fullgraph=True)
    response = torch.diag(torch.tensor([0.1, 0.2, 0.05, 0.15]))
    offset = torch.tensor([[0.1], [-0.2], [0.3], [0.05]])
    total = torch.tensor([0.4, -0.3])
    for ids in ([0, 0, 1, 1], [0, 1, 1, 1]):
        membership = torch.nn.functional.one_hot(torch.tensor(ids), 2).to(offset)
        fermi_response = membership * torch.tensor([0.1, 0.2, 0.3, 0.4])[:, None]
        args = (response, offset, fermi_response, membership, total, torch.zeros(2))
        density, fermi = compiled(*args)
        torch.testing.assert_close(
            membership.T @ density[:, 0], total, atol=1e-12, rtol=1e-12
        )
        expected = solver(*args)
        torch.testing.assert_close(density, expected[0])
        torch.testing.assert_close(fermi, expected[1])
    assert len(traces) == 1


@fp64_only
def test_disabled_readout_activation_keeps_local_source_linear():
    model = fixed_model(readout_gate="None")
    # The default source has three maps but no biases. With its gate disabled,
    # doubling its input must double the complete local density readout.
    source = model.source_maps[0]
    features = torch.randn(4, 8)
    torch.testing.assert_close(
        source(2 * features), 2 * source(features), atol=1e-12, rtol=1e-12
    )
