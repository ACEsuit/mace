"""Nonlinear SCF: implicit derivatives, covariance and executable contracts."""

import numpy as np
import pytest
import torch
from conftest import TOLERANCES, fp64_only
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.electrostatics import LONG_RANGE_PLAN, long_range_plan
from mace_torch.kernels import initialize_model_weights
from mace_torch.nn.scf_nonlinear import PotentialEmbedding, norm_tanh
from mace_torch.nn.scf_response import ResponseOps
from mace_torch.physics import DerivativeEngine
from mace_torch.physics.nonlinear_fixed_point import (
    NonlinearFixedPoint,
    ResponseInputs,
    residual,
)
from test_mace_torch_electrostatics_families import ENERGY_STRESS as ENERGY
from test_mace_torch_fixed_point_scf import (
    fixed_graph,
    fixed_model,
)
from test_mace_torch_fixed_point_scf import fixed_graph as graph_for
from test_mace_torch_fixed_point_scf import (
    test_training_optimizer_and_solver_preserving_checkpoint as training_contract,
)
from torch import nn


@pytest.fixture
def compiler_cache():
    # Every architecture shares the same Python forward code. Independent
    # captures must not inherit other tests' specialization-cache limit.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


class Response(nn.Module):
    def forward(self, nodes, attributes, field, *edges):
        return torch.tanh(field * nodes)


def small_inputs(local, weight, charge):
    return ResponseInputs(
        local,
        torch.tensor([[0.12, 0.03], [0.02, 0.1]]),
        torch.tensor([[0.2], [-0.1]]),
        torch.tensor([[0.3], [0.4]]),
        torch.ones(2, 1),
        charge,
        torch.tensor([0.1]),
        weight,
        torch.ones(2, 1),
        torch.empty(0),
        torch.empty(0),
        torch.empty(2, 0, dtype=torch.long),
    )


@fp64_only
@pytest.mark.parametrize("constant_charge", [False, True])
def test_nonlinear_implicit_first_and_second_derivatives(constant_charge):
    solver = NonlinearFixedPoint(constant_charge, 8, 1e-12)

    def solve(local, weights, charge):
        return torch.cat(
            [
                x.reshape(-1)
                for x in solver(Response(), {}, small_inputs(local, weights, charge))
            ]
        )

    args = (
        torch.tensor([[0.1], [-0.2]], requires_grad=True),
        torch.tensor([[0.7], [0.9]], requires_grad=True),
        torch.tensor([0.15], requires_grad=True),
    )
    assert torch.autograd.gradcheck(solve, args)
    assert torch.autograd.gradgradcheck(solve, args)
    inputs = small_inputs(*args)
    density, fermi = solver(Response(), {}, inputs)
    state = (
        torch.cat((density.flatten(), fermi)) if constant_charge else density.flatten()
    )
    torch.testing.assert_close(
        residual(state, Response(), {}, inputs, constant_charge),
        torch.zeros_like(state),
        atol=1e-12,
        rtol=0,
    )
    # A long, differentiable trajectory is an independent derivative oracle
    # for the fixed-Fermi case; the charge case is pinned by finite differences.
    if not constant_charge:
        unrolled = inputs.local
        for _ in range(80):
            field = (
                inputs.projection @ unrolled.flatten()
                + inputs.applied.flatten()
                + inputs.fermi_response @ inputs.fermi
            ).reshape_as(unrolled)
            unrolled = inputs.local + torch.tanh(field * inputs.nodes)
        for result in (density, unrolled):
            gradient = torch.autograd.grad(
                result.square().sum(), args[:2], create_graph=True, retain_graph=True
            )
            second = torch.autograd.grad(
                torch.stack([item.square().sum() for item in gradient]).sum(),
                args[:2],
                retain_graph=True,
            )
            if result is density:
                expected = (gradient, second)
            else:
                for before, after in zip(expected, (gradient, second), strict=True):
                    for a, b in zip(before, after, strict=True):
                        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


@fp64_only
def test_nonconverged_solution_is_not_returned():
    inputs = small_inputs(
        torch.tensor([[0.1], [-0.2]]), torch.tensor([[2.0], [3.0]]), torch.tensor([0.1])
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        NonlinearFixedPoint(False, 1, 1e-14)(Response(), {}, inputs)


@fp64_only
def test_solver_compiles_and_reads_changed_membership(compiler_cache):
    solver = NonlinearFixedPoint(True, 5, 1e-12)
    response = Response()

    def call(local, nodes, membership, charge):
        inputs = small_inputs(local, nodes, charge)._replace(
            membership=membership, fermi_response=membership * 0.3, fermi=torch.zeros(2)
        )
        return solver(response, {}, inputs)

    compiled = torch.compile(call, backend="eager", fullgraph=True)
    args = (torch.tensor([[0.1], [-0.2]]), torch.tensor([[0.7], [0.9]]))
    for membership in (torch.eye(2), torch.eye(2).flip(0)):
        charge = torch.tensor([0.2, -0.1])
        actual = compiled(*args, membership, charge)
        expected = call(*args, membership, charge)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(membership.T @ actual[0][:, 0], charge)


@fp64_only
@pytest.mark.parametrize("scale", [0.0, 1e-5, 0.4, 3.0])
def test_norm_tanh_equivariance_and_zero_second_derivatives(scale):
    from scipy.spatial.transform import Rotation

    torch.manual_seed(12)
    features = (torch.randn(2, 4) * scale).requires_grad_()
    rotation = torch.tensor(Rotation.from_rotvec([0.2, -0.4, 0.3]).as_matrix())

    def rotate(values):
        return torch.cat((values[:, :1], values[:, 1:] @ rotation.T), -1)

    result = norm_tanh(features, ((1, 1), (1, 3)))
    torch.testing.assert_close(
        norm_tanh(rotate(features), ((1, 1), (1, 3))),
        rotate(result),
        atol=1e-14,
        rtol=1e-14,
    )
    assert torch.autograd.gradcheck(
        lambda x: norm_tanh(x, ((1, 1), (1, 3))), (features,)
    )
    assert torch.autograd.gradgradcheck(
        lambda x: norm_tanh(x, ((1, 1), (1, 3))), (features,)
    )


@fp64_only
def test_scalar_norm_tanh_preserves_legacy_tanh():
    first = PotentialEmbedding(ResponseOps(), "2x0e", "2x0e+2x1o", 2, "tanh", "float64")
    second = PotentialEmbedding(
        ResponseOps(), "2x0e", "2x0e+2x1o", 2, "norm_tanh", "float64"
    )
    initialize_model_weights(first, 15)
    second.load_state_dict(first.state_dict())
    potential = torch.randn(3, 2)
    nodes = torch.randn(3, 8)
    attrs = torch.eye(2)[torch.tensor([0, 1, 1])]
    torch.testing.assert_close(
        first(potential, nodes, attrs), second(potential, nodes, attrs), atol=0, rtol=0
    )


CASES = [
    {"update_nonlinearity": "mlp"},
    {"update_nonlinearity": "skip_mlp"},
    {"potential_embedding": "tanh"},
    {"update": "scaled", "update_nonlinearity": "mlp"},
    {"update": "scaled_dot", "update_nonlinearity": "skip_mlp"},
    {
        "update": "many_body",
        "update_correlation": 2,
        "update_interaction": "augmented",
        "node_mixer": "product",
    },
    {"electron_energy": "mlp_field"},
    {"electron_energy": "quadratic_charges"},
    {"electron_energy": "many_body_charges"},
    {"electron_energy": "many_body_charges_field"},
    {
        "potential_embedding": "norm_tanh",
        "feature_max_l": 1,
        "multipole_max_l": 1,
        "local_source": "linear",
    },
]


def model_for(options, profile="molecular"):
    model = fixed_model(profile, constant_charge=False, **options)
    with torch.no_grad():
        for weight in model.updates[0].readout.parameters():
            weight.mul_(0.01)
    return model


@pytest.mark.parametrize("options", CASES)
def test_nonlinear_model_rotation_inversion_permutation(options):
    from scipy.spatial.transform import Rotation

    profile = "full_periodic" if options.get("multipole_max_l") else "molecular"
    model = model_for(options, profile)
    graph = fixed_graph(profile)
    result = model(graph)
    atol, rtol = TOLERANCES[torch.get_default_dtype()]
    for rotation in (Rotation.from_rotvec([0.2, -0.3, 0.4]).as_matrix(), -np.eye(3)):
        actual = model(fixed_graph(profile, rotation=rotation))
        torch.testing.assert_close(
            actual.total_energy, result.total_energy, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            actual.dipole,
            result.dipole @ torch.tensor(rotation, dtype=result.dipole.dtype).T,
            atol=atol,
            rtol=rtol,
        )
    permutation = np.array([2, 0, 3, 1])
    actual = model(fixed_graph(profile, permutation=permutation))
    torch.testing.assert_close(
        actual.extras["density_coefficients"],
        result.extras["density_coefficients"][permutation],
        atol=atol,
        rtol=rtol,
    )
    shifted = model(fixed_graph(profile, shift=np.array([0.2, -0.1, 0.3])))
    torch.testing.assert_close(
        shifted.total_energy, result.total_energy, atol=atol, rtol=rtol
    )


@fp64_only
@pytest.mark.parametrize("options", [CASES[0], CASES[5], CASES[-1]])
def test_model_position_and_fermi_gradgradcheck(options):
    model = model_for(options, "full_periodic")
    graph = graph_for("full_periodic")

    def energy(positions, fermi):
        return model(dict(graph, positions=positions, fermi_level=fermi)).total_energy

    assert torch.autograd.gradgradcheck(
        energy,
        (
            graph["positions"].clone().requires_grad_(),
            graph["fermi_level"].clone().requires_grad_(),
        ),
        fast_mode=True,
    )


@fp64_only
@pytest.mark.parametrize("options", [CASES[0], CASES[5], CASES[-1]])
def test_force_stress_match_energy_finite_differences(options):
    model = model_for(options, "full_periodic")
    graph = graph_for("full_periodic")
    output = DerivativeEngine(model, ENERGY)(
        dict(graph), compute=("forces", "stress"), training=True
    )
    step = 1e-5
    displaced = []
    strained = []
    for sign in (-1, 1):
        positions = graph["positions"].clone()
        positions[0, 1] += sign * step
        displaced.append(model(dict(graph, positions=positions)).total_energy)
        strain = torch.eye(3)
        strain[0, 0] += sign * step
        changed = dict(
            graph,
            positions=graph["positions"] @ strain,
            cell=graph["cell"].reshape(-1, 3, 3) @ strain,
            shifts=graph["shifts"] @ strain,
        )
        strained.append(model(changed).total_energy)
    torch.testing.assert_close(
        -(displaced[1] - displaced[0]) / (2 * step),
        output.forces[0, 1].reshape(1),
        atol=1e-7,
        rtol=1e-6,
    )
    volume = torch.linalg.det(graph["cell"].reshape(-1, 3, 3)).abs()
    torch.testing.assert_close(
        (strained[1] - strained[0]) / (2 * step) / volume,
        output.stress[:, 0, 0],
        atol=1e-8,
        rtol=1e-6,
    )


@fp64_only
@pytest.mark.parametrize("options", [CASES[0], CASES[5], CASES[-1]])
def test_model_fullgraph_capture(options, compiler_cache):
    model = model_for({**options, "max_iters": 4})
    graph = graph_for()
    graph[LONG_RANGE_PLAN] = long_range_plan(graph, model.solver_spec, dense_pairs=True)
    expected = model(graph)
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    actual = compiled(graph)
    torch.testing.assert_close(actual.total_energy, expected.total_energy)
    torch.testing.assert_close(
        actual.extras["density_coefficients"], expected.extras["density_coefficients"]
    )


@fp64_only
@pytest.mark.parametrize(
    "constant_charge,options",
    [
        (False, {"update_nonlinearity": "mlp", "electron_energy": "mlp_field"}),
        (True, {"update_nonlinearity": "mlp", "electron_energy": "mlp_field"}),
        (
            False,
            {
                "potential_embedding": "norm_tanh",
                "feature_max_l": 1,
                "multipole_max_l": 1,
            },
        ),
    ],
)
def test_training_and_checkpoint(tmp_path, monkeypatch, options, constant_charge):
    training_contract(
        tmp_path,
        monkeypatch,
        constant_charge,
        "fixedpoint-nonbit",
        options=options,
        learning_rate=1e-4,
    )


@fp64_only
@pytest.mark.parametrize(
    "kind",
    ["linear", "fully_connected_tp", "channelwise_tp_conv", "symmetric_contraction"],
)
def test_response_jacobian_matches_registered_kernel_backward(kind):
    from mace_core.kernels.specs import (
        ChannelwiseTPConvSpec,
        FullyConnectedTPSpec,
        LinearSpec,
        SymmetricContractionSpec,
    )

    descriptors = {
        "linear": LinearSpec(
            irreps_in="2x0e+2x1o", irreps_out="2x0e+2x1o", has_bias=True
        ),
        "fully_connected_tp": FullyConnectedTPSpec(
            irreps_in1="2x0e+2x1o", irreps_in2="2x0e", irreps_out="2x0e+2x1o"
        ),
        "channelwise_tp_conv": ChannelwiseTPConvSpec(
            irreps_node="0e+1o", irreps_edge="0e+1o", irreps_out="0e+1o", num_features=2
        ),
        "symmetric_contraction": SymmetricContractionSpec(
            irreps_in="0e+1o",
            irreps_out="0e+1o",
            num_features=2,
            num_elements=2,
            correlation=3,
        ),
    }
    solver_spec = descriptors[kind]
    native = getattr(ResponseOps(), "make_" + kind)(solver_spec)
    ordinary = getattr(ReferenceBackend(), "make_" + kind)(solver_spec)
    initialize_model_weights(ordinary, 15)
    if kind != "channelwise_tp_conv":
        native.load_canonical(ordinary.to_canonical())
    torch.manual_seed(2)
    features = torch.randn(2, 8, requires_grad=True)
    if kind == "linear":
        args = ()
    elif kind == "fully_connected_tp":
        args = (torch.eye(2),)
    elif kind == "symmetric_contraction":
        args = (torch.tensor([0, 1]),)
    else:
        assert isinstance(solver_spec, ChannelwiseTPConvSpec)
        args = (
            torch.randn(2, 4),
            torch.randn(2, len(solver_spec.paths), 2),
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
            2,
        )
    expected = ordinary(features, *args)
    actual = native(features, *args)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    jacobian = torch.func.jacrev(lambda values: native(values, *args))(features)
    rows = [
        torch.autograd.grad(value, features, retain_graph=True)[0]
        for value in expected.flatten()
    ]
    torch.testing.assert_close(
        jacobian, torch.stack(rows).reshape_as(jacobian), atol=1e-12, rtol=1e-12
    )
    assert jacobian.abs().max() > 1e-3
    assert torch.autograd.gradgradcheck(
        lambda values: native(values, *args), (features,), fast_mode=True
    )


@fp64_only
def test_nonlinear_solver_span_is_selected_only_at_build(monkeypatch):
    from mace_core.electrostatics import registry
    from mace_torch.electrostatics import ReferenceSolver
    from test_mace_torch_polar_training import _EntryPoint

    events = []

    class Solver(ReferenceSolver):
        name = "nonlinear-test-solver"

        def make_scf_solve(self, solver_spec):
            assert solver_spec.scf_spec.method == "nonlinear_fixed_fermi"
            events.append("build")

            class Span(NonlinearFixedPoint):
                def forward(self, response, parameters, inputs):
                    events.append("solve")
                    return super().forward(response, parameters, inputs)

            return Span(
                False, solver_spec.scf_spec.max_iters, solver_spec.scf_spec.tolerance
            )

    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda **kwargs: [
            _EntryPoint(Solver.name, Solver),
            _EntryPoint("reference", ReferenceSolver),
        ],
    )
    model = fixed_model(
        solver=Solver.name,
        constant_charge=False,
        update_nonlinearity="mlp",
        trains_derivatives=True,
    )

    def unexpected(**kwargs):
        raise AssertionError("solver resolved inside forward")

    monkeypatch.setattr(registry, "entry_points", unexpected)
    model(fixed_graph())
    model(fixed_graph())
    assert events == ["build", "solve", "solve"]


@pytest.mark.parametrize(
    "options",
    [
        {"potential_embedding": "tanh", "feature_max_l": 1},
        {"update_scales": [1, 2]},
        {"update": "scaled", "update_scales": [1, float("nan")]},
        {"update_correlation": 2},
        {"max_iters": 0},
        {"tolerance": float("inf")},
    ],
)
def test_nonlinear_options_reject_invalid_or_unused_values(options):
    from mace_core.config.fixed_point_scf import FixedPointSCFConfig

    with pytest.raises(ValueError):
        FixedPointSCFConfig(**options)


@fp64_only
def test_norm_tanh_feasible_fixed_charge_training_and_reload(tmp_path):
    from mace_torch.serialization import load_checkpoint, save_checkpoint

    options = {
        "potential_embedding": "norm_tanh",
        "feature_max_l": 1,
        "multipole_max_l": 1,
    }
    first = fixed_model("full_periodic", constant_charge=False, **options)
    graph = fixed_graph("full_periodic")
    # A bounded response cannot supply arbitrary charge. Obtain a feasible Q
    # from an independent fixed-Fermi solve, then hold Q fixed during training.
    expected = first(graph)
    graph["total_charge"] = expected.extras["total_charge"].detach()
    constrained = fixed_model("full_periodic", constant_charge=True, **options)
    constrained.load_state_dict(first.state_dict())
    actual = constrained(graph)
    torch.testing.assert_close(
        actual.total_energy, expected.total_energy, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        actual.extras["fermi_level"], graph["fermi_level"], atol=1e-9, rtol=1e-9
    )
    optimizer = torch.optim.Adam(constrained.parameters(), lr=1e-6)
    before = [p.detach().clone() for p in constrained.updates.parameters()]
    output = DerivativeEngine(constrained, ENERGY)(
        dict(graph), compute=("forces", "stress"), training=True
    )
    loss = (
        output.total_energy - expected.total_energy.detach() - 0.01
    ).square().sum() + output.forces.square().sum()
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all()
        for p in constrained.parameters()
        if p.grad is not None
    )
    optimizer.step()
    after = constrained(graph)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, constrained.updates.parameters(), strict=True)
    )
    torch.testing.assert_close(
        after.extras["total_charge"], graph["total_charge"], atol=1e-12, rtol=1e-12
    )
    path = tmp_path / "bounded.safetensors"
    save_checkpoint(
        path, constrained, {"model": constrained.config.model_dump(mode="json")}
    )
    loaded = fixed_model("full_periodic", constant_charge=True, **options)
    load_checkpoint(path, loaded)
    torch.testing.assert_close(
        loaded(graph).total_energy, after.total_energy, atol=0, rtol=0
    )


@fp64_only
def test_disabled_many_body_residual_has_no_unused_weights():
    options = {"update": "many_body", "update_use_skip": False, "central_mixer": "none"}
    model = model_for(options)
    update = model.updates[0]
    assert update.many_body.interaction.skip is None
    assert not any(".skip." in name for name, _ in update.named_parameters())
    model(fixed_graph()).total_energy.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in update.parameters()
        if p.numel()
    )
