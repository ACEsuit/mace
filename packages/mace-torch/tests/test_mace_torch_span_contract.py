"""Accepted spans change execution, never member weights or checkpoints."""

import itertools
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from conftest import assert_close, fp64_only
from mace_core.config import AgnesiTransformConfig, BesselBasisConfig, ModelConfig
from mace_core.config.precision import PrecisionConfig
from mace_core.edges import plan_edges
from mace_core.kernels.specs import RadialMLPSpec
from mace_torch.backends.composite import CompositeBackend
from mace_torch.backends.precision import PrecisionBackend, boundary_precision
from mace_torch.backends.radial import BesselBasis, GaussianBasis
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.backends.spans import SpanCapabilities, validate_span
from mace_torch.graph import apply_edge_plan
from mace_torch.kernels import initialize_model_weights
from mace_torch.nn.backbone import MACEBackbone
from mace_torch.nn.radial import AgnesiTransform, SoftTransform
from mace_torch.serialization import (
    canonical_metadata,
    canonical_state,
    check_layout,
    load_canonical_state,
    save_checkpoint,
)
from mace_torch_engine_fixtures import (
    ATOMIC_NUMBERS,
    build_engine,
    build_graph,
    molecule,
)
from mace_torch_span_fixtures import SpanBackend


@pytest.mark.parametrize("interaction", ["density", "density_residual", "nonlinear"])
@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
def test_conditioned_and_density_spans_preserve_live_weights(
    interaction, layout, dtype
):
    class NativeSpanBackend(SpanBackend):
        def capabilities(self):
            return replace(super().capabilities(), native_activation_layout=layout)

    source = NativeSpanBackend()
    backend = CompositeBackend(source, ReferenceBackend())
    config = ModelConfig(
        interaction_first=interaction,
        interaction=interaction,
        num_interactions=2,
        num_channels=2,
        max_ell=1,
        correlation=2,
        hidden_irreps="0e+1o",
        radial_mlp=(4,),
    )
    precision = "float64" if dtype == torch.float64 else "float32"
    offered = MACEBackbone(backend, config, ATOMIC_NUMBERS, precision=precision)
    plain = MACEBackbone(
        ReferenceBackend(), config, ATOMIC_NUMBERS, precision=precision
    )
    for model in (offered, plain):
        initialize_model_weights(model, 27)
    assert len(source.spans) == 2
    assert canonical_metadata(offered) == canonical_metadata(plain)
    calls = count_span_calls(source)
    graph = {
        name: value.to(dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for name, value in build_graph(*molecule()).items()
    }
    outputs = []
    gradients = []
    for model in (offered, plain):
        positions = graph["positions"].clone().requires_grad_(True)
        features = model(
            {**graph, "positions": positions, "differentiated_twice": True}
        )
        output = model.atomic_descriptors(features, invariants_only=False)
        force = torch.autograd.grad(output.sum(), positions, create_graph=True)[0]
        (output.square().sum() + force.square().sum()).backward()
        outputs.append(output.detach())
        gradients.append(
            {name: parameter.grad for name, parameter in model.named_parameters()}
        )
    assert all(call.call_count == 1 for call in calls)
    assert_close(outputs[0], outputs[1])
    assert gradients[0].keys() == gradients[1].keys()
    for name, actual in gradients[0].items():
        expected = gradients[1][name]
        assert (actual is None) == (expected is None), name
        if actual is not None:
            assert expected is not None
            assert_close(actual, expected.detach(), name)


def count_span_calls(backend):
    """Instrument only the eager routing tests, outside the stateless span."""
    hooks = [Mock(return_value=None) for _ in backend.spans]
    for span, hook in zip(backend.spans, hooks, strict=True):
        span.register_forward_hook(hook)
    return hooks


@pytest.mark.parametrize(
    "training, twice, double, weights",
    tuple(itertools.product((False, True), repeat=4)),
)
def test_declared_capabilities_choose_the_path(training, twice, double, weights):
    backend = SpanBackend()
    backend.span_capabilities = SpanCapabilities(double, weights)
    model = build_engine(seed=3, backend=backend)
    calls = count_span_calls(backend)
    model.train(training)
    graph = {**build_graph(*molecule()), "differentiated_twice": twice}
    actual = model.model_forward(graph)
    expected = build_engine(seed=3).model_forward(graph)
    assert expected.total_energy is not None
    assert_close(actual.total_energy, expected.total_energy.detach())
    assert all(
        hook.call_count == int((not twice or double) and (not training or weights))
        for hook in calls
    )


@fp64_only
def test_the_engine_requests_second_derivatives_for_force_training_and_hessians(dtype):
    backend = SpanBackend()
    backend.span_capabilities = SpanCapabilities(False, True)
    model = build_engine(backend=backend).eval()
    calls = count_span_calls(backend)
    graph = build_graph(*molecule())
    model(graph, compute=("forces",))
    assert all(hook.call_count == 1 for hook in calls)
    model(graph, compute=("forces",), training=True)
    model(graph, compute=("hessian",))
    assert all(hook.call_count == 1 for hook in calls)
    assert "differentiated_twice" not in graph


def test_repeated_span_forwards_reuse_the_compiled_graph(dtype):
    from torch._dynamo.testing import CompileCounter

    model = build_engine(seed=3, backend=SpanBackend()).eval()
    graph = build_graph(*molecule())
    counter = CompileCounter()
    compiled = torch.compile(model.model_forward, backend=counter, fullgraph=True)
    with torch.no_grad():
        expected = model.model_forward(graph).total_energy
        for _ in range(3):
            assert_close(compiled(graph).total_energy, expected)
    assert counter.frame_count == 1


@fp64_only
def test_checkpoint_tensors_and_metadata_are_identical_with_or_without_spans(
    tmp_path, dtype
):
    offered = build_engine(seed=19, backend=SpanBackend()).eval()
    plain = build_engine(seed=19).eval()
    graph = build_graph(*molecule())
    actual, expected = offered(graph), plain(graph)
    assert torch.equal(actual.total_energy, expected.total_energy)
    assert_close(actual.forces, expected.forces.detach())
    assert canonical_metadata(offered) == canonical_metadata(plain)
    for source, target in ((offered, plain), (plain, offered)):
        check_layout(target, canonical_metadata(source))
        load_canonical_state(target, canonical_state(source))
        assert torch.equal(source(graph).total_energy, target(graph).total_energy)
    first, second = tmp_path / "span.safetensors", tmp_path / "plain.safetensors"
    save_checkpoint(first, offered, {})
    save_checkpoint(second, plain, {})
    assert first.read_bytes() == second.read_bytes()
    assert (
        first.with_suffix(".json").read_bytes()
        == second.with_suffix(".json").read_bytes()
    )


@pytest.mark.parametrize("nonpersistent", [False, True])
def test_held_state_is_refused_even_when_absent_from_state_dict(nonpersistent):
    span = torch.nn.Module()
    span.register_buffer("hidden", torch.ones(1), persistent=not nonpersistent)
    with pytest.raises(ValueError, match=r"test_backend.*hidden"):
        validate_span(span, "test_backend")


def test_report_records_each_layer_and_layout():
    backend = CompositeBackend(SpanBackend(), ReferenceBackend())
    build_engine(backend=backend)
    decisions = [
        decision for decision in backend.decisions if decision.layer is not None
    ]
    assert [d.layer for d in decisions] == [0, 1]
    assert all(d.backend == "test_span" and not d.reason for d in decisions)
    assert all(
        member.activation_layout == "mul_ir"
        for d in decisions
        for member in d.kernel_spec
    )
    assert "layer 0 span: test_span" in backend.report()
    assert "layer 1 span: test_span" in backend.report()


@pytest.mark.parametrize(
    "features",
    [
        frozenset({"receiver_runs"}),
        frozenset({"sender_runs"}),
        frozenset({"receiver_runs", "sender_runs"}),
    ],
)
def test_declared_runs_reach_the_span_without_host_work(features, monkeypatch):
    backend = SpanBackend()
    backend.edge_features = features
    model = build_engine(backend=backend)
    graph = apply_edge_plan(build_graph(*molecule()), plan_edges(features, frozenset()))
    for span in backend.spans:
        forward = span.forward

        def checked(
            nodes, edges, sender, receiver, count, weights, runs, original=forward
        ):
            assert runs is not None
            assert runs.order.device == sender.device
            assert (edges.receiver_runs is not None) == ("receiver_runs" in features)
            assert (edges.sender_runs is not None) == ("sender_runs" in features)
            return original(nodes, edges, sender, receiver, count, weights, runs)

        monkeypatch.setattr(span, "forward", checked)
    model(graph)


def test_engine_builds_runs_before_compilation_and_restores_edge_force_order():
    backend = SpanBackend()
    backend.edge_features = frozenset(
        {"sorted_by_sender", "receiver_runs", "sender_runs"}
    )
    model, reference = build_engine(seed=3, backend=backend), build_engine(seed=3)
    graph = build_graph(*molecule())
    order = torch.arange(graph["edge_index"].shape[1] - 1, -1, -1)
    graph["edge_index"] = graph["edge_index"][:, order]
    for key in ("shifts", "unit_shifts"):
        graph[key] = graph[key][order]
    expected = reference(graph, compute=("forces", "edge_forces"))
    model.compile_model(backend="aot_eager")
    actual = model(graph, compute=("forces", "edge_forces"))
    assert_close(actual.total_energy, expected.total_energy.detach())
    assert_close(actual.forces, expected.forces.detach())
    assert_close(actual.extras["edge_forces"], expected.extras["edge_forces"].detach())
    assert "receiver_counts" not in graph and "sender_counts" not in graph


@pytest.mark.parametrize(
    "pair, key",
    [
        (
            (lambda: BesselBasis(3.0, 3), lambda: GaussianBasis(3.0, 3)),
            "model.radial_basis.kind",
        ),
        ((AgnesiTransform, SoftTransform), "model.distance_transform.kind"),
        (
            (
                lambda: ReferenceBackend().make_radial_mlp(
                    RadialMLPSpec(channels=(3, 4, 2))
                ),
                lambda: ReferenceBackend().make_radial_mlp(
                    RadialMLPSpec(channels=(3, 4, 2), kind="layer_norm_silu")
                ),
            ),
            "model.radial_mlp",
        ),
    ],
)
def test_interchangeable_members_refuse_each_others_weights(pair, key):
    first, second = (factory() for factory in pair)
    for source, target in ((first, second), (second, first)):
        with pytest.raises(ValueError, match=key):
            target.load_canonical(source.to_canonical())
        with pytest.raises(RuntimeError, match=key):
            check_layout(target, canonical_metadata(source))


@fp64_only
def test_live_member_weights_receive_force_loss_gradients(dtype):
    backend = SpanBackend()
    config = ModelConfig(
        num_interactions=2,
        num_channels=2,
        max_ell=1,
        hidden_irreps="0e+1o",
        correlation=2,
        radial_basis=BesselBasisConfig(num_basis=3, trainable=True),
        distance_transform=AgnesiTransformConfig(trainable=True),
    )
    offered = MACEBackbone(backend, config, ATOMIC_NUMBERS)
    plain = MACEBackbone(ReferenceBackend(), config, ATOMIC_NUMBERS)
    for model in (offered, plain):
        initialize_model_weights(model, 37)
    graph = build_graph(*molecule())

    # These geometry operators must remain lazy while the span replaces them.
    def forbidden(*args, **kwargs):
        pytest.fail("span materialized the replaced edge inputs")

    offered.radial_embedding.register_forward_hook(forbidden)
    offered.spherical_harmonics.register_forward_hook(forbidden)
    results = []
    for model in (offered, plain):
        points = graph["positions"].clone().requires_grad_(True)
        features = model(
            {**graph, "positions": points, "differentiated_twice": True}
        ).layers[-1]
        forces = -torch.autograd.grad(features.sum(), points, create_graph=True)[0]
        gradients = torch.autograd.grad(
            forces.square().sum(),
            tuple(model.parameters()),
            allow_unused=True,
            materialize_grads=True,
        )
        results.append((features, forces, gradients))
    for actual, expected in zip(results[0][:2], results[1][:2], strict=True):
        assert_close(actual, expected.detach())
    for actual, expected in zip(results[0][2], results[1][2], strict=True):
        assert_close(actual, expected.detach())
    assert any(value.numel() and value.abs().max() > 0 for value in results[0][2])


@pytest.mark.parametrize("dtype", [torch.float32], indirect=True)
def test_one_widest_precision_policy_casts_the_whole_span(dtype):
    config = ModelConfig(num_interactions=1, num_channels=2, max_ell=1, correlation=2)
    precision = PrecisionConfig.mixed()
    native = SpanBackend()
    backend = CompositeBackend(
        PrecisionBackend(native, precision),
        PrecisionBackend(ReferenceBackend(), precision),
    )
    offered = MACEBackbone(
        backend, config, ATOMIC_NUMBERS, precision="float32", radial_precision="float64"
    )
    plain = MACEBackbone(
        PrecisionBackend(ReferenceBackend(), precision),
        config,
        ATOMIC_NUMBERS,
        precision="float32",
        radial_precision="float64",
    )
    for model in (offered, plain):
        initialize_model_weights(model, 51)
    graph = {
        name: value.to(dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for name, value in build_graph(*molecule()).items()
    }
    assert native.spans and all(
        boundary_precision(span) == ("float64", "float32") for span in native.spans
    )
    actual, expected = offered(graph).layers[-1], plain(graph).layers[-1]
    assert actual.dtype == dtype
    assert_close(actual, expected.detach())


def test_a_declined_span_reports_why_and_keeps_the_reference_path(monkeypatch):
    primary = SpanBackend()
    monkeypatch.setattr(primary, "make_interaction_layer", lambda _: None)
    backend = CompositeBackend(primary, ReferenceBackend())
    build_engine(backend=backend)
    assert (
        "layer 0 span: reference; op by op (backend declined this span shape"
        in backend.report()
    )
    assert (
        "layer 1 span: reference; op by op (backend declined this span shape"
        in backend.report()
    )


def test_a_layer_without_a_described_span_still_has_a_report_entry():
    backend = CompositeBackend(SpanBackend(), ReferenceBackend())
    MACEBackbone(
        backend, ModelConfig(num_channels=2, apply_cutoff=False), ATOMIC_NUMBERS
    )
    assert (
        "layer 0 span: reference; op by op (a member has no complete kernel_spec)"
        in backend.report()
    )
    assert (
        "layer 1 span: reference; op by op (a member has no complete kernel_spec)"
        in backend.report()
    )


def test_a_model_with_spans_compiles_as_one_graph():
    model = build_engine(backend=SpanBackend()).eval()
    graph = build_graph(
        *molecule(),
        cell=(torch.eye(3, dtype=torch.float64) * 7).numpy(),
        pbc=(True, True, True),
    )
    expected = model(graph, compute=("forces", "stress"))
    assert expected.stress.abs().max() > 0
    # Compilation specializations from other tests share this Python code object.
    torch.compiler.reset()
    try:
        model.compile_model(backend="aot_eager")
        actual = model(graph, compute=("forces", "stress"))
    finally:
        torch.compiler.reset()
    assert_close(actual.total_energy, expected.total_energy.detach())
    assert_close(actual.forces, expected.forces.detach())
    assert_close(actual.stress, expected.stress.detach())
