"""The per-op fallback, the conformance harness, and the training refusal.

The partial backend here stands in for an accelerated one: it builds some ops
and declines others, which is exactly what cueq and oeq do.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from conftest import assert_close, fp64_only
from mace_core.kernels.capabilities import (
    BackendCapabilities,
    UnsupportedKernelSpecError,
)
from mace_core.kernels.specs import (
    ChannelwiseTPConvSpec,
    LinearSpec,
)
from mace_torch.backends import CompositeBackend, ReferenceBackend, resolve_backend
from mace_torch.backends.conformance import (
    TOLERANCES,
    conformance_cases,
    run_backend_conformance,
)


class PartialCapabilities(BackendCapabilities):
    def supports(self, kernel_spec) -> bool:
        if isinstance(kernel_spec, LinearSpec) and kernel_spec.has_bias:
            return False
        return super().supports(kernel_spec)


class PartialBackend:
    """Builds linears without a bias and the contraction, and nothing else."""

    name = "partial"

    def __init__(self, double_backward: bool = True) -> None:
        self.reference = ReferenceBackend()
        self.double_backward = double_backward

    def capabilities(self) -> BackendCapabilities:
        return PartialCapabilities(
            ops=frozenset({"linear", "symmetric_contraction"}),
            devices=frozenset({"cpu"}),
            dtypes=frozenset({"float64", "float32"}),
            supports_double_backward=self.double_backward,
        )

    def make_linear(self, kernel_spec):
        if not self.capabilities().supports(kernel_spec):
            raise UnsupportedKernelSpecError("partial declines a biased linear")
        return self.reference.make_linear(kernel_spec)

    def make_symmetric_contraction(self, kernel_spec):
        return self.reference.make_symmetric_contraction(kernel_spec)

    def make_interaction_layer(self, descriptors):
        return None


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
def test_the_reference_passes_its_own_conformance(dtype, layout):
    precision = "float64" if str(dtype).endswith("float64") else "float32"
    results = run_backend_conformance(
        ReferenceBackend(), precision=precision, activation_layout=layout
    )
    assert all(result.built for result in results)
    checked = {check for result in results for check in result.checks}
    assert {"weights", "values", "gradients", "equivariance"} <= checked
    if precision == "float64":
        assert "double backward" in checked


@fp64_only
@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
def test_backend_reorder_is_block_diagonal(layout):
    """Every op with weights maps canonical weights onto its own through
    independent, invertible blocks within the bound, and back exactly. The
    harness derives the map from the op; here it runs over the reference, and
    each accelerated backend's own suite runs it over that backend."""
    results = run_backend_conformance(ReferenceBackend(), activation_layout=layout)
    weighted = {"linear", "symmetric_contraction", "fully_connected_tp"}
    assert all(
        "reorder" in result.checks
        for result in results
        if result.built and result.op in weighted
    )


@fp64_only
def test_a_partial_backend_passes_and_its_declines_are_recorded():
    results = run_backend_conformance(PartialBackend())
    built = {(result.op, result.built) for result in results}
    assert ("linear", True) in built
    assert ("linear", False) in built
    assert ("channelwise_tp_conv", False) in built
    assert ("symmetric_contraction", True) in built


@fp64_only
def test_a_backend_that_builds_what_it_declines_fails_the_harness():
    class Dishonest(PartialBackend):
        def make_linear(self, kernel_spec):
            return self.reference.make_linear(kernel_spec)

    with pytest.raises(AssertionError, match="built it anyway"):
        run_backend_conformance(Dishonest())


@fp64_only
def test_a_backend_that_computes_something_else_fails_on_the_values():
    class Wrong(PartialBackend):
        def make_symmetric_contraction(self, kernel_spec):
            other = replace(kernel_spec, basis="full")
            return self.reference.make_symmetric_contraction(other)

    with pytest.raises(AssertionError, match="symmetric_contraction"):
        run_backend_conformance(Wrong())


def test_the_tolerances_are_pinned():
    """A change here is a tolerance change and goes through its own review."""
    assert TOLERANCES == {"float64": (1e-10, 1e-10), "float32": (5e-5, 1e-3)}


def test_the_composite_builds_each_op_where_it_is_supported(caplog):
    composite = CompositeBackend(PartialBackend(), ReferenceBackend())
    with caplog.at_level(logging.WARNING, logger="mace_torch.backends.composite"):
        for kernel_spec in conformance_cases():
            op = {
                "LinearSpec": "linear",
                "ChannelwiseTPConvSpec": "channelwise_tp_conv",
                "SymmetricContractionSpec": "symmetric_contraction",
                "FullyConnectedTPSpec": "fully_connected_tp",
                "SegmentReduceSpec": "segment_reduce",
            }[type(kernel_spec).__name__]
            getattr(composite, f"make_{op}")(kernel_spec)
    by = {(d.op, d.backend) for d in composite.decisions}
    assert ("linear", "partial") in by and ("linear", "reference") in by
    assert ("channelwise_tp_conv", "reference") in by
    assert ("symmetric_contraction", "partial") in by
    # A declined kernel_spec of an op the backend implements is a warning; an
    # op it never implements is the reference's without complaint.
    assert "declines this kernel specification" in caplog.text
    assert "channelwise_tp_conv" not in caplog.text
    report = composite.report()
    assert "linear: 2 by partial, 1 by reference" in report
    assert "channelwise_tp_conv: 1 by reference" in report
    assert "Every op uses mul_ir activations" in report
    assert composite.served_by_primary()


def test_the_reference_resolves_to_itself_and_others_to_a_composite(monkeypatch):
    assert isinstance(resolve_backend("reference"), ReferenceBackend)
    from mace_torch.backends import composite as module

    monkeypatch.setattr(module, "get_backend", lambda name: PartialBackend())
    resolved = module.resolve_backend("partial")
    assert isinstance(resolved, CompositeBackend)
    assert resolved.name == "partial"


def test_a_backend_without_double_backward_is_refused_for_force_training(
    monkeypatch,
):
    from mace_torch.backends import composite as module

    monkeypatch.setattr(
        module, "get_backend", lambda name: PartialBackend(double_backward=False)
    )
    with pytest.raises(UnsupportedKernelSpecError, match="training on forces"):
        module.require_double_backward("partial", "training on forces")
    composite = module.resolve_backend("partial")
    assert not composite.capabilities().supports_double_backward
    # Inference is still a first-class use: building it is not refused.
    composite.make_linear(LinearSpec(irreps_in="2x0e", irreps_out="2x0e"))


def test_the_conv_descriptor_carries_the_channel_count():
    assert ChannelwiseTPConvSpec(num_features=8).num_features == 8


class NativePartialBackend(PartialBackend):
    def capabilities(self):
        return replace(
            super().capabilities(),
            activation_layouts=frozenset({"mul_ir", "ir_mul"}),
            native_activation_layout="ir_mul",
        )


@fp64_only
@pytest.mark.parametrize("family", ["energy", "polar", "magnetic"])
def test_native_chain_preserves_model_outputs_and_force_gradients(family):
    from mace_torch.serialization import canonical_state, load_canonical_state
    from mace_torch_engine_fixtures import build_engine, build_graph, molecule

    backend = CompositeBackend(NativePartialBackend(), ReferenceBackend())
    if family == "energy":
        make, graph = build_engine, build_graph(*molecule())
    elif family == "polar":
        from mace_torch.physics import DerivativeEngine
        from test_mace_torch_polar_model import WITH_STRESS, build_model, polar_graph

        def make(backend=None):
            return DerivativeEngine(build_model(backend=backend), WITH_STRESS)

        graph = polar_graph(*molecule())
    else:
        from test_mace_torch_magnetic_model import engine, moments
        from test_mace_torch_magnetic_model import graph as magnetic_graph

        make, graph = engine, magnetic_graph(molecule()[0], moments())
    native, canonical = make(backend=backend), make()
    load_canonical_state(native, canonical_state(canonical))
    actual = native(graph, compute=("forces",), training=True)
    expected = canonical(graph, compute=("forces",), training=True)
    assert_close(actual.total_energy, expected.total_energy.detach())
    assert_close(actual.forces, expected.forces.detach())
    if expected.dipole is not None:
        assert_close(actual.dipole, expected.dipole.detach())
    for output in (actual, expected):
        (output.total_energy.sum() + output.forces.square().sum()).backward()
    for (name, mine), (_, theirs) in zip(
        native.named_parameters(), canonical.named_parameters(), strict=True
    ):
        assert (mine.grad is None) == (theirs.grad is None), name
        if mine.grad is not None:
            assert theirs.grad is not None
            assert_close(mine.grad, theirs.grad.detach(), name)
    assert backend.decisions and {
        d.kernel_spec.activation_layout for d in backend.decisions if d.layer is None
    } == {"ir_mul"}
    if family == "energy":
        from mace_torch.nn.backbone import MACEBackbone

        assert isinstance(native.model.backbone, MACEBackbone)
        assert isinstance(canonical.model.backbone, MACEBackbone)
        mine, theirs = native.model.backbone(graph), canonical.model.backbone(graph)
        assert mine.layout == "ir_mul"
        assert_close(
            native.model.backbone.atomic_descriptors(mine, invariants_only=False),
            canonical.model.backbone.atomic_descriptors(
                theirs, invariants_only=False
            ).detach(),
        )


@pytest.mark.parametrize("heads", [1, 2])
def test_native_readouts_return_canonical_components_per_head(heads, dtype):
    from mace_core.observables import ObservableSpec
    from mace_torch.backends.layout import Layout
    from mace_torch.kernels import initialize_model_weights
    from mace_torch.models.readouts import ObservableReadout
    from mace_torch.serialization import canonical_state, load_canonical_state

    declaration = "3x0e+3x1o+3x2e"
    backend = CompositeBackend(NativePartialBackend(), ReferenceBackend())
    spec = ObservableSpec(name="test", irreps="2x1o+2x2e", per_atom=True, units="1")

    def make(backend):
        return ObservableReadout(
            backend,
            spec,
            ["0e+1o+2e"] * 2,
            3,
            nonlinear=True,
            readout_irreps="3x0e+2x1o+2x2e",
            num_heads=heads,
            precision="float64" if dtype == torch.float64 else "float32",
        )

    native, canonical = make(backend), make(ReferenceBackend())
    initialize_model_weights(canonical, 31)
    load_canonical_state(native, canonical_state(canonical))
    features = [torch.randn(4, 27, requires_grad=True) for _ in range(2)]
    selected = torch.arange(4) % heads
    layout = Layout("ir_mul")
    moved = [
        layout.from_canonical(value, layout.terms(declaration)) for value in features
    ]
    actual, expected = native(moved, selected), canonical(features, selected)
    assert_close(actual, expected.detach())
    mine = torch.autograd.grad(actual.square().sum(), features, retain_graph=True)
    theirs = torch.autograd.grad(expected.square().sum(), features)
    for value, wanted in zip(mine, theirs, strict=True):
        assert_close(value, wanted.detach())


def test_no_model_code_permutes_the_layout():
    """Every op reads and writes the chain's layout, so nothing between the ops
    regroups features, and model code never picks a layout: it asks the backend
    for one with `layout_of`. Building a `Layout` by name, or permuting axes,
    is what the backends do inside their ops."""
    root = Path(__file__).resolve().parents[1] / "src" / "mace_torch"
    permutations = {"Layout", "CANONICAL"}
    offenders = []
    for folder in ("nn", "models"):
        for path in (root / folder).rglob("*.py"):
            if path.name == "layout.py":
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and any(
                    alias.name in permutations for alias in node.names
                ):
                    offenders.append(f"{path.name} builds its own layout")
                if isinstance(node, ast.Attribute) and node.attr in (
                    "permute",
                    "movedim",
                ):
                    offenders.append(f"{path.name}:{node.lineno} calls {node.attr}")
    assert not offenders
