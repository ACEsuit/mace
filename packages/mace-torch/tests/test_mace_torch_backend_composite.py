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
from conftest import fp64_only
from mace_core.kernels.capabilities import (
    BackendCapabilities,
    UnsupportedDescriptorError,
)
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    LinearDescriptor,
)
from mace_torch.backends import CompositeBackend, ReferenceBackend, resolve_backend
from mace_torch.backends.conformance import (
    TOLERANCES,
    conformance_cases,
    run_backend_conformance,
)


class PartialCapabilities(BackendCapabilities):
    def supports(self, descriptor) -> bool:
        if isinstance(descriptor, LinearDescriptor) and descriptor.has_bias:
            return False
        return super().supports(descriptor)


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

    def make_linear(self, descriptor):
        if not self.capabilities().supports(descriptor):
            raise UnsupportedDescriptorError("partial declines a biased linear")
        return self.reference.make_linear(descriptor)

    def make_symmetric_contraction(self, descriptor):
        return self.reference.make_symmetric_contraction(descriptor)

    def make_interaction_layer(self, descriptors):
        return None


def test_the_reference_passes_its_own_conformance(dtype):
    precision = "float64" if str(dtype).endswith("float64") else "float32"
    results = run_backend_conformance(ReferenceBackend(), precision=precision)
    assert all(result.built for result in results)
    checked = {check for result in results for check in result.checks}
    assert {"weights", "values", "gradients", "equivariance"} <= checked
    if precision == "float64":
        assert "double backward" in checked


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
        def make_linear(self, descriptor):
            return self.reference.make_linear(descriptor)

    with pytest.raises(AssertionError, match="built it anyway"):
        run_backend_conformance(Dishonest())


@fp64_only
def test_a_backend_that_computes_something_else_fails_on_the_values():
    class Wrong(PartialBackend):
        def make_symmetric_contraction(self, descriptor):
            other = replace(descriptor, basis="full")
            return self.reference.make_symmetric_contraction(other)

    with pytest.raises(AssertionError, match="symmetric_contraction"):
        run_backend_conformance(Wrong())


def test_the_tolerances_are_pinned():
    """A change here is a tolerance change and goes through its own review."""
    assert TOLERANCES == {"float64": (1e-10, 1e-10), "float32": (5e-5, 1e-3)}


def test_the_composite_builds_each_op_where_it_is_supported(caplog):
    composite = CompositeBackend(PartialBackend(), ReferenceBackend())
    with caplog.at_level(logging.WARNING, logger="mace_torch.backends.composite"):
        for descriptor in conformance_cases():
            op = {
                "LinearDescriptor": "linear",
                "ChannelwiseTPConvDescriptor": "channelwise_tp_conv",
                "SymmetricContractionDescriptor": "symmetric_contraction",
                "FullyConnectedTPDescriptor": "fully_connected_tp",
                "SegmentReduceDescriptor": "segment_reduce",
            }[type(descriptor).__name__]
            getattr(composite, f"make_{op}")(descriptor)
    by = {(d.op, d.backend) for d in composite.decisions}
    assert ("linear", "partial") in by and ("linear", "reference") in by
    assert ("channelwise_tp_conv", "reference") in by
    assert ("symmetric_contraction", "partial") in by
    # A declined descriptor of an op the backend implements is a warning; an
    # op it never implements is the reference's without complaint.
    assert "declines this descriptor" in caplog.text
    assert "channelwise_tp_conv" not in caplog.text
    report = composite.report()
    assert "linear: 2 by partial, 1 by reference" in report
    assert "channelwise_tp_conv: 1 by reference" in report
    assert "no seams" in report
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
    with pytest.raises(UnsupportedDescriptorError, match="training on forces"):
        module.require_double_backward("partial", "training on forces")
    composite = module.resolve_backend("partial")
    assert not composite.capabilities().supports_double_backward
    # Inference is still a first-class use: building it is not refused.
    composite.make_linear(LinearDescriptor(irreps_in="2x0e", irreps_out="2x0e"))


def test_the_conv_descriptor_carries_the_channel_count():
    assert ChannelwiseTPConvDescriptor(num_features=8).num_features == 8


def test_no_model_code_permutes_the_layout():
    """Every op reads and writes one layout, so nothing between the ops
    regroups features. The reference regroups inside its own kernels, which is
    where the permutations are allowed to live."""
    root = Path(__file__).resolve().parents[1] / "src" / "mace_torch"
    permutations = {"channel_layout_index", "inverse_layout_index", "path_layout_index"}
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
                    offenders.append(f"{path.name} imports a layout permutation")
                if isinstance(node, ast.Attribute) and node.attr in (
                    "permute",
                    "movedim",
                ):
                    offenders.append(f"{path.name}:{node.lineno} calls {node.attr}")
    assert not offenders
