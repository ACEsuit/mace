"""The public conformance suite runs a registered, test-only reference span."""

import pytest
import torch
from mace_torch.backends.span_conformance import run_span_conformance
from mace_torch_neighbor_fixtures import register, require_cuda
from mace_torch_span_fixtures import ReferenceSpan, SpanBackend


@pytest.fixture
def backend(tmp_path, monkeypatch):
    register(
        tmp_path,
        monkeypatch,
        "test_span",
        "mace_torch_span_fixtures:SpanBackend",
        group="mace.kernel_backends.torch",
    )
    return "test_span"


def test_the_reference_span_passes_the_public_suite(backend, dtype):
    results = run_span_conformance(
        backend,
        precision="float64" if dtype == torch.float64 else "float32",
        training=True,
        compile_spans=True,
    )
    assert len(results) == 5
    for result in results:
        assert result.built
        assert {
            "stateless",
            "values",
            "first derivatives",
            "weight gradients",
            "equivariance",
            "edge order",
            "empty",
            "padded",
            "compiles",
            "decline",
        } <= set(result.checks)
        assert ("double backward" in result.checks) == (dtype == torch.float64)


@pytest.mark.gpu
def test_cuda_graph_capture_and_replay(backend, dtype):
    require_cuda()
    results = run_span_conformance(
        backend,
        device="cuda",
        precision="float64" if dtype == torch.float64 else "float32",
        cuda_graphs=True,
    )
    assert all("cuda graph" in result.checks for result in results)


@pytest.mark.parametrize("fault", ["values", "weights"])
def test_the_suite_catches_wrong_values_and_detached_member_weights(
    backend, dtype, monkeypatch, fault
):
    original = ReferenceSpan.forward

    def broken(
        self, nodes, edges, sender, receiver, num_nodes, member_weights, edge_runs
    ):
        if fault == "weights":
            member_weights = tuple(value.detach() for value in member_weights)
        result = original(
            self, nodes, edges, sender, receiver, num_nodes, member_weights, edge_runs
        )
        return result * 1.01 if fault == "values" else result

    monkeypatch.setattr(ReferenceSpan, "forward", broken)
    with pytest.raises(
        AssertionError, match="span values" if fault == "values" else "span gradient"
    ):
        run_span_conformance(
            backend, precision="float64" if dtype == torch.float64 else "float32"
        )


@pytest.mark.parametrize("runs", ["sender_runs", "receiver_runs"])
def test_conformance_supplies_the_declared_runs(backend, dtype, monkeypatch, runs):
    monkeypatch.setattr(SpanBackend, "edge_features", frozenset({runs}))
    original = ReferenceSpan.forward

    def checked(
        self, nodes, edges, sender, receiver, num_nodes, member_weights, edge_runs
    ):
        assert edge_runs is not None
        index = sender if runs == "sender_runs" else receiver
        assert torch.equal(edge_runs.order, torch.argsort(index, stable=True))
        assert edge_runs.offsets[-1] == len(sender)
        return original(
            self, nodes, edges, sender, receiver, num_nodes, member_weights, edge_runs
        )

    monkeypatch.setattr(ReferenceSpan, "forward", checked)
    results = run_span_conformance(
        backend, precision="float64" if dtype == torch.float64 else "float32"
    )
    assert all("edge order" in result.checks for result in results)
