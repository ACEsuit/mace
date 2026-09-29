"""The example backend, held to everything a backend is held to.

Run from its own distribution, against the installed MACE, as a third party's
backend would be: discovered through its entry point, checked by the
conformance suite, and used to build a model a run would build.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
import torch
from mace_backend_example import ExampleBackend
from mace_backend_example_models import engine, graph
from mace_core.kernels import available_backends, get_backend
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    LinearDescriptor,
    SegmentReduceDescriptor,
    SymmetricContractionDescriptor,
)
from mace_torch.backends.conformance import conformance_cases, run_backend_conformance
from mace_torch.serialization import canonical_state, load_canonical_state
from torch.fx.experimental.proxy_tensor import make_fx

SOURCE = Path(__file__).resolve().parents[1] / "src" / "mace_backend_example"


def test_it_is_discovered_through_its_entry_point():
    found = {backend.name: backend for backend in available_backends("torch")}
    assert found["example"].loaded, found["example"].reason
    assert isinstance(get_backend("example"), ExampleBackend)


def test_it_imports_only_the_contract():
    """Nothing of the implementation it plugs into: the model calls a
    backend, never the other way round."""
    imported = set()
    for path in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
    assert imported - {"__future__"} <= {
        "mace_core",
        "mace_backend_example",
        "torch",
        "numpy",
        "itertools",
        "collections",
        "importlib",
    }, imported


#: What every case it builds must have passed, beyond the case's own checks.
ALWAYS = {"layout", "values", "gradients", "compiles", "one compile for every size"}
WEIGHTED = {"weights", "weight gradients"}


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_it_passes_the_whole_suite(precision):
    results = run_backend_conformance("example", precision=precision, compile_ops=True)
    assert all(result.built for result in results), [
        result.descriptor for result in results if not result.built
    ]
    for result in results:
        wanted = set(ALWAYS)
        if precision == "float64":
            wanted.add("double backward")
        if result.op in {"linear", "symmetric_contraction", "fully_connected_tp"}:
            wanted |= WEIGHTED | ({"reorder"} if precision == "float64" else set())
        biased = isinstance(result.descriptor, LinearDescriptor) and (
            result.descriptor.has_bias
        )
        if result.op != "segment_reduce" and not biased:
            wanted.add("equivariance")
        assert wanted <= set(result.checks), (result.op, wanted - set(result.checks))


def _namespaces(function, *arguments) -> set[str]:
    """The operator namespaces a traced call dispatches to."""
    traced = make_fx(function)(*arguments)
    return {
        node.target.namespace
        for node in traced.graph.nodes
        if isinstance(node.target, torch._ops.OpOverload)
    }


def _case(kind):
    return next(case for case in conformance_cases() if isinstance(case, kind))


def test_the_hot_ops_run_on_its_own_registrations():
    """Its convolution, contraction and reduction dispatch to its own
    ``custom_op``s, and none of its ops to the reference's. Delegating to the
    reference's registrations would make the double-backward and compile
    checks above a test of the reference."""
    backend = ExampleBackend()
    conv = backend.make_channelwise_tp_conv(_case(ChannelwiseTPConvDescriptor))
    nodes, edges = 4, 6
    sender = torch.tensor([0, 1, 2, 3, 0, 2])
    receiver = torch.tensor([1, 0, 3, 2, 2, 0])
    conv_inputs = (
        torch.randn(nodes, 4 * 4, dtype=torch.float64),
        torch.randn(edges, 16, dtype=torch.float64),
        torch.randn(edges, conv.num_paths, 4, dtype=torch.float64),
    )

    contraction = backend.make_symmetric_contraction(
        _case(SymmetricContractionDescriptor)
    )
    reduce = backend.make_segment_reduce(_case(SegmentReduceDescriptor))

    seen = {
        "tp_conv": _namespaces(
            lambda x, y, w: conv(x, y, w, sender, receiver, nodes), *conv_inputs
        ),
        "contraction": _namespaces(
            lambda x: contraction(x, torch.tensor([0, 1, 1, 0])),
            torch.randn(4, 4 * 16, dtype=torch.float64),
        ),
        "segment_sum": _namespaces(
            lambda v: reduce(v, torch.tensor([0, 2, 1, 0]), 3),
            torch.randn(4, 6, dtype=torch.float64),
        ),
    }
    for op, namespaces in seen.items():
        assert "mace_backend_example" in namespaces, (op, namespaces)
        assert "mace" not in namespaces, (op, namespaces)


def test_a_model_on_it_is_the_reference_model_and_back():
    """A checkpoint written on the reference loads here and computes the same
    energy, forces and stress, and one written here loads into the reference."""
    reference = engine("reference")
    mine = engine("example", seed=11)
    for source, target in ((reference, mine), (mine, reference)):
        load_canonical_state(target, canonical_state(source))
        expected = source(graph(), compute=("forces", "stress"))
        actual = target(graph(), compute=("forces", "stress"))
        for name in ("total_energy", "forces", "stress"):
            torch.testing.assert_close(
                getattr(actual, name),
                getattr(expected, name),
                atol=1e-10,
                rtol=1e-10,
                msg=name,
            )


def test_its_ops_are_the_ones_the_model_is_built_with(caplog):
    caplog.set_level("INFO", logger="mace_torch.train.model_stage")
    engine("example")
    for op in (
        "linear",
        "channelwise_tp_conv",
        "symmetric_contraction",
        "fully_connected_tp",
    ):
        assert f"{op}: " in caplog.text
    assert "by example" in caplog.text


def test_a_force_loss_sends_the_reference_s_gradient_to_the_weights():
    """The silent failure a backward that is not itself differentiable causes:
    training on forces takes the derivative of a derivative. Compared along
    random canonical directions, since the two models hold their parameters
    differently."""
    reference = engine("reference")
    mine = engine("example")
    start = canonical_state(reference)
    load_canonical_state(mine, start)

    def gradients(model):
        out = model(graph(), compute=("forces", "stress"), training=True)
        loss = (
            out.total_energy.pow(2).sum()
            + out.forces.pow(2).sum()
            + out.stress.pow(2).sum()
        )
        parameters = [p for p in model.parameters() if p.requires_grad]
        return parameters, torch.autograd.grad(loss, parameters)

    reference_parameters, reference_gradients = gradients(reference)
    my_parameters, my_gradients = gradients(mine)
    generator = torch.Generator().manual_seed(5)
    for _ in range(3):
        direction = {
            path: {
                name: torch.randn(value.shape, generator=generator, dtype=value.dtype)
                if value.is_floating_point()
                else value
                for name, value in tensors.items()
            }
            for path, tensors in start.items()
        }
        load_canonical_state(reference, direction)
        load_canonical_state(mine, direction)
        expected = sum(
            (g * p.detach()).sum()
            for g, p in zip(reference_gradients, reference_parameters, strict=True)
        )
        actual = sum(
            (g * p.detach()).sum()
            for g, p in zip(my_gradients, my_parameters, strict=True)
        )
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


def _cuda_or_skip() -> None:
    """Skip without CUDA, unless the run guarantees it.

    The capability contract of the repository's own suites: a job that
    exports ``gpu`` in ``MACE_REQUIRE_CAPS`` fails here instead of skipping, so
    a broken CUDA setup cannot pass by absence.
    """
    if torch.cuda.is_available():
        return
    required = {
        cap.strip()
        for cap in os.environ.get("MACE_REQUIRE_CAPS", "").split(",")
        if cap.strip()
    }
    if "gpu" in required:
        pytest.fail("gpu is required by MACE_REQUIRE_CAPS and CUDA is not available")
    pytest.skip("needs CUDA")


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_its_ops_replay_from_a_cuda_graph(precision):
    _cuda_or_skip()
    results = run_backend_conformance(
        "example", precision=precision, device="cuda", cuda_graphs=True
    )
    assert all("cuda graph" in result.checks for result in results)
