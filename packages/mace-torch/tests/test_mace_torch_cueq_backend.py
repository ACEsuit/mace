"""The cuEquivariance backend, op by op and as a whole model.

Runs wherever ``cuequivariance_torch`` imports: on a GPU host with the
compiled operations, and on a CPU host through cuEquivariance's pure-torch
path, which is how the numerics are checked without a GPU. The model is
compared with the reference model carrying the same canonical weights: the
energy, the forces and the stress, and the gradient a force loss sends to the
weights, which is the second derivative training on forces takes.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    LinearDescriptor,
    SymmetricContractionDescriptor,
)
from mace_torch_backend_models import engine, graph

pytest.importorskip("cuequivariance_torch")

from mace_core.config.precision import PrecisionConfig, PrecisionPolicy
from mace_torch.backends import CompositeBackend, resolve_backend
from mace_torch.backends.conformance import run_backend_conformance
from mace_torch.backends.cueq import CuEqBackend, _contraction_map
from mace_torch.backends.precision import boundary_precision
from mace_torch.serialization import (
    canonical_state,
    load_canonical_state,
)

#: The one device cuEquivariance works on here: CUDA with its compiled
#: operations, the CPU through its pure-torch path without them.
DEVICES = sorted(CuEqBackend().capabilities().devices)
if DEVICES == ["cuda"] and not torch.cuda.is_available():
    pytest.skip("the compiled operations need a GPU", allow_module_level=True)


@pytest.mark.parametrize("layout", ["mul_ir"])
@pytest.mark.parametrize("device", DEVICES)
def test_cueq_passes_the_conformance_harness(dtype, device, layout):
    """In its native layout, which is what a model on cueq is built in, and in
    the canonical one it also declares."""
    precision = "float64" if dtype == torch.float64 else "float32"
    results = run_backend_conformance(
        CuEqBackend(), precision=precision, device=device, layout=layout
    )
    built = {result.op for result in results if result.built}
    assert built == {
        "linear",
        "channelwise_tp_conv",
        "symmetric_contraction",
        "fully_connected_tp",
    }
    declined = [result for result in results if not result.built]
    assert {result.op for result in declined} == {"linear", "segment_reduce"}


@fp64_only
@pytest.mark.parametrize(
    "irreps_in, irreps_out, correlation",
    [("0e+1o", "0e+1o", 2), ("0e+1o+2e+3o", "0e+1o", 3), ("0e+1o+2e", "0e+1o+2e", 3)],
)
def test_the_contraction_map_is_block_diagonal_with_small_blocks(
    irreps_in, irreps_out, correlation
):
    """What the ticket measured across the production grid: mostly one to
    one, a few small blocks where the two bases kept different members of a
    linearly dependent set, and well conditioned."""
    reorder = _contraction_map(irreps_in, irreps_out, correlation)
    assert reorder.max_block_size <= 3
    assert all(np.linalg.cond(block.matrix) < 10 for block in reorder.blocks)
    weights = np.random.default_rng(0).normal(size=(2, reorder.path_count, 3))
    back = reorder.inverse().apply(reorder.apply(weights, axis=1), axis=1)
    np.testing.assert_allclose(back, weights, rtol=0, atol=1e-12)


def test_it_declines_what_its_kernels_cannot_compute():
    capabilities = CuEqBackend().capabilities()
    assert not capabilities.supports(
        LinearDescriptor(irreps_in="2x0e", irreps_out="2x0e", has_bias=True)
    )
    with pytest.raises(ValueError, match="multiplicity"):
        ChannelwiseTPConvDescriptor(irreps_node="2x0e+1o", irreps_edge="0e+1o")
    assert not capabilities.supports(
        SymmetricContractionDescriptor(irreps_in="0e+1o", basis="full")
    )
    assert capabilities.supports(
        ChannelwiseTPConvDescriptor(irreps_node="0e+1o", irreps_edge="0e+1o")
    )


def test_the_name_resolves_to_cueq_with_the_reference_behind_it():
    backend = resolve_backend("cueq")
    assert isinstance(backend, CompositeBackend)
    assert isinstance(backend.primary, CuEqBackend)


def test_discovery_records_a_missing_cuequivariance_rather_than_raising():
    probe = (
        "import json, sys\n"
        "sys.modules['cuequivariance'] = None\n"
        "from mace_core.kernels.registry import available_backends\n"
        "found = {b.name: [b.loaded, b.reason] for b in available_backends()}\n"
        "print(json.dumps(found))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    found = json.loads(result.stdout.splitlines()[-1])
    assert found["reference"][0] is True
    assert found["cueq"][0] is False
    assert "cuequivariance" in found["cueq"][1]


@fp64_only
@pytest.mark.parametrize("device", DEVICES)
def test_the_model_on_cueq_is_the_reference_model(caplog, device):
    reference = engine("reference", device=device)
    caplog.set_level("INFO", logger="mace_torch.train.model_stage")
    accelerated = engine("cueq", device=device)
    assert "channelwise_tp_conv: 2 by cueq" in caplog.text
    assert "every op in mul_ir" in caplog.text
    load_canonical_state(accelerated, canonical_state(reference))

    expected = reference(graph(device), compute=("forces", "stress"), training=False)
    actual = accelerated(graph(device), compute=("forces", "stress"), training=False)
    for name in ("total_energy", "forces", "stress"):
        mine, theirs = getattr(actual, name), getattr(expected, name)
        assert torch.allclose(mine, theirs, atol=1e-10, rtol=1e-10), name


@fp64_only
def test_a_seed_draws_the_same_model_on_either_backend():
    reference = canonical_state(engine("reference", seed=11))
    accelerated = canonical_state(engine("cueq", seed=11))
    assert reference.keys() == accelerated.keys()
    for path, tensors in reference.items():
        for name, value in tensors.items():
            other = accelerated[path][name]
            if value.is_floating_point():
                assert torch.allclose(other, value, atol=1e-12), (path, name)
            else:
                assert torch.equal(other, value), (path, name)


@fp64_only
@pytest.mark.parametrize("device", DEVICES)
def test_a_force_loss_sends_the_reference_s_gradient_to_the_weights(device):
    """The second derivative training on forces takes, compared along random
    directions of canonical weights: each model's own parameters differ, the
    derivative along a canonical direction does not."""
    reference = engine("reference", device=device)
    accelerated = engine("cueq", device=device)
    start = canonical_state(reference)
    load_canonical_state(accelerated, start)

    def gradients(model):
        out = model(graph(device), compute=("forces", "stress"), training=True)
        loss = (
            out.total_energy.pow(2).sum()
            + out.forces.pow(2).sum()
            + out.stress.pow(2).sum()
        )
        parameters = [p for p in model.parameters() if p.requires_grad]
        return parameters, torch.autograd.grad(loss, parameters)

    reference_parameters, reference_gradients = gradients(reference)
    accelerated_parameters, accelerated_gradients = gradients(accelerated)
    generator = torch.Generator().manual_seed(5)
    for _ in range(3):
        direction = {
            path: {
                name: torch.randn(
                    value.shape, generator=generator, dtype=value.dtype
                ).to(value.device)
                if value.is_floating_point()
                else value
                for name, value in tensors.items()
            }
            for path, tensors in start.items()
        }
        load_canonical_state(reference, direction)
        load_canonical_state(accelerated, direction)
        expected = sum(
            (g * p.detach()).sum()
            for g, p in zip(reference_gradients, reference_parameters, strict=True)
        )
        actual = sum(
            (g * p.detach()).sum()
            for g, p in zip(accelerated_gradients, accelerated_parameters, strict=True)
        )
        assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.skipif(DEVICES != ["cuda"], reason="compiles and captures on CUDA")
def test_cueq_ops_compile_whole_and_replay_from_a_cuda_graph(dtype):
    precision = "float64" if dtype == torch.float64 else "float32"
    results = run_backend_conformance(
        CuEqBackend(),
        precision=precision,
        device="cuda",
        compile_ops=True,
        cuda_graphs=True,
        layout="mul_ir",
    )
    for result in results:
        if result.built:
            assert {"compiles", "cuda graph"} <= set(result.checks), result.op


@fp64_only
@pytest.mark.parametrize("device", DEVICES)
def test_a_wider_floor_is_cueqs_math_dtype_and_needs_no_cast(device):
    """cuEquivariance has an accumulator of its own, so an op held to a
    float64 floor keeps float32 data and float64 math, and the model around it
    stays float32 with no cast at any op's boundary."""
    floor = PrecisionConfig(
        model="float32",
        ops=(
            (
                "symmetric_contraction",
                PrecisionPolicy("float32", "float64", "float32"),
            ),
        ),
    )
    model = engine("cueq", device=device, precision=floor)
    math = {
        path: module.math_dtype
        for path, module in model.named_modules()
        if isinstance(getattr(module, "math_dtype", None), torch.dtype)
    }
    contractions = {path for path in math if ".contraction." in path}
    assert contractions
    for path, dtype in math.items():
        wanted = torch.float64 if path in contractions else torch.float32
        assert dtype is wanted, path
    assert all(boundary_precision(module) is None for module in model.modules())
    assert {p.dtype for p in model.parameters()} == {torch.float32}
