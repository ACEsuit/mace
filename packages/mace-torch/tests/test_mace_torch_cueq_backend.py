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

from mace_torch.backends import CompositeBackend, resolve_backend
from mace_torch.backends.conformance import run_backend_conformance
from mace_torch.backends.cueq import CuEqBackend, _contraction_map
from mace_torch.serialization import (
    canonical_state,
    load_canonical_state,
)

#: The one device cuEquivariance works on here: CUDA with its compiled
#: operations, the CPU through its pure-torch path without them.
DEVICES = sorted(CuEqBackend().capabilities().devices)
if DEVICES == ["cuda"] and not torch.cuda.is_available():
    pytest.skip("the compiled operations need a GPU", allow_module_level=True)


@pytest.mark.parametrize("device", DEVICES)
def test_cueq_passes_the_conformance_harness(dtype, device):
    precision = "float64" if dtype == torch.float64 else "float32"
    results = run_backend_conformance(CuEqBackend(), precision=precision, device=device)
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
    projection = _contraction_map(irreps_in, irreps_out, correlation)
    support = np.abs(projection) > 1e-9
    assert support.sum(axis=0).max() <= 3
    assert support.sum(axis=1).max() <= 3
    assert np.linalg.cond(projection) < 10


def test_it_declines_what_its_kernels_cannot_compute():
    capabilities = CuEqBackend().capabilities()
    assert not capabilities.supports(
        LinearDescriptor(irreps_in="2x0e", irreps_out="2x0e", has_bias=True)
    )
    assert not capabilities.supports(
        ChannelwiseTPConvDescriptor(irreps_node="2x0e+1o", irreps_edge="0e+1o")
    )
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
