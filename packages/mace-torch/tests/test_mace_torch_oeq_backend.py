"""The OpenEquivariance backend: the fused convolution, and the reference for
everything else.

OpenEquivariance compiles for CUDA only, so these run on a GPU host and skip
elsewhere. The model built on it is compared with the reference model carrying
the same canonical weights.
"""

from __future__ import annotations

import pytest
import torch
from conftest import fp64_only

pytest.importorskip("openequivariance")
if not torch.cuda.is_available():
    pytest.skip("OpenEquivariance compiles for CUDA only", allow_module_level=True)

from mace_torch.backends import CompositeBackend, resolve_backend
from mace_torch.backends.conformance import run_backend_conformance
from mace_torch.backends.oeq import OeqBackend
from mace_torch.serialization import canonical_state, load_canonical_state
from mace_torch_backend_models import engine, graph


def test_oeq_passes_the_conformance_harness(dtype):
    precision = "float64" if dtype == torch.float64 else "float32"
    results = run_backend_conformance(OeqBackend(), precision=precision, device="cuda")
    assert {result.op for result in results if result.built} == {"channelwise_tp_conv"}


def test_the_name_resolves_to_oeq_with_the_reference_behind_it():
    backend = resolve_backend("oeq")
    assert isinstance(backend, CompositeBackend)
    assert isinstance(backend.primary, OeqBackend)


@fp64_only
def test_the_model_on_oeq_is_the_reference_model():
    reference = engine("reference", device="cuda")
    accelerated = engine("oeq", device="cuda")
    load_canonical_state(accelerated, canonical_state(reference))
    expected = reference(graph("cuda"), compute=("forces", "stress"), training=False)
    actual = accelerated(graph("cuda"), compute=("forces", "stress"), training=False)
    for name in ("total_energy", "forces", "stress"):
        mine, theirs = getattr(actual, name), getattr(expected, name)
        assert torch.allclose(mine, theirs, atol=1e-10, rtol=1e-10), name
