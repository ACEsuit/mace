"""LSC extensions compared with pinned external values and training gradients."""

import json

import pytest
import torch

from tests.golden.harness import FP32, FP64_ACCELERATED_BACKEND, FP64_CPU_REFERENCE
from tests.parity.elec3 import PROFILES
from tests.parity.elec3_local_extensions import (
    VARIANTS,
    graph_for,
    load_model,
    outputs,
    reference_path,
)


def compare(variant, degree, profile, dtype, backend="reference", device="cpu"):
    model = load_model(variant, degree, profile, dtype, backend).to(device)
    graph = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in graph_for(variant, profile, dtype).items()
    }
    actual = outputs(model, graph)
    expected = json.loads(reference_path().read_text())["cases"][
        f"{variant}/l{degree}/{profile}"
    ]
    row = (
        FP32
        if dtype == torch.float32
        else (FP64_CPU_REFERENCE if device == "cpu" else FP64_ACCELERATED_BACKEND)
    )
    assert set(actual) == set(expected)
    for name, value in actual.items():
        torch.testing.assert_close(
            value.cpu(),
            torch.tensor(expected[name], dtype=value.dtype),
            atol=row.atol,
            rtol=row.rtol,
            msg=f"{variant} l{degree} {profile} {backend} {name}",
        )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("degree", [0, 1])
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32], ids=["float64", "float32"]
)
def test_external_reference_cpu(variant, degree, profile, dtype):
    compare(variant, degree, profile, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "backend", ["reference", pytest.param("cueq", marks=pytest.mark.cueq)]
)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("degree", [0, 1])
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32], ids=["float64", "float32"]
)
def test_external_reference_gpu(variant, degree, profile, dtype, backend):
    compare(variant, degree, profile, dtype, backend, "cuda")
