"""Affine FixedPointSCF models against the frozen, external MACE 0.3.14 oracle.

No external source checkout is needed in CI. The canonical tiny weights and
outputs were captured together by the explicit ELEC-3 golden target.
"""

import json

import pytest
import torch

from tests.golden.harness import FP32, FP64_CPU_REFERENCE
from tests.parity.elec3_fixed_point import (
    PROFILES,
    VARIANTS,
    graph_for,
    load_model,
    outputs,
    reference_path,
)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32], ids=["float64", "float32"]
)
def test_fixed_point_frozen_external_reference(variant, profile, dtype):
    model = load_model(variant, profile, dtype)
    actual = outputs(model, graph_for(profile, dtype))
    expected = json.loads(reference_path().read_text())["cases"][f"{variant}/{profile}"]
    row = FP64_CPU_REFERENCE if dtype == torch.float64 else FP32
    assert set(actual) == set(expected)
    for name, value in actual.items():
        torch.testing.assert_close(
            value,
            torch.tensor(expected[name], dtype=value.dtype),
            atol=row.atol,
            rtol=row.rtol,
            msg=f"{variant} {profile} {name}",
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "backend", ["reference", pytest.param("cueq", marks=pytest.mark.cueq)]
)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32], ids=["float64", "float32"]
)
def test_fixed_point_cuda_reference_and_cueq(variant, profile, dtype, backend):
    from tests.golden.harness import FP64_ACCELERATED_BACKEND

    model = load_model(variant, profile, dtype, backend).to("cuda")
    graph = {
        key: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for key, value in graph_for(profile, dtype).items()
    }
    actual = outputs(model, graph)
    expected = json.loads(reference_path().read_text())["cases"][f"{variant}/{profile}"]
    row = FP64_ACCELERATED_BACKEND if dtype == torch.float64 else FP32
    for name, value in actual.items():
        torch.testing.assert_close(
            value.cpu(),
            torch.tensor(expected[name], dtype=value.dtype),
            atol=row.atol,
            rtol=row.rtol,
            msg=f"{variant} {profile} {backend} {name}",
        )
    (actual["forces"].square().sum() + actual["stress"].square().sum()).backward()
    gradients = [
        parameter.grad
        for parameter in model.updates.parameters()
        if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
