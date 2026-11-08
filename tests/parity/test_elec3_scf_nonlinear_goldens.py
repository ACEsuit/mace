"""Nonlinear FixedPointSCF against independently captured external references.

All legacy activation/readout families, periodicity profiles and both dtypes.
"""

import json

import pytest
import torch

from tests.golden.harness import FP32, FP64_CPU_REFERENCE
from tests.parity.elec3_fixed_point import PROFILES, graph_for, outputs
from tests.parity.elec3_scf_nonlinear import (
    VARIANTS,
    case_graph,
    load_model,
    reference_path,
)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32], ids=["float64", "float32"]
)
def test_nonlinear_scf_frozen_external_reference(variant, profile, dtype):
    model = load_model(variant, profile, dtype)
    actual = outputs(model, case_graph(variant, profile, dtype))
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
def test_nonlinear_scf_cuda_reference_and_cueq(variant, profile, dtype, backend):
    from tests.golden.harness import FP64_ACCELERATED_BACKEND

    model = load_model(variant, profile, dtype, backend).to("cuda")
    graph = {
        key: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for key, value in case_graph(variant, profile, dtype).items()
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
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32], ids=["float64", "float32"]
)
def test_nonlinear_block_values_and_first_second_derivatives(variant, dtype):
    from tests.parity.elec3_scf_nonlinear import block_outputs

    actual = block_outputs(load_model(variant, "molecular", dtype))
    expected = json.loads(reference_path().read_text())["blocks"][variant]
    row = FP64_CPU_REFERENCE if dtype == torch.float64 else FP32
    for name, value in actual.items():
        torch.testing.assert_close(
            value,
            torch.tensor(expected[name], dtype=value.dtype),
            atol=row.atol,
            rtol=row.rtol,
            msg=f"{variant}/{name}",
        )


@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32], ids=["float64", "float32"]
)
def test_norm_tanh_canonical_cueq_and_derivative_parity(tmp_path, dtype):
    from mace_torch.kernels import initialize_model_weights
    from mace_torch.serialization import load_checkpoint, save_checkpoint

    from tests.golden.harness import FP64_ACCELERATED_BACKEND
    from tests.parity.elec3_fixed_point import build_model

    options = {
        "constant_charge": False,
        "potential_embedding": "norm_tanh",
        "feature_max_l": 1,
        "multipole_max_l": 1,
        "local_source": "linear",
    }
    reference = build_model("full_periodic", dtype, **options)
    initialize_model_weights(reference, 7)
    checkpoint = tmp_path / "norm-tanh.safetensors"
    save_checkpoint(
        checkpoint, reference, {"model": reference.config.model_dump(mode="json")}
    )
    accelerated = build_model("full_periodic", dtype, "cueq", **options)
    load_checkpoint(checkpoint, lambda _: accelerated)
    reference.to("cuda")
    accelerated.to("cuda")
    graph = {
        key: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for key, value in graph_for("full_periodic", dtype).items()
    }
    expected, actual = outputs(reference, graph), outputs(accelerated, graph)
    row = FP64_ACCELERATED_BACKEND if dtype == torch.float64 else FP32
    for name in actual:
        torch.testing.assert_close(
            actual[name], expected[name], atol=row.atol, rtol=row.rtol, msg=name
        )
    actual["forces"].square().sum().backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in accelerated.parameters()
        if parameter.grad is not None
    )
