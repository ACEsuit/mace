"""Standalone #1713 reproduction; no foundation checkpoint or GPU required."""

import io
import math

import pytest
import torch

from mace.modules.gate import GatedEquivariantBlock


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("kind", ["gated", "scalars_only", "no_activation"])
def test_gate_script_roundtrip_preserves_values_and_gradients(
    layout, dtype, batch_shape, kind
):
    scalar_only = kind == "scalars_only"
    active = kind != "no_activation"
    gate = GatedEquivariantBlock(
        "2x0e",
        [torch.nn.functional.silu if active else None],
        "" if scalar_only else "2x0e",
        [] if scalar_only else [torch.sigmoid if active else None],
        "" if scalar_only else "2x1o",
        layout=layout,
    ).to(dtype=dtype)
    shape = batch_shape + (gate.irreps_in.dim,)
    x = torch.linspace(-1.0, 1.0, math.prod(shape), dtype=dtype).reshape(shape)
    x.requires_grad_(True)
    expected = gate(x)
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    assert torch.isfinite(expected).all()
    assert torch.isfinite(expected_grad).all()

    scripted = torch.jit.script(gate)
    archive = io.BytesIO()
    torch.jit.save(scripted, archive)
    archive.seek(0)
    restored = torch.jit.load(archive)
    restored_x = x.detach().clone().requires_grad_(True)
    actual = restored(restored_x)
    actual_grad = torch.autograd.grad(actual.square().sum(), restored_x)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_legacy_gate_checkpoint_without_activation_flags():
    gate = GatedEquivariantBlock(
        "2x0e",
        [torch.nn.functional.silu],
        "2x0e",
        [torch.sigmoid],
        "2x1o",
    )
    x = torch.linspace(-1.0, 1.0, 10)
    expected = gate(x)
    # Simulate the actual attribute set of older full-module checkpoints.
    del gate._has_act_scalar
    del gate._has_act_gate
    archive = io.BytesIO()
    torch.save(gate, archive)
    archive.seek(0)
    restored = torch.load(archive, weights_only=False)
    torch.testing.assert_close(restored(x), expected)
    torch.testing.assert_close(torch.jit.script(restored)(x), expected)
