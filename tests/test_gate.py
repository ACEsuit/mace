"""Tests for GatedEquivariantBlock vs e3nn nn.Gate.

Parametrized over irreps configs, dtypes, batch sizes, and layouts.
Compares forward values, backward gradients, second derivatives, and
verifies zero graph breaks under torch.compile for both mul_ir and ir_mul.
"""

import pytest
import torch
import torch.serialization

torch.serialization.add_safe_globals([slice])

from e3nn import nn as e3nn_nn  # noqa: E402
from e3nn import o3  # noqa: E402

from mace.modules.gate import GatedEquivariantBlock  # noqa: E402

# ---------------------------------------------------------------------------
# Irreps configurations matching MACE usage patterns
# ---------------------------------------------------------------------------
IRREPS_CONFIGS = {
    "scalars_only": {
        "irreps_scalars": o3.Irreps("128x0e"),
        "irreps_gates": o3.Irreps(""),
        "irreps_gated": o3.Irreps(""),
    },
    "scalars_l1": {
        "irreps_scalars": o3.Irreps("128x0e"),
        "irreps_gates": o3.Irreps("128x0e"),
        "irreps_gated": o3.Irreps("128x1o"),
    },
    "scalars_l1_l2": {
        "irreps_scalars": o3.Irreps("64x0e"),
        "irreps_gates": o3.Irreps("64x0e+64x0e"),
        "irreps_gated": o3.Irreps("64x1o+64x2e"),
    },
    "scalars_l1_l2_simplified_gates": {
        "irreps_scalars": o3.Irreps("64x0e"),
        "irreps_gates": o3.Irreps("128x0e"),
        "irreps_gated": o3.Irreps("64x1o+64x2e"),
    },
}

CONFIGS_WITH_GATED = [k for k, v in IRREPS_CONFIGS.items() if v["irreps_gated"].dim > 0]
DTYPES = [torch.float32, torch.float64]
BATCH_SIZES = [1, 32, 256]
LAYOUTS = ["mul_ir", "ir_mul"]


def _make_gates(config, layout="mul_ir", act_scalars=None, act_gates=None):
    """Build both an e3nn Gate and a GatedEquivariantBlock from the same spec."""
    irreps_scalars = config["irreps_scalars"]
    irreps_gates = config["irreps_gates"]
    irreps_gated = config["irreps_gated"]

    if act_scalars is None:
        act_scalars = [torch.nn.functional.silu for _ in irreps_scalars]
    if act_gates is None:
        act_gates = [torch.nn.functional.sigmoid] * len(irreps_gates)

    ref = e3nn_nn.Gate(
        irreps_scalars=irreps_scalars,
        act_scalars=act_scalars,
        irreps_gates=irreps_gates,
        act_gates=act_gates,
        irreps_gated=irreps_gated,
    )
    ours = GatedEquivariantBlock(
        irreps_scalars=irreps_scalars,
        act_scalars=act_scalars,
        irreps_gates=irreps_gates,
        act_gates=act_gates,
        irreps_gated=irreps_gated,
        layout=layout,
    )
    return ref, ours


# ---------------------------------------------------------------------------
# Layout transpose helper
# ---------------------------------------------------------------------------
def _transpose_group(x: torch.Tensor, mul: int, ir_dim: int, to_ir_mul: bool):
    """Transpose a single (mul, ir) group between mul_ir and ir_mul layouts."""
    batch_shape = x.shape[:-1]
    if to_ir_mul:
        return (
            x.reshape(*batch_shape, mul, ir_dim)
            .transpose(-1, -2)
            .reshape(*batch_shape, mul * ir_dim)
        )
    return (
        x.reshape(*batch_shape, ir_dim, mul)
        .transpose(-1, -2)
        .reshape(*batch_shape, mul * ir_dim)
    )


def _transpose_irreps(x: torch.Tensor, irreps: o3.Irreps, to_ir_mul: bool):
    """Transpose a full irreps tensor between mul_ir and ir_mul layouts."""
    parts = []
    offset = 0
    for mul_ir in irreps:
        dim = mul_ir.dim
        chunk = x.narrow(-1, offset, dim)
        ir_dim = mul_ir.ir.dim
        if ir_dim > 1:
            chunk = _transpose_group(chunk, mul_ir.mul, ir_dim, to_ir_mul)
        parts.append(chunk)
        offset += dim
    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Test: irreps_in / irreps_out properties
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config_name", IRREPS_CONFIGS.keys())
@pytest.mark.parametrize("layout", LAYOUTS)
def test_irreps_properties(config_name, layout):
    config = IRREPS_CONFIGS[config_name]
    ref, ours = _make_gates(config, layout=layout)
    assert (
        ours.irreps_in == ref.irreps_in
    ), f"irreps_in mismatch: {ours.irreps_in} vs {ref.irreps_in}"
    assert (
        ours.irreps_out == ref.irreps_out
    ), f"irreps_out mismatch: {ours.irreps_out} vs {ref.irreps_out}"


# ---------------------------------------------------------------------------
# Test: forward values match e3nn (mul_ir layout, direct comparison)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config_name", IRREPS_CONFIGS.keys())
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=["B1", "B32", "B256"])
def test_forward_matches_e3nn(config_name, dtype, batch_size):
    config = IRREPS_CONFIGS[config_name]
    ref, ours = _make_gates(config, layout="mul_ir")
    ref, ours = ref.to(dtype), ours.to(dtype)

    torch.manual_seed(42)
    x = torch.randn(batch_size, ref.irreps_in.dim, dtype=dtype)

    y_ref = ref(x)
    y_ours = ours(x)

    atol = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(
        y_ours, y_ref, atol=atol, rtol=0
    ), f"Forward mismatch: max diff = {(y_ours - y_ref).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test: ir_mul layout produces same result as mul_ir with manual transposes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config_name", CONFIGS_WITH_GATED)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("batch_size", [1, 64], ids=["B1", "B64"])
def test_ir_mul_matches_mul_ir_via_transpose(config_name, dtype, batch_size):
    """ir_mul gate on transposed input should give same output as mul_ir gate."""
    config = IRREPS_CONFIGS[config_name]

    gate_mul_ir = GatedEquivariantBlock(
        irreps_scalars=config["irreps_scalars"],
        act_scalars=[torch.nn.functional.silu] * len(config["irreps_scalars"]),
        irreps_gates=config["irreps_gates"],
        act_gates=[torch.nn.functional.sigmoid] * len(config["irreps_gates"]),
        irreps_gated=config["irreps_gated"],
        layout="mul_ir",
    ).to(dtype)

    gate_ir_mul = GatedEquivariantBlock(
        irreps_scalars=config["irreps_scalars"],
        act_scalars=[torch.nn.functional.silu] * len(config["irreps_scalars"]),
        irreps_gates=config["irreps_gates"],
        act_gates=[torch.nn.functional.sigmoid] * len(config["irreps_gates"]),
        irreps_gated=config["irreps_gated"],
        layout="ir_mul",
    ).to(dtype)

    torch.manual_seed(99)
    x_mul_ir = torch.randn(batch_size, gate_mul_ir.irreps_in.dim, dtype=dtype)

    x_ir_mul = _transpose_irreps(x_mul_ir, gate_mul_ir.irreps_in, to_ir_mul=True)

    y_mul_ir = gate_mul_ir(x_mul_ir)
    y_ir_mul = gate_ir_mul(x_ir_mul)

    y_ir_mul_as_mul_ir = _transpose_irreps(
        y_ir_mul, gate_ir_mul.irreps_out, to_ir_mul=False
    )

    atol = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(y_mul_ir, y_ir_mul_as_mul_ir, atol=atol, rtol=0), (
        f"Layout mismatch: max diff = "
        f"{(y_mul_ir - y_ir_mul_as_mul_ir).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test: backward gradients match e3nn
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config_name", IRREPS_CONFIGS.keys())
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=["B1", "B32", "B256"])
def test_backward_matches_e3nn(config_name, dtype, batch_size):
    config = IRREPS_CONFIGS[config_name]
    ref, ours = _make_gates(config, layout="mul_ir")
    ref, ours = ref.to(dtype), ours.to(dtype)

    torch.manual_seed(42)
    x_ref = torch.randn(batch_size, ref.irreps_in.dim, dtype=dtype, requires_grad=True)
    x_ours = x_ref.detach().clone().requires_grad_(True)

    y_ref = ref(x_ref)
    y_ours = ours(x_ours)

    grad_out = torch.randn_like(y_ref)
    y_ref.backward(grad_out)
    y_ours.backward(grad_out)

    atol = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(
        x_ours.grad, x_ref.grad, atol=atol, rtol=0
    ), f"Backward mismatch: max diff = {(x_ours.grad - x_ref.grad).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test: backward gradients for ir_mul layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config_name", CONFIGS_WITH_GATED)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_backward_ir_mul(config_name, dtype):
    """Gradients through ir_mul gate should be self-consistent."""
    config = IRREPS_CONFIGS[config_name]
    gate = GatedEquivariantBlock(
        irreps_scalars=config["irreps_scalars"],
        act_scalars=[torch.nn.functional.silu] * len(config["irreps_scalars"]),
        irreps_gates=config["irreps_gates"],
        act_gates=[torch.nn.functional.sigmoid] * len(config["irreps_gates"]),
        irreps_gated=config["irreps_gated"],
        layout="ir_mul",
    ).to(dtype)

    torch.manual_seed(77)
    x = torch.randn(16, gate.irreps_in.dim, dtype=dtype, requires_grad=True)
    y = gate(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


# ---------------------------------------------------------------------------
# Test: second derivatives (Hessian-vector products) match e3nn
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config_name", IRREPS_CONFIGS.keys())
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_second_derivative_matches_e3nn(config_name, dtype):
    config = IRREPS_CONFIGS[config_name]
    ref, ours = _make_gates(config, layout="mul_ir")
    ref, ours = ref.to(dtype), ours.to(dtype)

    batch_size = 8
    torch.manual_seed(42)
    x_ref = torch.randn(batch_size, ref.irreps_in.dim, dtype=dtype, requires_grad=True)
    x_ours = x_ref.detach().clone().requires_grad_(True)

    grad_out = torch.randn(batch_size, ref.irreps_out.dim, dtype=dtype)
    v = torch.randn_like(x_ref)

    y_ref = ref(x_ref)
    (g_ref,) = torch.autograd.grad(y_ref, x_ref, grad_out, create_graph=True)
    (hvp_ref,) = torch.autograd.grad(g_ref, x_ref, v)

    y_ours = ours(x_ours)
    (g_ours,) = torch.autograd.grad(y_ours, x_ours, grad_out, create_graph=True)
    (hvp_ours,) = torch.autograd.grad(g_ours, x_ours, v)

    atol = 1e-4 if dtype == torch.float32 else 1e-10
    assert torch.allclose(
        hvp_ours, hvp_ref, atol=atol, rtol=0
    ), f"Second-derivative mismatch: max diff = {(hvp_ours - hvp_ref).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test: no graph breaks under torch.compile (both layouts)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "config_name", ["scalars_l1", "scalars_l1_l2", "scalars_l1_l2_simplified_gates"]
)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_no_graph_breaks(config_name, layout):
    config = IRREPS_CONFIGS[config_name]
    _, ours = _make_gates(config, layout=layout)

    x = torch.randn(4, ours.irreps_in.dim)
    explanation = torch._dynamo.explain(ours)(x)
    assert explanation.graph_break_count == 0, (
        f"Graph breaks found ({layout}): {explanation.graph_break_count}\n"
        f"Break reasons: {explanation.break_reasons}"
    )


# ---------------------------------------------------------------------------
# Test: custom activation functions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_custom_activations(dtype):
    config = IRREPS_CONFIGS["scalars_l1"]
    act_scalars = [torch.tanh for _ in config["irreps_scalars"]]
    act_gates = [torch.tanh] * len(config["irreps_gates"])

    ref, ours = _make_gates(
        config, layout="mul_ir", act_scalars=act_scalars, act_gates=act_gates
    )
    ref, ours = ref.to(dtype), ours.to(dtype)

    torch.manual_seed(123)
    x = torch.randn(16, ref.irreps_in.dim, dtype=dtype)

    atol = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(ours(x), ref(x), atol=atol, rtol=0)


# ---------------------------------------------------------------------------
# Test: invalid layout raises ValueError
# ---------------------------------------------------------------------------
def test_invalid_layout():
    config = IRREPS_CONFIGS["scalars_l1"]
    with pytest.raises(ValueError, match="layout must be"):
        GatedEquivariantBlock(
            irreps_scalars=config["irreps_scalars"],
            act_scalars=[torch.nn.functional.silu],
            irreps_gates=config["irreps_gates"],
            act_gates=[torch.nn.functional.sigmoid],
            irreps_gated=config["irreps_gated"],
            layout="bad_layout",
        )


# ---------------------------------------------------------------------------
# Test: repr includes layout info
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("layout", LAYOUTS)
def test_repr(layout):
    config = IRREPS_CONFIGS["scalars_l1"]
    _, ours = _make_gates(config, layout=layout)
    r = repr(ours)
    assert layout in r


# ---------------------------------------------------------------------------
# Test: _normalize2mom_cst caching
# ---------------------------------------------------------------------------
def test_normalize2mom_caching():
    from mace.modules.gate import _NORM_CACHE, _normalize2mom_cst

    _NORM_CACHE.clear()
    fn = torch.nn.functional.silu
    c1 = _normalize2mom_cst(fn)
    c2 = _normalize2mom_cst(fn)
    assert c1 == c2
    assert id(fn) in _NORM_CACHE
