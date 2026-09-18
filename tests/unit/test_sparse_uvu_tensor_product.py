"""`SparseUvuTensorProduct` must agree with the `e3nn.o3.TensorProduct` it replaces.

The class is a hand-written torch specialisation of the sparse `uvu` tensor
products PolarMACE uses, written so that it can also read and write
cuEquivariance's `ir_mul` layout. It builds its weights from a reference
`o3.TensorProduct`, so that reference is the natural oracle: in `mul_ir` the two
must agree elementwise, and in `ir_mul` they must agree once the layout is
transposed on the way in and out.

These tests cover both supported path types (`l x l -> 0` and `l x 0 -> l`), the
degenerate `d = 1` case where the contraction over the irrep dimension is a
no-op, and `l` up to 2 so that `d = 2l+1` takes all of 1, 3 and 5. They need
nothing beyond e3nn, so they live here rather than under `tests/extensions`.
"""

from __future__ import annotations

import pytest
import torch
from e3nn import o3

from mace.modules.field_blocks import (
    SparseUvuTensorProduct,
    instructions_for_sparse_tp,
)

# (mul, l_max). Multiplicity above the irrep dimension is the regime the model
# runs in, and the one where the contraction order matters; the last case checks
# the degenerate opposite.
CASES = [
    (8, 0),  # scalars only: every path has d = 1
    (8, 1),  # adds d = 3
    (7, 2),  # adds d = 5, and a multiplicity that is not a power of two
    (1, 2),  # degenerate single channel
]


@pytest.fixture(name="float64")
def _float64():
    """Compare in double precision; the contraction orders differ by rounding."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _irreps(mul: int, l_max: int) -> o3.Irreps:
    return o3.Irreps([(mul, (l, (-1) ** l)) for l in range(l_max + 1)])


def _reference_for(sparse: SparseUvuTensorProduct, instructions) -> o3.TensorProduct:
    """The e3nn TensorProduct `sparse` stands in for, sharing its weights."""
    reference = o3.TensorProduct(
        sparse.irreps_in1,
        sparse.irreps_in2,
        sparse.irreps_out,
        instructions=instructions,
        shared_weights=True,
        internal_weights=True,
    )
    with torch.no_grad():
        reference.weight.copy_(sparse.weight)
    return reference


def _build(mul: int, l_max: int, layout: str):
    irreps_in1 = _irreps(mul, l_max)
    irreps_in2 = _irreps(mul, l_max)
    irreps_out = o3.Irreps(f"{mul}x0e")
    instructions = instructions_for_sparse_tp(irreps_in1, irreps_in2, irreps_out)
    sparse = SparseUvuTensorProduct(
        irreps_in1, irreps_in2, irreps_out, instructions=instructions, layout=layout
    )
    return sparse, _reference_for(sparse, instructions)


def _transpose_layout(x: torch.Tensor, irreps: o3.Irreps, to_ir_mul: bool):
    """Re-lay a [batch, irreps.dim] tensor between mul_ir and ir_mul ordering."""
    batch = x.shape[0]
    blocks = []
    for (mul, ir), block in zip(irreps, irreps.slices()):
        shape = (mul, ir.dim) if to_ir_mul else (ir.dim, mul)
        blocks.append(
            x[:, block].view(batch, *shape).transpose(1, 2).reshape(batch, -1)
        )
    return torch.cat(blocks, dim=-1)


@pytest.mark.parametrize("mul,l_max", CASES)
def test_it_matches_e3nn_in_the_mul_ir_layout(mul: int, l_max: int, float64):
    torch.manual_seed(0)
    sparse, reference = _build(mul, l_max, "mul_ir")
    x1 = sparse.irreps_in1.randn(16, -1)
    x2 = sparse.irreps_in2.randn(16, -1)
    torch.testing.assert_close(sparse(x1, x2), reference(x1, x2))


@pytest.mark.parametrize("mul,l_max", CASES)
def test_it_matches_e3nn_in_the_ir_mul_layout(mul: int, l_max: int, float64):
    """The ir_mul path must produce the same physics, only laid out differently."""
    torch.manual_seed(0)
    sparse, reference = _build(mul, l_max, "ir_mul")
    x1 = sparse.irreps_in1.randn(16, -1)
    x2 = sparse.irreps_in2.randn(16, -1)
    got = sparse(
        _transpose_layout(x1, sparse.irreps_in1, True),
        _transpose_layout(x2, sparse.irreps_in2, True),
    )
    torch.testing.assert_close(
        _transpose_layout(got, sparse.irreps_out, False), reference(x1, x2)
    )


@pytest.mark.parametrize("mul,l_max", CASES)
def test_its_gradients_match_e3nn(mul: int, l_max: int, float64):
    """Forces and fine-tuning reach this block through autograd, so the
    backward must agree too -- in the weights as well as the inputs."""
    torch.manual_seed(0)
    sparse, reference = _build(mul, l_max, "mul_ir")
    x1 = sparse.irreps_in1.randn(16, -1)
    x2 = sparse.irreps_in2.randn(16, -1)

    grads = []
    for module in (sparse, reference):
        module.zero_grad(set_to_none=True)
        a, b = x1.clone().requires_grad_(True), x2.clone().requires_grad_(True)
        module(a, b).square().sum().backward()
        grads.append((a.grad, b.grad, module.weight.grad))
    for got, want in zip(grads[0], grads[1]):
        torch.testing.assert_close(got, want)


def test_the_scalar_times_vector_path_matches_e3nn(float64):
    """`l x 0 -> l` is the other supported path; CASES only reach `l x l -> 0`."""
    torch.manual_seed(0)
    irreps = o3.Irreps("8x0e+8x1o")
    instructions = [(0, 0, 0, "uvu", True), (1, 0, 1, "uvu", True)]
    sparse = SparseUvuTensorProduct(
        irreps, irreps, irreps, instructions=instructions, layout="mul_ir"
    )
    modes = {meta[9] for meta in sparse._path_meta}  # pylint: disable=protected-access
    assert modes == {0, 1}, f"expected both path modes, got {modes}"

    reference = _reference_for(sparse, instructions)
    x1 = irreps.randn(16, -1)
    x2 = irreps.randn(16, -1)
    torch.testing.assert_close(sparse(x1, x2), reference(x1, x2))


def test_no_intermediate_reaches_the_squared_multiplicity(float64, monkeypatch):
    """Guard the contraction order: nothing of size batch*mul1*mul2 is allocated.

    Summing over `v` last builds a [B, mul1, mul2] tensor and contracts it away
    immediately. That intermediate sets the peak allocation of a PolarMACE
    forward, and a regression is silent -- it surfaces only as an OOM at a
    system size that used to fit. Here the forbidden intermediate is 1048576
    elements against 24576 for the largest legitimate one.
    """
    torch.manual_seed(0)
    mul, batch = 128, 64
    irreps_in1 = _irreps(mul, 1)
    irreps_out = o3.Irreps(f"{mul}x0e")
    instructions = instructions_for_sparse_tp(irreps_in1, irreps_in1, irreps_out)
    sparse = SparseUvuTensorProduct(
        irreps_in1, irreps_in1, irreps_out, instructions=instructions, layout="mul_ir"
    )
    x1 = irreps_in1.randn(batch, -1)
    x2 = irreps_in1.randn(batch, -1)

    sizes = []
    original_einsum = torch.einsum

    def recording_einsum(equation, *operands):
        result = original_einsum(equation, *operands)
        sizes.append(result.numel())
        return result

    monkeypatch.setattr(torch, "einsum", recording_einsum)
    sparse(x1, x2)

    assert sizes, "expected the forward to use einsum"
    assert max(sizes) < batch * mul * mul, (
        f"an intermediate of {max(sizes)} elements was allocated; anything "
        f"reaching batch*mul1*mul2 = {batch * mul * mul} means the "
        f"[B, mul1, mul2] outer product is back"
    )
