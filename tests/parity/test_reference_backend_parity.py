"""The reference kernel backend against the frozen legacy ops, in one process.

Each op is compared twice: the forward with the same weights, and the gradient
with respect to those weights. The forward alone cannot tell where a constant
sits. A ``1 / sqrt(fan_in)`` folded into a stored weight gives the forward of
one applied in the computation and a gradient ``sqrt(fan_in)`` times larger,
which is a different training run. The canonical layout holds the raw weights
``e3nn`` holds, so for the linear map and the skip connection the two weight
gradients must agree element by element once the layouts are matched.

The layouts differ by a permutation, which each test derives from the legacy
module's own instruction list rather than from the reference's code. The
symmetric contraction differs by more than a permutation: legacy stores
weights against its own coupling matrices and the reference against the
reduced basis, which span the same functions on symmetric inputs. The test
solves for the linear map between the two and carries weights and gradients
through it.

The legacy side is the module legacy models build: ``e3nn``'s linear and fully
connected tensor product through ``mace.modules.wrapper_ops`` (``e3nn``'s own
linear for the bias case, which the wrapper never asks for), the channelwise
tensor product with the instructions ``mace.modules.irreps_tools`` computes,
and ``mace.modules.symmetric_contraction``.
"""

import itertools
import math

import numpy as np
import pytest
import torch
from e3nn import o3
from mace_core.kernels import (
    ChannelwiseTPConvDescriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    SymmetricContractionDescriptor,
)
from mace_torch.backends.reference import ReferenceBackend

from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from mace.modules.symmetric_contraction import SymmetricContraction
from mace.modules.wrapper_ops import FullyConnectedTensorProduct, TensorProduct
from mace.modules.wrapper_ops import Linear as LegacyLinear
from mace.tools.scatter import scatter_sum
from tests.golden.harness import tolerance

#: Two implementations of the same algebra in one process at fp64. They differ
#: in the order of the arithmetic only, as the closed forms in the other files
#: here do, so the same row applies.
CLOSED_FORM = tolerance("closed_form_fp64")

BACKEND = ReferenceBackend()


def assert_parity(reference: torch.Tensor, legacy: torch.Tensor, what: str) -> None:
    torch.testing.assert_close(
        reference,
        legacy,
        atol=CLOSED_FORM.atol,
        rtol=CLOSED_FORM.rtol,
        msg=lambda message: f"{what}: {message}",
    )


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _canonical_linear_index(irreps_in: o3.Irreps, irreps_out: o3.Irreps) -> dict:
    """``(i_out, copy_out, i_in, copy_in) -> canonical position``.

    The canonical order as the layout states it: output copies outermost, and
    within one output copy the matching input copies in declaration order.
    """
    index, position = {}, 0
    for i_out, (multiplicity_out, irrep_out) in enumerate(irreps_out):
        for copy_out in range(multiplicity_out):
            for i_in, (multiplicity_in, irrep_in) in enumerate(irreps_in):
                if irrep_in != irrep_out:
                    continue
                for copy_in in range(multiplicity_in):
                    index[(i_out, copy_out, i_in, copy_in)] = position
                    position += 1
    return index


# ---------------------------------------------------------------------------
# The linear map
# ---------------------------------------------------------------------------


def _linear_permutation(legacy: o3.Linear) -> torch.Tensor:
    """For each flat legacy weight, the canonical weight it is."""
    canonical = _canonical_linear_index(legacy.irreps_in, legacy.irreps_out)
    order = []
    for instruction in legacy.instructions:
        if instruction.i_in == -1:
            continue
        multiplicity_in, multiplicity_out = instruction.path_shape
        for copy_in in range(multiplicity_in):
            for copy_out in range(multiplicity_out):
                order.append(
                    canonical[(instruction.i_out, copy_out, instruction.i_in, copy_in)]
                )
    return torch.tensor(order)


LINEAR_CASES = [
    ("8x0e+4x1o", "6x0e+3x1o", False),
    ("4x0e+2x0e+3x1o", "5x0e+2x1o", False),
    ("4x0e+2x1o+2x2e", "3x0e+3x0e+2x2e", False),
    ("8x0e+4x1o", "6x0e+3x1o", True),
]


@pytest.mark.parametrize(("irreps_in", "irreps_out", "has_bias"), LINEAR_CASES)
def test_the_linear_map_matches_legacy_in_value_and_weight_gradient(
    fp64, irreps_in, irreps_out, has_bias
):
    """The repeated ``0e`` input pins the fan-in, summed over both terms."""
    legacy = (
        o3.Linear(irreps_in, irreps_out, biases=True)
        if has_bias
        else LegacyLinear(o3.Irreps(irreps_in), o3.Irreps(irreps_out))
    )
    reference = BACKEND.make_linear(
        LinearDescriptor(irreps_in=irreps_in, irreps_out=irreps_out, has_bias=has_bias)
    )
    reference.initialize_weights(3)
    permutation = _linear_permutation(legacy)
    with torch.no_grad():
        legacy.weight.copy_(reference.weight[permutation])
        if has_bias:
            reference.bias.copy_(
                torch.randn(reference.bias.shape, generator=_generator(4))
            )
            legacy.bias.copy_(reference.bias)

    features = torch.randn(7, o3.Irreps(irreps_in).dim, generator=_generator(5))
    cotangent = torch.randn(7, o3.Irreps(irreps_out).dim, generator=_generator(6))
    reference_out, legacy_out = reference(features), legacy(features)
    assert_parity(reference_out, legacy_out, "linear forward")

    (reference_out * cotangent).sum().backward()
    (legacy_out * cotangent).sum().backward()
    assert_parity(reference.weight.grad[permutation], legacy.weight.grad, "linear dW")
    if has_bias:
        assert_parity(reference.bias.grad, legacy.bias.grad, "linear d bias")


# ---------------------------------------------------------------------------
# The skip connection's fully connected tensor product
# ---------------------------------------------------------------------------


def _skip_permutation(legacy: o3.FullyConnectedTensorProduct) -> torch.Tensor:
    """For each flat legacy weight, its flat position in ``[scalar, weight]``."""
    canonical = _canonical_linear_index(legacy.irreps_in1, legacy.irreps_out)
    per_scalar = len(canonical)
    scalar_offsets = np.cumsum([0] + [mul for mul, _ in legacy.irreps_in2])
    order = []
    for instruction in legacy.instructions:
        multiplicity_1, multiplicity_2, multiplicity_out = instruction.path_shape
        for copy_1 in range(multiplicity_1):
            for copy_2 in range(multiplicity_2):
                scalar = int(scalar_offsets[instruction.i_in2]) + copy_2
                for copy_out in range(multiplicity_out):
                    weight = canonical[
                        (instruction.i_out, copy_out, instruction.i_in1, copy_1)
                    ]
                    order.append(scalar * per_scalar + weight)
    return torch.tensor(order)


SKIP_CASES = [
    ("8x0e+4x1o", "3x0e", "6x0e+3x1o"),
    ("4x0e+2x0e+3x1o", "3x0e", "5x0e+2x1o"),
    ("4x0e+2x1o", "2x0e+1x0e", "3x0e+2x1o"),
]


@pytest.mark.parametrize(("irreps_in1", "irreps_in2", "irreps_out"), SKIP_CASES)
def test_the_skip_connection_matches_legacy_in_value_and_weight_gradient(
    fp64, irreps_in1, irreps_in2, irreps_out
):
    """``4x0e+2x0e`` against ``3x0e`` is the case whose fan-in is shared: both
    paths into ``0e`` take ``1 / sqrt(18)``, not ``1 / sqrt(12)`` and
    ``1 / sqrt(6)``."""
    legacy = FullyConnectedTensorProduct(
        o3.Irreps(irreps_in1), o3.Irreps(irreps_in2), o3.Irreps(irreps_out)
    )
    reference = BACKEND.make_fully_connected_tp(
        FullyConnectedTPDescriptor(
            irreps_in1=irreps_in1, irreps_in2=irreps_in2, irreps_out=irreps_out
        )
    )
    reference.initialize_weights(7)
    permutation = _skip_permutation(legacy)
    with torch.no_grad():
        legacy.weight.copy_(reference.weight.reshape(-1)[permutation])

    num_scalars = o3.Irreps(irreps_in2).dim
    features = torch.randn(9, o3.Irreps(irreps_in1).dim, generator=_generator(8))
    attributes = torch.eye(num_scalars)[torch.arange(9) % num_scalars]
    cotangent = torch.randn(9, o3.Irreps(irreps_out).dim, generator=_generator(9))
    reference_out = reference(features, attributes)
    legacy_out = legacy(features, attributes)
    assert_parity(reference_out, legacy_out, "skip forward")

    (reference_out * cotangent).sum().backward()
    (legacy_out * cotangent).sum().backward()
    assert_parity(
        reference.weight.grad.reshape(-1)[permutation], legacy.weight.grad, "skip dW"
    )


# ---------------------------------------------------------------------------
# The channelwise convolution
# ---------------------------------------------------------------------------


def _per_channel_to_legacy(values: torch.Tensor, irreps: str) -> torch.Tensor:
    """``[n, channels, dim]`` per channel to legacy's flat ``mul_ir``."""
    blocks = []
    for piece, _ in _slices(irreps):
        blocks.append(values[:, :, piece].reshape(values.shape[0], -1))
    return torch.cat(blocks, dim=-1)


def _slices(irreps: str):
    start = 0
    for _, irrep in o3.Irreps(irreps):
        yield slice(start, start + irrep.dim), irrep
        start += irrep.dim


CONVOLUTION_CASES = [
    ("0e+1o", "0e+1o", "0e+1o"),
    ("0e+1o", "0e+1o+2e", "0e+1o+2e"),
    ("0e+1o+2e", "0e+1o+2e+3o", "0e+1o+2e+3o"),
]


@pytest.mark.parametrize(
    ("irreps_node", "irreps_edge", "irreps_out"), CONVOLUTION_CASES
)
def test_the_convolution_matches_legacy_in_value_and_radial_gradient(
    fp64, irreps_node, irreps_edge, irreps_out
):
    """Legacy writes each path to its own output term and lets the next linear
    mix them; the reference sums the paths reaching one irrep. So the legacy
    terms are summed per irrep before comparing, which is exact because every
    legacy path carries its own ``sqrt(2 l_out + 1)`` and no shared fan-in."""
    channels, num_nodes, num_edges = 3, 5, 12
    legacy_node = o3.Irreps(
        "+".join(f"{channels}x{ir}" for _, ir in o3.Irreps(irreps_node))
    )
    legacy_target = o3.Irreps(
        "+".join(f"{channels}x{ir}" for _, ir in o3.Irreps(irreps_out))
    )
    irreps_mid, instructions = tp_out_irreps_with_instructions(
        legacy_node, o3.Irreps(irreps_edge), legacy_target
    )
    legacy = TensorProduct(
        legacy_node,
        o3.Irreps(irreps_edge),
        irreps_mid,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
    )
    reference = BACKEND.make_channelwise_tp_conv(
        ChannelwiseTPConvDescriptor(
            irreps_node=irreps_node, irreps_edge=irreps_edge, irreps_out=irreps_out
        )
    )

    # Which reference path each legacy instruction is: the reference walks
    # output irrep, then node term, then edge term.
    out_terms = [ir for _, ir in o3.Irreps(irreps_out)]
    reference_paths = [
        (out_ir, i_node, i_edge)
        for out_ir in out_terms
        for i_node, (_, node_ir) in enumerate(o3.Irreps(irreps_node))
        for i_edge, (_, edge_ir) in enumerate(o3.Irreps(irreps_edge))
        if out_ir in node_ir * edge_ir
    ]
    assert reference.num_paths == len(reference_paths)
    path_of_instruction = [
        reference_paths.index((irreps_mid[ins.i_out].ir, ins.i_in1, ins.i_in2))
        for ins in legacy.instructions
    ]

    generator = _generator(11)
    node_features = torch.randn(
        num_nodes, channels, o3.Irreps(irreps_node).dim, generator=generator
    )
    vectors = torch.randn(num_edges, 3, generator=generator)
    edge_attributes = o3.spherical_harmonics(
        o3.Irreps(irreps_edge), vectors, normalize=True, normalization="component"
    )
    radial = torch.randn(num_edges, reference.num_paths, channels, generator=generator)
    radial.requires_grad_(True)
    sender = torch.randint(0, num_nodes, (num_edges,), generator=generator)
    receiver = torch.randint(0, num_nodes, (num_edges,), generator=generator)

    reference_out = reference(
        node_features, edge_attributes, radial, sender, receiver, num_nodes
    )

    legacy_radial = torch.cat(
        [radial[:, path, :] for path in path_of_instruction], dim=-1
    )
    messages = legacy(
        _per_channel_to_legacy(node_features, irreps_node)[sender],
        edge_attributes,
        legacy_radial,
    )
    legacy_nodes = scatter_sum(messages, receiver, dim=0, dim_size=num_nodes)

    # Sum legacy's output terms into the reference's one slot per irrep.
    legacy_out = torch.zeros_like(reference_out)
    start = 0
    for multiplicity, irrep in irreps_mid:
        block = legacy_nodes[:, start : start + multiplicity * irrep.dim]
        start += multiplicity * irrep.dim
        slot = out_terms.index(irrep)
        piece = list(_slices(irreps_out))[slot][0]
        legacy_out[:, :, piece] += block.reshape(num_nodes, multiplicity, irrep.dim)
    assert_parity(reference_out, legacy_out, "convolution forward")

    cotangent = torch.randn(reference_out.shape, generator=_generator(12))
    (reference_gradient,) = torch.autograd.grad(
        (reference_out * cotangent).sum(), radial
    )
    (legacy_gradient,) = torch.autograd.grad((legacy_out * cotangent).sum(), radial)
    assert_parity(reference_gradient, legacy_gradient, "convolution d radial")


def test_a_scalar_node_passes_the_edge_harmonics_through(fp64):
    """The audited case, written out: a scalar node of one against edge
    harmonics with radial weights of one gives back the harmonics themselves,
    ``[1, sqrt(3) x, sqrt(3) y, sqrt(3) z]`` on a unit edge. Without the
    ``sqrt(2 l_out + 1)`` path weight the ``1o`` block came out ``sqrt(3)``
    too small."""
    reference = BACKEND.make_channelwise_tp_conv(
        ChannelwiseTPConvDescriptor(
            irreps_node="0e+1o", irreps_edge="0e+1o", irreps_out="0e+1o"
        )
    )
    node_features = torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]])
    direction = torch.tensor([[0.3, -0.5, 0.8]])
    direction = direction / direction.norm()
    harmonics = torch.cat([torch.ones(1, 1), math.sqrt(3.0) * direction], dim=-1)
    radial = torch.ones(1, reference.num_paths, 1)
    out = reference(
        node_features, harmonics, radial, torch.tensor([0]), torch.tensor([1]), 2
    )
    assert_parity(out[1, 0], harmonics[0], "scalar node through the convolution")


# ---------------------------------------------------------------------------
# The symmetric contraction
# ---------------------------------------------------------------------------


def _legacy_blocks(contraction) -> list[tuple[torch.Tensor, torch.nn.Parameter, int]]:
    """``(coupling matrix, weight, body order)`` for every legacy block."""
    blocks = [
        (
            contraction.U_tensors(contraction.correlation),
            contraction.weights_max,
            contraction.correlation,
        )
    ]
    for position, weight in enumerate(contraction.weights):
        order = contraction.correlation - 1 - position
        blocks.append((contraction.U_tensors(order), weight, order))
    return blocks


def _symmetric_part(tensor: np.ndarray, order: int) -> np.ndarray:
    """Average over permutations of the ``order`` input axes after axis 0."""
    total = np.zeros_like(tensor)
    permutations = list(itertools.permutations(range(1, order + 1)))
    for permutation in permutations:
        total += np.transpose(tensor, (0, *permutation))
    return total / len(permutations)


def _legacy_to_reference_map(
    coupling: torch.Tensor, basis: torch.Tensor, order: int
) -> np.ndarray:
    """``M[p, q]`` with ``sum_q M[p, q] basis_q = sym(coupling_p)``.

    Legacy contracts its coupling matrices with a symmetric power of the
    features, so only their symmetric part acts. The reference basis is
    symmetric already, so a legacy weight ``w_p`` is the reference weight
    ``sum_p M[p, q] w_p``.
    """
    legacy = coupling.detach().numpy()
    if legacy.ndim == order + 1:  # a scalar output carries no output axis
        legacy = legacy[None]
    legacy = np.moveaxis(legacy, -1, 0)
    targets = np.stack([_symmetric_part(path, order).reshape(-1) for path in legacy])
    flat_basis = basis.detach().numpy().reshape(basis.shape[0], -1)
    solution, residual, *_ = np.linalg.lstsq(flat_basis.T, targets.T, rcond=None)
    np.testing.assert_allclose(
        flat_basis.T @ solution, targets.T, atol=CLOSED_FORM.atol
    )
    return solution.T


CONTRACTION_CASES = [
    ("0e+1o", "0e", 2),
    ("0e+1o", "0e+1o", 3),
    ("0e+1o+2e", "0e+1o+2e", 3),
]


@pytest.mark.parametrize("basis", ["reduced", "full"])
@pytest.mark.parametrize(("irreps_in", "irreps_out", "correlation"), CONTRACTION_CASES)
def test_the_contraction_matches_legacy_in_value_and_weight_gradient(
    fp64, irreps_in, irreps_out, correlation, basis
):
    channels, num_elements, num_nodes = 4, 3, 8
    torch.manual_seed(13)
    legacy = SymmetricContraction(
        irreps_in=o3.Irreps(
            "+".join(f"{channels}x{ir}" for _, ir in o3.Irreps(irreps_in))
        ),
        irreps_out=o3.Irreps(
            "+".join(f"{channels}x{ir}" for _, ir in o3.Irreps(irreps_out))
        ),
        correlation=correlation,
        num_elements=num_elements,
        use_reduced_cg=False,
    )
    reference = BACKEND.make_symmetric_contraction(
        SymmetricContractionDescriptor(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            num_features=channels,
            basis=basis,
        )
    )

    # Carry legacy's weights into the reference through the map, block by block.
    maps = []
    with torch.no_grad():
        for position, contraction in enumerate(legacy.contractions):
            tables = list(reference.bases[position])
            for coupling, weight, order in _legacy_blocks(contraction):
                if weight.numel() == 0:
                    continue
                target = reference.weights[position * correlation + order - 1]
                mapping = torch.from_numpy(
                    _legacy_to_reference_map(coupling, tables[order - 1], order)
                )
                target.copy_(torch.einsum("pq,zpc->zqc", mapping, weight))
                maps.append((mapping, weight, target))

    generator = _generator(14)
    features = torch.randn(
        num_nodes, channels, o3.Irreps(irreps_in).dim, generator=generator
    )
    element = torch.randint(0, num_elements, (num_nodes,), generator=generator)
    one_hot = torch.nn.functional.one_hot(element, num_elements).to(features.dtype)

    reference_out = reference(features, element)
    legacy_out = legacy(features, one_hot)
    assert_parity(
        _per_channel_to_legacy(reference_out, irreps_out),
        legacy_out,
        "contraction forward",
    )

    cotangent = torch.randn(reference_out.shape, generator=_generator(15))
    (reference_out * cotangent).sum().backward()
    (legacy_out * _per_channel_to_legacy(cotangent, irreps_out)).sum().backward()
    for mapping, legacy_weight, reference_weight in maps:
        assert_parity(
            torch.einsum("pq,zqc->zpc", mapping, reference_weight.grad),
            legacy_weight.grad,
            "contraction dW",
        )


def test_the_contraction_draw_has_the_legacy_spread_per_block(fp64):
    """Legacy draws each block as ``randn / num_params``. The reference in the
    full basis has the same blocks with the same path counts, so the two
    spreads agree block by block.

    A sampling bound rather than a numerical tolerance: the seeds are fixed so
    the outcome is deterministic, a block holds at least 1536 draws so its
    sample spread sits within about 2% of the true one, and the blocks have
    1, 2 and 4 paths, so a draw that ignored the path count would be off by a
    factor of two or four rather than by a few percent.
    """
    channels, num_elements = 128, 12
    torch.manual_seed(17)
    legacy = SymmetricContraction(
        irreps_in=o3.Irreps(f"{channels}x0e+{channels}x1o"),
        irreps_out=o3.Irreps(f"{channels}x0e"),
        correlation=3,
        num_elements=num_elements,
        use_reduced_cg=False,
    )
    reference = BACKEND.make_symmetric_contraction(
        SymmetricContractionDescriptor(
            irreps_in="0e+1o",
            irreps_out="0e",
            correlation=3,
            num_elements=num_elements,
            num_features=channels,
            basis="full",
        )
    )
    reference.initialize_weights(19)

    legacy_by_order = {
        order: weight for _, weight, order in _legacy_blocks(legacy.contractions[0])
    }
    for order in range(1, 4):
        reference_weight = reference.weights[order - 1].detach()
        legacy_weight = legacy_by_order[order].detach()
        path_count = reference_weight.shape[1]
        assert legacy_weight.shape == reference_weight.shape
        assert float(reference_weight.std()) * path_count == pytest.approx(1.0, rel=0.1)
        assert float(legacy_weight.std()) * path_count == pytest.approx(1.0, rel=0.1)
