"""The node embedding and the radial embedding block.

The radial block's forward runs three steps in an order that is load-bearing
and invisible from outside unless asserted: the cutoff is computed on the RAW
edge lengths, then the distance transform, then the basis. With an Agnesi
transform, an envelope computed from the transformed lengths would differ by
0.87 at r = 0.9 Ang, a different model rather than a rounding difference.
"""

import pytest
import torch
from mace_torch.nn.embedding import (
    DistanceTransformKind,
    LinearNodeEmbeddingBlock,
    RadialBasisKind,
    RadialEmbedding,
    RadialEmbeddingBlock,
)
from mace_torch.nn.radial import (
    AgnesiTransform,
    BesselBasis,
    PolynomialCutoff,
    SoftTransform,
)

EMBEDDING_R_MAX = 3.0
RADIAL_BASES = ["bessel", "gaussian", "chebyshev"]
DISTANCE_TRANSFORMS = ["none", "agnesi", "soft"]


def _embedding_inputs():
    lengths = torch.tensor([[0.9], [1.7], [2.5]])
    node_atomic_numbers = torch.tensor([1, 6])
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 1]])
    return lengths, node_atomic_numbers, edge_index


def _block(
    radial_basis: RadialBasisKind = "bessel",
    distance_transform: DistanceTransformKind = "none",
    apply_cutoff: bool = True,
) -> RadialEmbeddingBlock:
    return RadialEmbeddingBlock(
        r_max=EMBEDDING_R_MAX,
        num_basis=4,
        num_polynomial_cutoff=6,
        radial_basis=radial_basis,
        distance_transform=distance_transform,
        apply_cutoff=apply_cutoff,
    )


# ---------------------------------------------------------------------------
# LinearNodeEmbeddingBlock
# ---------------------------------------------------------------------------


def test_node_embedding_is_a_row_lookup_of_the_weight(fp64):
    """One-hot in, so the output of node i is the row of its element."""
    block = LinearNodeEmbeddingBlock(num_elements=3, num_channels=5)
    assert block.weight.shape == (3, 5)
    one_hot = torch.eye(3)[[2, 0, 2, 1]]
    out = block(one_hot)
    assert out.shape == (4, 5)
    assert torch.equal(out, block.weight[[2, 0, 2, 1]])
    assert block.weight.requires_grad
    assert "num_channels=5" in repr(block)


def test_node_embedding_initialisation_scale(fp64):
    """`N(0, 1/num_elements)`: the distribution of an equivariant linear layer
    with unit-normal weights and `1/sqrt(fan_in)` normalisation."""
    torch.manual_seed(0)
    block = LinearNodeEmbeddingBlock(num_elements=16, num_channels=4096)
    assert block.weight.std().item() == pytest.approx(1 / 4.0, rel=0.05)
    assert block.weight.mean().item() == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# RadialEmbeddingBlock: --apply_cutoff and the order of the three steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radial_basis", RADIAL_BASES)
@pytest.mark.parametrize("distance_transform", DISTANCE_TRANSFORMS)
def test_apply_cutoff_true_returns_the_product_and_no_envelope(
    radial_basis, distance_transform, fp64
):
    block = _block(radial_basis=radial_basis, distance_transform=distance_transform)
    assert block.apply_cutoff is True  # the CLI default
    embedding = block(*_embedding_inputs())
    assert isinstance(embedding, RadialEmbedding)
    assert embedding.edge_cutoff is None
    assert embedding.edge_radial_features.shape == (3, 4)
    assert block.num_basis == 4


@pytest.mark.parametrize("radial_basis", RADIAL_BASES)
@pytest.mark.parametrize("distance_transform", DISTANCE_TRANSFORMS)
def test_apply_cutoff_false_defers_the_envelope_to_the_consumer(
    radial_basis, distance_transform, fp64
):
    """The product of the two returned tensors is bit-for-bit the applied
    branch's output -- that identity is the whole contract."""
    applied = _block(radial_basis, distance_transform, apply_cutoff=True)(
        *_embedding_inputs()
    )
    deferred = _block(radial_basis, distance_transform, apply_cutoff=False)(
        *_embedding_inputs()
    )
    assert deferred.edge_cutoff is not None
    assert deferred.edge_cutoff.shape == (3, 1)
    assert torch.equal(
        applied.edge_radial_features,
        deferred.edge_radial_features * deferred.edge_cutoff,
    )
    # the envelope is not all ones
    assert not torch.equal(applied.edge_radial_features, deferred.edge_radial_features)


@pytest.mark.parametrize("distance_transform", ["agnesi", "soft"])
def test_the_cutoff_is_computed_before_the_distance_transform(distance_transform, fp64):
    lengths, node_atomic_numbers, edge_index = _embedding_inputs()
    block = _block(distance_transform=distance_transform, apply_cutoff=False)
    embedding = block(lengths, node_atomic_numbers, edge_index)

    cutoff = PolynomialCutoff(r_max=EMBEDDING_R_MAX, polynomial_order=6)
    transform = {"agnesi": AgnesiTransform, "soft": SoftTransform}[distance_transform]()
    transformed = transform(lengths, node_atomic_numbers, edge_index)

    assert torch.equal(embedding.edge_cutoff, cutoff(lengths))
    assert not torch.allclose(embedding.edge_cutoff, cutoff(transformed))


def test_the_basis_sees_the_transformed_lengths(fp64):
    lengths, node_atomic_numbers, edge_index = _embedding_inputs()
    block = _block(distance_transform="agnesi", apply_cutoff=False)
    embedding = block(lengths, node_atomic_numbers, edge_index)
    transformed = AgnesiTransform()(lengths, node_atomic_numbers, edge_index)
    reference = BesselBasis(r_max=EMBEDDING_R_MAX, num_basis=4)
    assert torch.equal(embedding.edge_radial_features, reference(transformed))
    assert not torch.allclose(embedding.edge_radial_features, reference(lengths))


def test_no_transform_is_an_explicit_none(fp64):
    """Configuration is explicit: the transform slot is None, never an absent
    attribute probed with hasattr."""
    assert _block().distance_transform is None
    assert isinstance(
        _block(distance_transform="soft").distance_transform, SoftTransform
    )


def test_a_padding_edge_contributes_exactly_nothing(fp64):
    """A self-loop edge shifted by 2*r_max has length 2*r_max and embeds to
    exactly zero: by multiplication in the default branch, through an exactly
    zero envelope in the deferred one."""
    padded = torch.tensor([[2 * EMBEDDING_R_MAX]])
    node_atomic_numbers = torch.tensor([1])
    edge_index = torch.tensor([[0], [0]])
    embedding = _block()(padded, node_atomic_numbers, edge_index)
    features = embedding.edge_radial_features
    assert torch.equal(features, torch.zeros_like(features))

    deferred = _block(apply_cutoff=False)(padded, node_atomic_numbers, edge_index)
    assert torch.equal(deferred.edge_cutoff, torch.zeros_like(deferred.edge_cutoff))


def test_unknown_kinds_are_errors_naming_the_value():
    with pytest.raises(ValueError, match="radial_basis='fourier'"):
        _block(radial_basis="fourier")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="distance_transform='Agnesi'"):
        _block(distance_transform="Agnesi")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("radial_basis", RADIAL_BASES)
@pytest.mark.parametrize("distance_transform", DISTANCE_TRANSFORMS)
def test_radial_embedding_passes_gradgradcheck(radial_basis, distance_transform, fp64):
    """Force training differentiates twice through the whole block."""
    block = _block(radial_basis=radial_basis, distance_transform=distance_transform)
    _, node_atomic_numbers, edge_index = _embedding_inputs()
    lengths = torch.tensor([[0.9], [1.7], [2.5]], requires_grad=True)

    def function(lengths_):
        return block(lengths_, node_atomic_numbers, edge_index).edge_radial_features

    assert torch.autograd.gradcheck(function, (lengths,))
    assert torch.autograd.gradgradcheck(function, (lengths,))
