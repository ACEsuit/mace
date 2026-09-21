"""The canonical weight layout, and the map from it to a backend's own order.

Two things are pinned here. The flat path order of the symmetric-contraction
weights, which is the on-disk format, and the structure of the conversion a
backend applies once at build time.

The conversion is **not** a scaled permutation in general, which is the point
the golden exists to hold. Order and normalization are pinned, but which of the
linearly dependent coupling paths survives the reduction is a free choice, so
two implementations resolving it differently are related by a change of basis
inside the degenerate blocks. Asserting a permutation would pass on the easy
configurations and quietly misread weights on the rest.
"""

import numpy as np
import pytest
from mace_core.clebsch_gordan import reduced_symmetric_tensor_product_basis
from mace_core.kernels import (
    ReorderError,
    contraction_path_labels,
    contraction_path_order,
    derive_reorder,
)
from mace_core.kernels.descriptors import SymmetricContractionDescriptor

ATOL = 1e-12

GRID = [
    ("0e+1o", "0e+1o", 3),
    ("0e+1o+2e", "0e+1o", 3),
    ("0e+1o+2e+3o", "0e+1o+2e", 3),
    ("0e+1o+2e", "0e+1o", 4),
]


def test_the_flat_order_is_irrep_major_and_body_order_ascending():
    """The order of the joined pieces is the file format, so it is written out."""
    assert contraction_path_order("0e+1o", 3) == (
        ("0e", 1),
        ("0e", 2),
        ("0e", 3),
        ("1o", 1),
        ("1o", 2),
        ("1o", 3),
    )


@pytest.mark.parametrize(("irreps_in", "irreps_out", "correlation"), GRID)
def test_one_label_per_weight_on_the_path_axis(irreps_in, irreps_out, correlation):
    descriptor = SymmetricContractionDescriptor(
        irreps_in=irreps_in, irreps_out=irreps_out, correlation=correlation
    )
    labels = contraction_path_labels(irreps_in, irreps_out, correlation)
    assert len(labels) == descriptor.path_count


@pytest.mark.parametrize(("irreps_in", "irreps_out", "correlation"), GRID)
def test_the_labels_are_unique_across_the_whole_flat_array(
    irreps_in, irreps_out, correlation
):
    """A label has to identify a path in the file, not only within its slot."""
    labels = contraction_path_labels(irreps_in, irreps_out, correlation)
    assert len(set(labels)) == len(labels)


def test_the_full_basis_is_labelled_too_and_is_wider():
    reduced = contraction_path_labels("0e+1o+2e+3o", "0e+1o", 3, basis="reduced")
    full = contraction_path_labels("0e+1o+2e+3o", "0e+1o", 3, basis="full")
    assert len(reduced) == 29
    assert len(full) == 86
    assert set(reduced) <= set(full)


def test_an_unknown_basis_name_is_refused():
    with pytest.raises(ValueError, match="'reduced' or 'full'"):
        contraction_path_labels("0e+1o", "0e", 2, basis="cueq")


def _basis(irreps_in, correlation, target):
    basis = reduced_symmetric_tensor_product_basis(irreps_in, correlation, target)
    return basis[target]


def test_a_basis_against_itself_is_the_identity():
    basis = _basis("0e+1o+2e", 3, "1o")
    reorder = derive_reorder(basis, basis)
    assert reorder.is_scaled_permutation
    assert reorder.block_sizes() == {1: basis.shape[0]}
    assert reorder.residual < ATOL


def test_a_permuted_and_rescaled_basis_comes_back_as_1x1_blocks():
    basis = _basis("0e+1o+2e", 3, "1o")
    order = np.array([3, 0, 4, 1, 2])
    scales = np.array([2.0, -1.5, 0.25, 1.0, -3.0])
    reorder = derive_reorder(basis, basis[order] * scales[:, None, None, None, None])
    assert reorder.is_scaled_permutation
    assert reorder.path_count == basis.shape[0]


@pytest.mark.parametrize(("irreps_in", "irreps_out", "correlation"), GRID)
def test_the_map_carries_the_weights_and_comes_back(irreps_in, irreps_out, correlation):
    """The property that matters: the same function, over either basis.

    A rotation inside the span is the hardest case the derivation has to
    handle, and it is the case a scaled permutation cannot express. Mixing the
    basis by a full orthogonal matrix makes every block as large as its
    degenerate subspace allows.
    """
    generator = np.random.default_rng(20250921)
    for target, order in contraction_path_order(irreps_out, correlation):
        basis = _basis(irreps_in, order, target)
        if basis.shape[0] < 2:
            continue
        mixed = np.linalg.qr(generator.normal(size=(basis.shape[0],) * 2))[0]
        other = np.tensordot(mixed, basis, axes=1)
        reorder = derive_reorder(basis, other)

        weights = generator.normal(size=(3, basis.shape[0], 4))
        carried = reorder.apply(weights)
        np.testing.assert_allclose(
            np.einsum("zam,a...->zm...", weights, basis),
            np.einsum("zam,a...->zm...", carried, other),
            atol=ATOL,
        )
        np.testing.assert_allclose(reorder.inverse().apply(carried), weights, atol=ATOL)


def test_two_bases_of_different_spaces_are_refused_rather_than_approximated():
    """The failure that matters: a backend computing something else.

    A least-squares fit always returns a matrix. Without the residual check it
    would be accepted, and the checkpoint would load into a model that
    evaluates a different function.
    """
    basis = _basis("0e+1o+2e", 3, "1o")
    broken = basis.copy()
    broken[0] = np.roll(broken[0], 1)
    with pytest.raises(ReorderError, match="span different spaces"):
        derive_reorder(basis, broken)


def test_a_different_path_count_is_refused():
    with pytest.raises(ReorderError, match="would have to drop or invent"):
        derive_reorder(_basis("0e+1o+2e", 3, "1o"), _basis("0e+1o+2e", 3, "1o")[:-1])


def test_an_empty_slot_maps_to_an_empty_map():
    """An output irrep unreachable at this body order still has to be handled."""
    empty = _basis("0e+1o", 1, "2e")
    assert empty.shape[0] == 0
    reorder = derive_reorder(empty, empty)
    assert reorder.blocks == ()
    assert reorder.path_count == 0
    assert reorder.max_block_size == 0


def test_applying_to_the_wrong_width_names_both_numbers():
    basis = _basis("0e+1o+2e", 3, "1o")
    reorder = derive_reorder(basis, basis)
    with pytest.raises(ReorderError, match="this map covers"):
        reorder.apply(np.zeros((2, basis.shape[0] + 1, 4)))
