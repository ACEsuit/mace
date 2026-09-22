"""The reduced symmetric tensor-product basis.

The path counts are the golden that matters. On the frozen tree that number
changes with whether `cuequivariance` happens to be installed, which means the
same hyperparameters train a different network on a different host. Pinning it
here is what stops that reappearing.
"""

import itertools

import numpy as np
import pytest
from mace_core.clebsch_gordan.irreps import Irrep, Irreps, IrrepsError
from mace_core.clebsch_gordan.real_basis import real_basis_change, wigner_3j_real
from mace_core.clebsch_gordan.reduced_basis import (
    full_symmetric_tensor_product_basis,
    path_count,
    reduced_symmetric_tensor_product_basis,
)

ATOL = 1e-12

# The anchor the ticket measures on the legacy cueq-only path, summed over the
# correlation orders a model of that body order actually builds.
ANCHOR_IRREPS = "0e+1o+2e+3o"
ANCHOR_PER_ORDER = {
    "0e": (1, 4, 8),
    "1o": (1, 3, 12),
    "2e": (1, 5, 14),
}
ANCHOR_TOTALS = {"0e": 13, "1o": 16, "2e": 20}


@pytest.mark.parametrize(("target", "per_order"), sorted(ANCHOR_PER_ORDER.items()))
def test_the_path_counts_reproduce_the_measured_anchor(target, per_order):
    for correlation, expected in enumerate(per_order, start=1):
        assert path_count(ANCHOR_IRREPS, correlation, target) == expected


@pytest.mark.parametrize(("target", "total"), sorted(ANCHOR_TOTALS.items()))
def test_the_totals_over_the_body_orders_reproduce_the_anchor(target, total):
    assert sum(path_count(ANCHOR_IRREPS, nu, target) for nu in (1, 2, 3)) == total


def test_the_twenty_nine_the_defect_used_to_change():
    """The single number a host without cuequivariance silently turned into 86."""
    assert sum(path_count(ANCHOR_IRREPS, nu, "0e+1o") for nu in (1, 2, 3)) == 29


def test_every_path_is_symmetric_under_permuting_its_factors():
    """What makes the product symmetric. A basis vector that were not invariant
    would be carrying a direction the contraction can never reach."""
    basis = reduced_symmetric_tensor_product_basis(ANCHOR_IRREPS, 3, "1o")["1o"]
    for path in basis:
        for order in itertools.permutations((1, 2, 3)):
            assert np.abs(np.transpose(path, (0, *order)) - path).max() < ATOL


def test_every_path_has_unit_norm_and_a_positive_leading_entry():
    """The pinned normalization and the pinned sign, which together are half of
    the on-disk weight format."""
    for target in ("0e", "1o", "2e"):
        for path in reduced_symmetric_tensor_product_basis(ANCHOR_IRREPS, 2, target)[
            target
        ]:
            assert float(np.linalg.norm(path)) == pytest.approx(1.0, abs=ATOL)
            flat = path.reshape(-1)
            first = flat[np.flatnonzero(np.abs(flat) > 1e-9)[0]]
            assert first > 0


def test_the_paths_are_linearly_independent():
    basis = reduced_symmetric_tensor_product_basis(ANCHOR_IRREPS, 3, "0e")["0e"]
    flat = basis.reshape(basis.shape[0], -1)
    assert np.linalg.matrix_rank(flat, tol=1e-9) == basis.shape[0]


def test_the_basis_is_the_same_on_every_call():
    """No dependence on dictionary order, on a random seed, or on what is
    installed. The basis is model state, and model state cannot vary."""
    first = reduced_symmetric_tensor_product_basis("0e+1o", 2, "0e")["0e"]
    second = reduced_symmetric_tensor_product_basis("0e+1o", 2, "0e")["0e"]
    assert np.array_equal(first, second)
    assert first is not second, "a caller must not be able to mutate the cache"


def test_the_returned_shape_states_the_layout():
    width = Irreps.parse(ANCHOR_IRREPS).dimension
    basis = reduced_symmetric_tensor_product_basis(ANCHOR_IRREPS, 3, "1o")["1o"]
    assert basis.shape == (12, 3, width, width, width)


def test_an_unreachable_irrep_gives_an_empty_basis_rather_than_an_error():
    """A model may ask for an output its inputs cannot reach at a given body
    order. That is a zero-path basis, not a failure."""
    basis = reduced_symmetric_tensor_product_basis("0e", 1, "1o")["1o"]
    assert basis.shape[0] == 0


def test_float32_is_a_cast_and_not_a_different_computation():
    wide = reduced_symmetric_tensor_product_basis("0e+1o", 2, "0e", dtype="float64")
    narrow = reduced_symmetric_tensor_product_basis("0e+1o", 2, "0e", dtype="float32")
    assert narrow["0e"].dtype == np.float32
    assert np.abs(narrow["0e"].astype(np.float64) - wide["0e"]).max() < 1e-6


def test_the_errors_name_the_offending_value():
    with pytest.raises(ValueError, match="correlation"):
        reduced_symmetric_tensor_product_basis("0e", 0, "0e")
    with pytest.raises(ValueError, match="float64"):
        reduced_symmetric_tensor_product_basis("0e", 1, "0e", dtype="float16")
    with pytest.raises(IrrepsError, match="1u"):
        Irreps.parse("1u")


# ---------------------------------------------------------------------------
# The real basis underneath it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", range(6))
def test_the_change_to_the_real_basis_is_unitary(degree):
    matrix = real_basis_change(degree)
    assert np.abs(matrix @ matrix.conj().T - np.eye(2 * degree + 1)).max() < ATOL


@pytest.mark.parametrize(
    ("l1", "l2", "l3"),
    [(0, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2), (2, 2, 2), (2, 3, 4), (3, 3, 3)],
)
def test_the_real_3j_is_real_and_keeps_the_norm(l1, l2, l3):
    """A unitary change of basis cannot move the sum of the squares, so this is
    the property that catches a wrong phase convention."""
    table = wigner_3j_real(l1, l2, l3)
    assert table.dtype == np.float64
    assert float((table**2).sum()) == pytest.approx(1.0, abs=ATOL)


def test_the_irrep_order_is_the_documented_one():
    """Ascending degree, even before odd. This order is the weight order, so it
    is pinned rather than left to whatever sorted() happens to do."""
    assert sorted([Irrep(1, -1), Irrep(0, 1), Irrep(2, 1), Irrep(1, 1)]) == [
        Irrep(0, 1),
        Irrep(1, 1),
        Irrep(1, -1),
        Irrep(2, 1),
    ]


# ---------------------------------------------------------------------------
# Output irreps the symmetric product does not carry
# ---------------------------------------------------------------------------


def test_an_irrep_carried_only_by_the_antisymmetric_part_has_no_paths():
    """`1o x 1o -> 1e` is the cross product, and it is antisymmetric.

    The paths to it exist, which is why the unreduced basis has one; they
    cancel under symmetrization. That is a basis of no paths rather than a
    failure, and the difference matters because the two are reached by
    different code: an unreachable irrep yields nothing to enumerate, and this
    one enumerates something that then vanishes.
    """
    reduced = reduced_symmetric_tensor_product_basis("1o", 2, "1e")["1e"]
    unreduced = full_symmetric_tensor_product_basis("1o", 2, "1e")["1e"]
    assert reduced.shape == (0, 3, 3, 3)
    assert unreduced.shape[0] == 1
    assert path_count("1o", 2, "1e") == 0


def test_an_unreachable_irrep_answers_the_same_way():
    """The other route to no paths, so the two agree on the answer's shape."""
    assert reduced_symmetric_tensor_product_basis("1o", 1, "2e")["2e"].shape == (
        0,
        5,
        3,
    )
    assert path_count("1o", 1, "2e") == 0


def test_a_mixed_request_keeps_the_irreps_that_do_have_paths():
    """The case the crash was reached through: one slot of a loop over irreps.

    A caller asking for several output irreps at once gets a zero-path entry
    for the ones the symmetric product does not carry, and real bases for the
    rest, rather than an exception that says nothing about which irrep caused
    it.
    """
    basis = reduced_symmetric_tensor_product_basis("1o", 2, "0e+1e+2e")
    assert basis["1e"].shape[0] == 0
    assert basis["0e"].shape[0] == 1
    assert basis["2e"].shape[0] == 1
