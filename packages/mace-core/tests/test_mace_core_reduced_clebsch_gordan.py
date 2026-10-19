"""The reduced symmetric tensor-product basis.

The path counts are the golden that matters. On the frozen tree that number
changes with whether `cuequivariance` happens to be installed, which means the
same hyperparameters train a different network on a different host. Pinning it
here is what stops that reappearing.
"""

import itertools
import re

import numpy as np
import pytest
from mace_core.clebsch_gordan import reduced_basis
from mace_core.clebsch_gordan.irreps import Irrep, Irreps, IrrepsError
from mace_core.clebsch_gordan.real_basis import real_basis_change, wigner_3j_real
from mace_core.clebsch_gordan.reduced_basis import (
    full_path_labels,
    full_symmetric_tensor_product_basis,
    path_count,
    path_labels,
    reduced_symmetric_tensor_product_basis,
)

ATOL = 1e-12

# The anchor the ticket measures on the legacy cueq-only path, per single body
# order. A model of correlation 3 carries the sum over the three, 13, 16 and 20.
ANCHOR_IRREPS = "0e+1o+2e+3o"
ANCHOR_PER_ORDER = {
    "0e": (1, 4, 8),
    "1o": (1, 3, 12),
    "2e": (1, 5, 14),
}


@pytest.mark.parametrize(("target", "per_order"), sorted(ANCHOR_PER_ORDER.items()))
def test_the_path_counts_reproduce_the_measured_anchor(target, per_order):
    for correlation, expected in enumerate(per_order, start=1):
        assert path_count(ANCHOR_IRREPS, correlation, target) == expected


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
    """No dependence on dictionary order or on a random seed. The basis is model
    state, and model state cannot vary. The cache is cleared between the two
    calls, or the second would be served the first one's array and the
    comparison could not fail."""
    first = reduced_symmetric_tensor_product_basis(ANCHOR_IRREPS, 3, "1o")["1o"]
    reduced_basis._basis_for.cache_clear()
    second = reduced_symmetric_tensor_product_basis(ANCHOR_IRREPS, 3, "1o")["1o"]
    assert np.array_equal(first, second)


def test_a_caller_cannot_mutate_the_cached_basis():
    first = reduced_symmetric_tensor_product_basis("0e+1o", 2, "0e")["0e"]
    first[...] = 0.0
    second = reduced_symmetric_tensor_product_basis("0e+1o", 2, "0e")["0e"]
    assert np.abs(second).max() > 0


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


# Every public entry point that takes the same arguments validates them the
# same way, so none of them can be the one that loops or truncates.
BUILDERS = [
    reduced_symmetric_tensor_product_basis,
    full_symmetric_tensor_product_basis,
    path_labels,
    full_path_labels,
    path_count,
]
BASES = [reduced_symmetric_tensor_product_basis, full_symmetric_tensor_product_basis]


@pytest.mark.parametrize("build", BUILDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("correlation", [0, -1, 1.0, True])
def test_every_builder_refuses_a_correlation_that_is_not_a_positive_integer(
    build, correlation
):
    """The label functions used to recurse without end on 0 and below."""
    with pytest.raises(ValueError, match=re.escape(f"got {correlation!r}")):
        build("0e+1o", correlation, "0e")


@pytest.mark.parametrize("build", BASES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("dtype", ["float16", "int8", "bool", "float"])
def test_every_basis_refuses_an_unsupported_dtype(build, dtype):
    """``int8`` would truncate every entry to zero, so it cannot pass silently."""
    with pytest.raises(ValueError, match=re.escape(repr(dtype))):
        build("0e+1o", 2, "0e", dtype=dtype)


@pytest.mark.parametrize("build", BUILDERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize(
    ("keep_ir", "complaint"),
    [("2x0e", "multiplicity of 2"), ("0e+0e", "more than once")],
)
def test_every_builder_refuses_a_keep_ir_that_would_collapse(build, keep_ir, complaint):
    """Both used to come back as a single ``0e`` entry, without a word."""
    with pytest.raises(ValueError, match=complaint):
        build("0e+1o", 2, keep_ir)


@pytest.mark.parametrize("build", BUILDERS, ids=lambda f: f.__name__)
def test_every_builder_refuses_an_input_term_with_no_copies(build):
    with pytest.raises(IrrepsError, match="'0x0e'"):
        build("0x0e+1o", 2, "0e")


def test_a_malformed_declaration_names_the_term():
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


@pytest.mark.parametrize(
    ("degree", "parity"), [(1.5, 1), (True, 1), (-1, 1), (1, 0), (1, True), (1, 1.0)]
)
def test_an_irrep_refuses_anything_but_integer_degree_and_signed_parity(degree, parity):
    """The constructor, not only the parser, is a construction path."""
    with pytest.raises(IrrepsError):
        Irrep(degree, parity)


@pytest.mark.parametrize(
    "terms",
    [
        (),
        ((0, Irrep(0, 1)),),
        ((-1, Irrep(0, 1)),),
        ((1.0, Irrep(0, 1)),),
        ((1, "0e"),),
    ],
)
def test_irreps_refuses_terms_the_parser_would_refuse(terms):
    with pytest.raises(IrrepsError):
        Irreps(terms)


@pytest.mark.parametrize("text", ["1x0e", " 0e", "0e+1o", "0e\n", "2e3"])
def test_a_single_irrep_is_parsed_strictly(text):
    """Where one irrep is expected, a declaration is a caller's mistake."""
    with pytest.raises(IrrepsError, match=re.escape(repr(text))):
        Irrep.parse(text)


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
