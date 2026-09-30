"""The two boundaries: between the bases, and between the layouts.

The basis conversion is tested on values and never on weights, because it is a
projection and cannot be bit-exact in weight space. Asserting weight equality
would be asserting something false; asserting the function is what the
conversion actually promises.
"""

import re

import numpy as np
import pytest
from mace_core.clebsch_gordan import IrrepsError
from mace_core.clebsch_gordan.conversion import (
    FLOAT32_PROJECTION_TOLERANCE,
    PROJECTION_TOLERANCE,
    BasisConversionError,
    _projection,
    from_canonical,
    full_to_reduced,
    ir_mul_to_mul_ir,
    mul_ir_to_ir_mul,
    reduced_to_full,
    source_tolerance,
    to_canonical,
)
from mace_core.clebsch_gordan.reduced_basis import (
    full_symmetric_tensor_product_basis,
    reduced_symmetric_tensor_product_basis,
)

GRID = [("0e+1o", 2, "0e"), ("0e+1o+2e", 3, "1o"), ("0e+1o+2e+3o", 3, "2e")]
ATOL = 1e-11


def contract(basis, inputs, weights, correlation):
    """What the model computes: the basis against the same feature vector on
    every input axis, then against the weights."""
    axes = "abcdefg"[:correlation]
    subscript = f"po{axes}," + ",".join(f"n{axis}" for axis in axes) + ",zpm->zomn"
    return np.einsum(subscript, basis, *([inputs] * correlation), weights)


@pytest.mark.parametrize(("irreps", "correlation", "target"), GRID)
def test_full_to_reduced_preserves_the_function(irreps, correlation, target):
    rng = np.random.default_rng(0)
    full = full_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    reduced = reduced_symmetric_tensor_product_basis(irreps, correlation, target)[
        target
    ]
    weights = rng.normal(size=(2, full.shape[0], 4))
    carried = full_to_reduced(weights, irreps, correlation, target)
    assert carried.shape == (2, reduced.shape[0], 4)

    inputs = rng.normal(size=(5, full.shape[2]))
    before = contract(full, inputs, weights, correlation)
    after = contract(reduced, inputs, carried, correlation)
    assert np.abs(before - after).max() < ATOL


@pytest.mark.parametrize(("irreps", "correlation", "target"), GRID)
def test_the_round_trip_preserves_the_function_and_returns_the_smallest_preimage(
    irreps, correlation, target
):
    rng = np.random.default_rng(1)
    full = full_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    weights = rng.normal(size=(2, full.shape[0], 4))
    carried = full_to_reduced(weights, irreps, correlation, target)
    recovered = reduced_to_full(carried, irreps, correlation, target)

    inputs = rng.normal(size=(5, full.shape[2]))
    assert (
        np.abs(
            contract(full, inputs, weights, correlation)
            - contract(full, inputs, recovered, correlation)
        ).max()
        < ATOL
    )
    assert np.linalg.norm(recovered) <= np.linalg.norm(weights) + 1e-9


def test_the_conversion_is_not_bit_exact_in_weight_space():
    """Stated as a test so nobody later 'fixes' it by asserting equality. The
    map is a projection: the full basis carries directions the reduced one does
    not, and those directions are gauge."""
    rng = np.random.default_rng(2)
    irreps, correlation, target = "0e+1o+2e", 3, "1o"
    full = full_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    weights = rng.normal(size=(2, full.shape[0], 4))
    recovered = reduced_to_full(
        full_to_reduced(weights, irreps, correlation, target),
        irreps,
        correlation,
        target,
    )
    assert np.abs(recovered - weights).max() > 1e-3


def test_the_path_axis_length_is_checked_with_both_numbers_in_the_message():
    with pytest.raises(ValueError, match="path axis has length 7"):
        full_to_reduced(np.zeros((2, 7, 4)), "0e+1o", 2, "0e")


@pytest.mark.parametrize("convert", [full_to_reduced, reduced_to_full])
@pytest.mark.parametrize("target", ["1x0e", " 0e", "0e+1o"])
def test_a_target_that_is_not_one_irrep_is_refused_by_name(convert, target):
    """Each of these used to surface as a bare ``KeyError`` from a dict."""
    with pytest.raises(IrrepsError, match=re.escape(repr(target))):
        convert(np.zeros((2, 3, 4)), "0e+1o", 2, target)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_conversion_hands_back_the_dtype_it_was_given(dtype):
    """Computed at float64 either way; float32 in is float32 out."""
    irreps, correlation, target = "0e+1o+2e", 3, "1o"
    full = full_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    weights = np.random.default_rng(3).normal(size=(2, full.shape[0], 4))
    carried = full_to_reduced(weights.astype(dtype), irreps, correlation, target)
    assert carried.dtype == dtype
    assert reduced_to_full(carried, irreps, correlation, target).dtype == dtype
    exact = full_to_reduced(weights, irreps, correlation, target)
    assert np.abs(carried - exact).max() < 1e-5


# ---------------------------------------------------------------------------
# Layout, which costs nothing
# ---------------------------------------------------------------------------


def test_canonical_join_and_split_round_trip():
    parts = [np.arange(2 * n * 3).reshape(2, n, 3).astype(float) for n in (1, 4, 8)]
    flat = to_canonical(parts)
    assert flat.shape == (2, 13, 3)
    again = from_canonical(flat, [1, 4, 8])
    for before, after in zip(parts, again, strict=True):
        assert np.array_equal(before, after)


def test_a_split_that_does_not_add_up_says_both_totals():
    with pytest.raises(ValueError, match="add up to 12"):
        from_canonical(np.zeros((2, 13, 3)), [1, 4, 7])


def test_the_layout_change_is_its_own_inverse_both_ways():
    values = np.arange(2 * 6).reshape(2, 6).astype(float)
    assert np.array_equal(
        ir_mul_to_mul_ir(mul_ir_to_ir_mul(values, "2x1o"), "2x1o"), values
    )


def test_the_layout_change_actually_moves_the_values():
    """`2x1o` in mul_ir is two blocks of three; in ir_mul it is three blocks of
    two. A no-op here would mean a backend silently reading scrambled weights."""
    values = np.array([[1.0, 2, 3, 4, 5, 6]])
    assert np.array_equal(
        mul_ir_to_ir_mul(values, "2x1o"), np.array([[1.0, 4, 2, 5, 3, 6]])
    )


def test_the_layout_change_refuses_a_declaration_it_cannot_reshape():
    with pytest.raises(ValueError, match="separately"):
        mul_ir_to_ir_mul(np.zeros((1, 4)), "1x0e+1x1o")


# ---------------------------------------------------------------------------
# From an artifact's own full basis
# ---------------------------------------------------------------------------

#: The coupling every published foundation model uses, one channel of it.
PUBLISHED = "1x0e+1x1o+1x2e+1x3o"


def another_gauge(full, seed):
    """A different basis of the same space: the paths mixed by an invertible
    matrix, the way a second implementation orders and rotates its own."""
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(full.shape[0], full.shape[0]))
    return np.einsum("qp,p...->q...", mixing, full)


@pytest.mark.parametrize(
    ("target", "dimensions"), [("0e", (1, 4, 8)), ("1o", (1, 3, 12))]
)
def test_the_reduced_basis_has_the_published_dimensions(target, dimensions):
    """The compatibility target: every published artifact's contraction lands
    on exactly these, per body order."""
    got = tuple(
        reduced_symmetric_tensor_product_basis(PUBLISHED, order, target)[target].shape[
            0
        ]
        for order in (1, 2, 3)
    )
    assert got == dimensions


@pytest.mark.parametrize(("target", "total"), [("0e", 28), ("1o", 58)])
def test_the_full_basis_has_the_published_path_counts(target, total):
    assert (
        sum(
            full_symmetric_tensor_product_basis(PUBLISHED, order, target)[target].shape[
                0
            ]
            for order in (1, 2, 3)
        )
        == total
    )


@pytest.mark.parametrize(("irreps", "correlation", "target"), GRID)
def test_weights_from_another_basis_keep_their_function(irreps, correlation, target):
    rng = np.random.default_rng(3)
    full = full_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    source = another_gauge(full, seed=4)
    reduced = reduced_symmetric_tensor_product_basis(irreps, correlation, target)[
        target
    ]
    weights = rng.normal(size=(2, source.shape[0], 4))
    carried = full_to_reduced(weights, irreps, correlation, target, source=source)

    inputs = rng.normal(size=(5, full.shape[2]))
    before = contract(source, inputs, weights, correlation)
    after = contract(reduced, inputs, carried, correlation)
    assert np.abs(before - after).max() < ATOL * np.abs(before).max()


def test_the_basis_is_an_argument_not_an_assumption():
    """Weights from another basis converted as if they were this package's
    compute something else. That is why the artifact has to carry its own."""
    irreps, correlation, target = "0e+1o+2e", 3, "0e"
    rng = np.random.default_rng(5)
    full = full_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    source = another_gauge(full, seed=6)
    reduced = reduced_symmetric_tensor_product_basis(irreps, correlation, target)[
        target
    ]
    weights = rng.normal(size=(1, source.shape[0], 2))
    assumed = full_to_reduced(weights, irreps, correlation, target)
    inputs = rng.normal(size=(3, full.shape[2]))
    before = contract(source, inputs, weights, correlation)
    assert np.abs(before - contract(reduced, inputs, assumed, correlation)).max() > 1e-3


@pytest.mark.parametrize("target", ["0e", "1o"])
def test_the_published_projection_is_exact_to_the_tolerance(target):
    """At the published coupling and correlation three, on a basis in another
    gauge. The call refuses above the tolerance, so succeeding is the check."""
    full = full_symmetric_tensor_product_basis(PUBLISHED, 3, target)[target]
    matrix = _projection(PUBLISHED, 3, target, source=another_gauge(full, seed=7))
    assert matrix.shape == (full.shape[0], {"0e": 8, "1o": 12}[target])
    assert PROJECTION_TOLERANCE <= 1e-14


def test_the_unsymmetrized_fit_is_the_trap():
    """Treating the input slots as independent sees every full path as its own
    direction: rank 23 for a scalar at body order three, where only 8 compute
    anything. The symmetrized paths have exactly the reduced rank."""
    from mace_core.clebsch_gordan.reduced_basis import symmetrize

    full = full_symmetric_tensor_product_basis(PUBLISHED, 3, "0e")["0e"]
    raw = np.linalg.matrix_rank(full.reshape(full.shape[0], -1))
    symmetric = np.linalg.matrix_rank(
        np.stack([symmetrize(path, 3).reshape(-1) for path in full])
    )
    assert (raw, symmetric) == (23, 8)


def test_a_source_that_spans_something_else_is_refused():
    """A path that is not equivariant has a symmetric part no reduced basis
    reproduces. Converting it would return weights of the right shape that
    compute a different function, so it is refused instead."""
    irreps, correlation, target = "0e+1o", 2, "0e"
    full = full_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    stray = np.random.default_rng(8).normal(size=(1, *full.shape[1:]))
    with pytest.raises(BasisConversionError, match="relative residual"):
        full_to_reduced(
            np.ones((1, full.shape[0] + 1, 1)),
            irreps,
            correlation,
            target,
            source=np.concatenate([full, stray]),
        )


def test_a_source_in_the_wrong_layout_is_refused():
    irreps, correlation, target = "0e+1o", 2, "1o"
    full = full_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    with pytest.raises(BasisConversionError, match="path axis first"):
        full_to_reduced(
            np.ones((1, full.shape[0], 1)),
            irreps,
            correlation,
            target,
            source=np.moveaxis(full, 0, -1),
        )


def test_a_basis_held_at_float32_is_judged_at_float32():
    """A model built in float32 holds its basis rounded, off the exact span by
    that rounding. The float64 bound refuses it; the float32 one, chosen from
    the values themselves, carries it with the function kept to float32."""
    irreps, correlation, target = "0e+1o+2e", 3, "1o"
    full = full_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    rounded = another_gauge(full, seed=9).astype(np.float32).astype(np.float64)
    weights = np.random.default_rng(10).normal(size=(1, rounded.shape[0], 2))
    assert source_tolerance(rounded) == FLOAT32_PROJECTION_TOLERANCE
    with pytest.raises(BasisConversionError, match="relative residual"):
        full_to_reduced(
            weights,
            irreps,
            correlation,
            target,
            source=rounded,
            tolerance=PROJECTION_TOLERANCE,
        )
    carried = full_to_reduced(weights, irreps, correlation, target, source=rounded)
    reduced = reduced_symmetric_tensor_product_basis(irreps, correlation, target)[
        target
    ]
    inputs = np.random.default_rng(11).normal(size=(3, full.shape[2]))
    before = contract(rounded, inputs, weights, correlation)
    after = contract(reduced, inputs, carried, correlation)
    assert np.abs(before - after).max() < 1e-6 * np.abs(before).max()


def test_a_float32_basis_of_another_space_is_still_refused():
    """The looser bound is for rounding: a stray direction rounded to float32
    is off by order one and refused as before."""
    irreps, correlation, target = "0e+1o", 2, "0e"
    full = full_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    stray = np.random.default_rng(12).normal(size=(1, *full.shape[1:]))
    source = np.concatenate([full, stray]).astype(np.float32).astype(np.float64)
    assert source_tolerance(source) == FLOAT32_PROJECTION_TOLERANCE
    with pytest.raises(BasisConversionError, match="relative residual"):
        full_to_reduced(
            np.ones((1, source.shape[0], 1)), irreps, correlation, target, source=source
        )


def test_a_basis_held_at_float64_keeps_the_float64_bound():
    full = full_symmetric_tensor_product_basis("0e+1o+2e", 3, "1o")["1o"]
    assert source_tolerance(another_gauge(full, seed=13)) == PROJECTION_TOLERANCE
