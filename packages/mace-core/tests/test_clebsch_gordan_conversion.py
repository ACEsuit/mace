"""The two boundaries: between the bases, and between the layouts.

The basis conversion is tested on values and never on weights, because it is a
projection and cannot be bit-exact in weight space. Asserting weight equality
would be asserting something false; asserting the function is what the
conversion actually promises.
"""

import numpy as np
import pytest
from mace_core.clebsch_gordan.conversion import (
    from_canonical,
    full_to_reduced,
    ir_mul_to_mul_ir,
    mul_ir_to_ir_mul,
    reduced_to_full,
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
