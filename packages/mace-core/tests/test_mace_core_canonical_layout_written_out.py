"""The canonical layout, written out as literals.

Path order, path selection and the tensors themselves are the on-disk format
of the symmetric-contraction weights. Every other test of the basis checks a
property (symmetry, unit norm, a label that rebuilds its tensor, a span shared
with cuequivariance) and every one of those properties survives a reordering
of the enumeration or a different choice among the dependent paths. These pins
do not: each is a value committed here, so any change to the order, the
selection, the normalization or the 3j tables fails a test instead of silently
moving every weight in every checkpoint.

The label lists and the checksums were computed once from this package and
committed. They are a regression lock, not an independent derivation: the
correctness of what they lock is argued by the cuequivariance span oracle and
by the label-rebuild test. The 3j pins at the bottom are closed forms, and do
not depend on this package.
"""

import math

import numpy as np
import pytest
from mace_core.clebsch_gordan.real_basis import real_basis_change, wigner_3j_real
from mace_core.clebsch_gordan.reduced_basis import (
    full_path_labels,
    path_labels,
    reduced_symmetric_tensor_product_basis,
)

# Three body orders over three different irreps, so both the intermediate
# irrep order and the selection among dependent paths are exercised; and a
# repeated input term, so the slice order is.
PINNED_LABELS = {
    ("0e+1o+2e", 3, "0e+1o"): {
        "0e": (
            "0:0e|0:0e|0:0e",
            "1:1o|1:0e|0:0e",
            "2:2e|2:0e|0:0e",
            "1:1o|2:1o|1:0e",
            "2:2e|2:2e|2:0e",
        ),
        "1o": (
            "0:0e|0:0e|1:1o",
            "1:1o|1:0e|1:1o",
            "2:2e|2:0e|1:1o",
            "0:0e|1:1o|2:1o",
            "1:1o|2:1o|2:1o",
        ),
    },
    ("2x0e+1o", 2, "0e+1o"): {
        "0e": ("0:0e|0:0e", "0:0e|1:0e", "1:0e|1:0e", "2:1o|2:0e"),
        "1o": ("0:0e|2:1o", "1:0e|2:1o"),
    },
}

PINNED_FULL_LABELS = {
    ("2x0e+1o", 2, "0e+1o"): {
        "0e": ("0:0e|0:0e", "0:0e|1:0e", "1:0e|0:0e", "1:0e|1:0e", "2:1o|2:0e"),
        "1o": ("0:0e|2:1o", "1:0e|2:1o", "2:1o|0:1o", "2:1o|1:1o"),
    },
}

# Two weighted sums per path, in path order. The weights are small integers
# cycling with the flat index, so a permutation of entries inside a path moves
# them as surely as a change of value does.
PINNED_CHECKSUMS = {
    ("0e+1o+2e", 3, "0e+1o"): {
        "0e": (
            (1.0, 1.0),
            (14.6666666666667, 5.33333333333333),
            (14.7173367155882, 6.45497224367903),
            (-10.7567408762035, -10.2034781259527),
            (0.296815841900803, -3.16076576984569),
        ),
        "1o": (
            (10.6666666666667, 7.0),
            (16.3978318349985, 9.5405567039991),
            (25.9383885389976, 15.6524758424985),
            (-17.0304238928014, -14.9462949285938),
            (26.2528238371022, 18.6849014197076),
        ),
    },
    ("2x0e+1o", 2, "0e+1o"): {
        "0e": (
            (1.0, 1.0),
            (5.65685424949238, 2.12132034355964),
            (7.0, 2.0),
            (8.66025403784439, 6.92820323027551),
        ),
        "1o": (
            (8.57321409974112, 6.12372435695794),
            (10.2062072615966, 7.34846922834953),
        ),
    },
}


def checksums(path: np.ndarray) -> tuple[float, float]:
    flat = path.reshape(-1)
    index = np.arange(flat.size)
    return (
        float(np.sum(flat * (1 + index % 7))),
        float(np.sum(flat * (1 + index % 5))),
    )


@pytest.mark.parametrize("case", sorted(PINNED_LABELS), ids=str)
def test_the_reduced_path_order_and_selection_are_the_written_ones(case):
    labels = path_labels(*case)
    assert {
        target: tuple(str(tree) for tree in trees) for target, trees in labels.items()
    } == PINNED_LABELS[case]


@pytest.mark.parametrize("case", sorted(PINNED_FULL_LABELS), ids=str)
def test_the_unreduced_path_order_is_the_written_one(case):
    labels = full_path_labels(*case)
    assert {
        target: tuple(str(tree) for tree in trees) for target, trees in labels.items()
    } == PINNED_FULL_LABELS[case]


@pytest.mark.parametrize("case", sorted(PINNED_CHECKSUMS), ids=str)
def test_the_tensors_are_the_written_ones(case):
    basis = reduced_symmetric_tensor_product_basis(*case)
    assert basis.keys() == PINNED_CHECKSUMS[case].keys()
    for target, expected in PINNED_CHECKSUMS[case].items():
        assert len(basis[target]) == len(expected), target
        for position, (path, sums) in enumerate(
            zip(basis[target], expected, strict=True)
        ):
            assert checksums(path) == pytest.approx(sums, abs=1e-12), (
                f"path {position} of {target!r} for {case} moved"
            )


# ---------------------------------------------------------------------------
# The real basis and the 3j tables, as closed forms
# ---------------------------------------------------------------------------

HALF = 1 / math.sqrt(2)


def test_the_change_to_the_real_basis_is_the_written_one_at_degree_one():
    """Rows are real components ``m = -1, 0, 1``; columns complex ``m``."""
    expected = np.array(
        [
            [1j * HALF, 0, 1j * HALF],
            [0, 1, 0],
            [HALF, 0, -HALF],
        ]
    )
    np.testing.assert_allclose(real_basis_change(1), expected, atol=1e-15)


def test_the_change_to_the_real_basis_is_the_written_one_at_degree_two():
    expected = np.array(
        [
            [1j * HALF, 0, 0, 0, -1j * HALF],
            [0, 1j * HALF, 0, 1j * HALF, 0],
            [0, 0, 1, 0, 0],
            [0, HALF, 0, -HALF, 0],
            [HALF, 0, 0, 0, HALF],
        ]
    )
    np.testing.assert_allclose(real_basis_change(2), expected, atol=1e-15)


def test_the_scalar_coupling_of_two_vectors_is_the_identity_over_root_three():
    np.testing.assert_allclose(
        wigner_3j_real(1, 1, 0)[:, :, 0], np.eye(3) / math.sqrt(3), atol=1e-15
    )


def test_the_vector_coupling_of_two_vectors_is_minus_levi_civita_over_root_six():
    levi_civita = np.zeros((3, 3, 3))
    for (a, b, c), sign in {
        (0, 1, 2): 1,
        (1, 2, 0): 1,
        (2, 0, 1): 1,
        (0, 2, 1): -1,
        (2, 1, 0): -1,
        (1, 0, 2): -1,
    }.items():
        levi_civita[a, b, c] = sign
    np.testing.assert_allclose(
        wigner_3j_real(1, 1, 1), -levi_civita / math.sqrt(6), atol=1e-15
    )


def test_the_rank_two_coupling_of_two_vectors_is_the_written_table():
    """Every entry of ``(1, 1, 2)``: eight of 1/sqrt(10) in magnitude, three on
    the ``m = 0`` component in the ratio -1 : 2 : -1 over sqrt(30)."""
    tenth, thirtieth = 1 / math.sqrt(10), 1 / math.sqrt(30)
    expected = np.zeros((3, 3, 5))
    for index, value in {
        (0, 0, 2): -thirtieth,
        (0, 0, 4): -tenth,
        (0, 1, 1): tenth,
        (0, 2, 0): tenth,
        (1, 0, 1): tenth,
        (1, 1, 2): 2 * thirtieth,
        (1, 2, 3): tenth,
        (2, 0, 0): tenth,
        (2, 1, 3): tenth,
        (2, 2, 2): -thirtieth,
        (2, 2, 4): tenth,
    }.items():
        expected[index] = value
    np.testing.assert_allclose(wigner_3j_real(1, 1, 2), expected, atol=1e-15)
