"""The real basis of `mace_core.clebsch_gordan` against e3nn's, in one process.

`mace_core.clebsch_gordan.real_basis` states the convention and makes two
claims about e3nn that only a test with e3nn installed can hold it to: e3nn's
real harmonics are the textbook ones of `real_basis_change` evaluated at the
permuted point (z, x, y), and the real 3j tables equal `e3nn.o3.wigner_3j` up
to one sign per triple. The first is why the l = 2 mixing seen when comparing
harmonics is a change of frame and not a property of the tables; the second is
what that argument predicts, and a sign per triple is absorbed by the weight of
the path it scales.
"""

import math

import numpy as np
import pytest
import torch
from e3nn import o3
from mace_core.clebsch_gordan.real_basis import real_basis_change, wigner_3j_real

from tests.golden.harness import tolerance

CLOSED_FORM = tolerance("closed_form_fp64")

MAX_DEGREE = 5

#: The triples, all degrees up to MAX_DEGREE, on which the real 3j table is
#: minus e3nn's. Measured once and committed; every other triple is equal.
SIGN_FLIPPED = frozenset(
    {
        (1, 1, 1),
        (2, 2, 3),
        (2, 3, 2),
        (2, 3, 4),
        (2, 4, 3),
        (2, 4, 5),
        (2, 5, 4),
        (3, 2, 2),
        (3, 2, 4),
        (3, 3, 3),
        (3, 3, 5),
        (3, 4, 2),
        (3, 4, 4),
        (3, 5, 3),
        (3, 5, 5),
        (4, 2, 3),
        (4, 2, 5),
        (4, 3, 2),
        (4, 3, 4),
        (4, 4, 3),
        (4, 5, 2),
        (5, 2, 4),
        (5, 3, 3),
        (5, 3, 5),
        (5, 4, 2),
        (5, 5, 3),
    }
)

TRIPLES = [
    (l1, l2, l3)
    for l1 in range(MAX_DEGREE + 1)
    for l2 in range(MAX_DEGREE + 1)
    for l3 in range(abs(l1 - l2), min(l1 + l2, MAX_DEGREE) + 1)
]


@pytest.mark.parametrize(("l1", "l2", "l3"), TRIPLES)
def test_the_real_3j_table_is_e3nn_up_to_one_sign_per_triple(l1, l2, l3):
    sign = -1.0 if (l1, l2, l3) in SIGN_FLIPPED else 1.0
    np.testing.assert_allclose(
        wigner_3j_real(l1, l2, l3),
        sign * o3.wigner_3j(l1, l2, l3, dtype=torch.float64).numpy(),
        atol=CLOSED_FORM.atol,
        rtol=CLOSED_FORM.rtol,
    )


def test_the_sign_table_names_only_triples_that_exist():
    """A pin that lists a triple outside the grid could never fail for it."""
    assert set(TRIPLES) >= SIGN_FLIPPED


@pytest.mark.parametrize("degree", range(MAX_DEGREE + 1))
def test_e3nn_harmonics_are_the_textbook_ones_at_z_x_y(degree):
    special = pytest.importorskip("scipy.special")
    if not hasattr(special, "sph_harm_y"):
        pytest.skip("needs scipy.special.sph_harm_y, scipy 1.15 or later")
    rng = np.random.default_rng(degree)
    directions = rng.normal(size=(64, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    x, y, z = directions[:, 2], directions[:, 0], directions[:, 1]
    polar, azimuth = np.arccos(z), np.arctan2(y, x)
    complex_harmonics = np.stack(
        [
            special.sph_harm_y(degree, order, polar, azimuth)
            for order in range(-degree, degree + 1)
        ]
    )
    textbook = real_basis_change(degree) @ complex_harmonics
    assert np.abs(textbook.imag).max() < CLOSED_FORM.atol

    e3nn = o3.spherical_harmonics(
        degree, torch.from_numpy(directions), normalize=True, normalization="component"
    ).numpy()
    np.testing.assert_allclose(
        math.sqrt(4 * math.pi) * textbook.real.T,
        e3nn,
        atol=CLOSED_FORM.atol,
        rtol=CLOSED_FORM.rtol,
    )
