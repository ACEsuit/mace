"""The Racah coefficients, against an independent derivation.

The oracle is sympy's, which computes the same quantities by a different route
and is not the code under test. It is a test-only dependency: `mace_core` must
not acquire it, and the suite skips rather than fails where it is absent.
"""

import importlib.util
import math

import numpy as np
import pytest
from mace_core.clebsch_gordan import clebsch_gordan, wigner_3j_complex

if importlib.util.find_spec("sympy") is None:  # pragma: no cover
    pytest.skip("needs sympy as an independent oracle", allow_module_level=True)

from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_3j as sympy_3j

# One unit in the last place at fp64. These coefficients are a closed form, so
# the bar is the closed-form row of the project's tolerance table, not the
# looser rows that exist for cross-implementation comparisons.
ATOL = 1e-12
RTOL = 1e-12

DEGREES = range(5)


def triples():
    for l1 in DEGREES:
        for l2 in DEGREES:
            for l3 in range(abs(l1 - l2), min(l1 + l2, max(DEGREES)) + 1):
                yield l1, l2, l3


def test_the_selection_rules_return_zero_rather_than_raising():
    assert clebsch_gordan(1, 0, 1, 0, 5, 0) == 0.0  # triangle inequality
    assert clebsch_gordan(1, 1, 1, 1, 2, 0) == 0.0  # m1 + m2 != m3
    assert clebsch_gordan(1, 2, 1, 0, 1, 2) == 0.0  # |m| > l


@pytest.mark.parametrize(("l1", "l2", "l3"), list(triples()))
def test_the_coefficients_match_an_independent_derivation(l1, l2, l3):
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            m3 = m1 + m2
            if abs(m3) > l3:
                continue
            mine = clebsch_gordan(l1, m1, l2, m2, l3, m3)
            theirs = float(CG(l1, m1, l2, m2, l3, m3).doit())
            assert mine == pytest.approx(theirs, abs=ATOL, rel=RTOL)


@pytest.mark.parametrize(("l1", "l2", "l3"), list(triples()))
def test_the_3j_table_matches_an_independent_derivation(l1, l2, l3):
    table = wigner_3j_complex(l1, l2, l3)
    assert table.shape == (2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
    for a, m1 in enumerate(range(-l1, l1 + 1)):
        for b, m2 in enumerate(range(-l2, l2 + 1)):
            for c, m3 in enumerate(range(-l3, l3 + 1)):
                theirs = float(sympy_3j(l1, l2, l3, m1, m2, m3))
                assert table[a, b, c] == pytest.approx(theirs, abs=ATOL, rel=RTOL)


@pytest.mark.parametrize(("l1", "l2", "l3"), list(triples()))
def test_the_3j_table_is_normalised_to_one(l1, l2, l3):
    """The property the change to the real basis has to preserve: a unitary
    basis change cannot move the sum of the squares."""
    assert float((wigner_3j_complex(l1, l2, l3) ** 2).sum()) == pytest.approx(
        1.0, abs=ATOL, rel=RTOL
    )


def test_a_forbidden_triple_gives_an_all_zero_table():
    assert not wigner_3j_complex(1, 1, 5).any()


def test_two_known_closed_forms():
    """Two cases whose value is known without any table at all: the l=1 scalar
    coupling is the identity over sqrt(3), and 3j(1,1,1) carries the
    Levi-Civita structure, one over sqrt(6)."""
    scalar = wigner_3j_complex(1, 1, 0)[:, :, 0]
    assert abs(float(np.abs(scalar).max()) - 1 / math.sqrt(3)) < ATOL
    antisymmetric = wigner_3j_complex(1, 1, 1)
    assert abs(float(np.abs(antisymmetric).max()) - 1 / math.sqrt(6)) < ATOL
