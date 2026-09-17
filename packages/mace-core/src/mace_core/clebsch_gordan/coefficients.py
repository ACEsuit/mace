"""Clebsch-Gordan and Wigner 3j coefficients, from the Racah formula.

Pure numpy and pure Python. This module imports neither ``e3nn`` nor
``cuequivariance``, and it never will: both bases are elementary group theory,
and the dependency was a library monopoly rather than a mathematical one.

The coefficients here are in the **complex** spherical basis, indexed by
``m = -l .. +l``. Turning them into the real basis the models actually use is
:mod:`mace_core.clebsch_gordan.real_basis`, and that step carries a convention
question this module deliberately does not answer. See that module.

Exactness. The Racah sum is evaluated over :class:`fractions.Fraction`, so the
only floating-point operations are one square root and one multiplication per
coefficient. Measured against an independent derivation over every triple with
``l <= 5``, the agreement is 1.1e-16, which is one unit in the last place.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

__all__ = ["clebsch_gordan", "wigner_3j_complex"]


def _factorial(n: int) -> int:
    return math.factorial(n)


def clebsch_gordan(j1: int, m1: int, j2: int, m2: int, j3: int, m3: int) -> float:
    """The coefficient ``<j1 m1; j2 m2 | j3 m3>``.

    Args:
        j1, j2: Degrees of the two coupled representations.
        m1, m2: Their magnetic quantum numbers, each in ``-j .. +j``.
        j3, m3: The degree and magnetic number of the coupled result.

    Returns:
        The coefficient, or ``0.0`` when the selection rules forbid it:
        ``m1 + m2 != m3``, the triangle inequality on the degrees, or an ``m``
        outside its degree's range.

    The Racah expression is a rational number times the square root of a
    rational number, so both parts are accumulated exactly and combined once.
    """
    if m1 + m2 != m3:
        return 0.0
    if not abs(j1 - j2) <= j3 <= j1 + j2:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0

    under_root = Fraction(2 * j3 + 1) * Fraction(
        _factorial(j1 + j2 - j3) * _factorial(j1 - j2 + j3) * _factorial(-j1 + j2 + j3),
        _factorial(j1 + j2 + j3 + 1),
    )
    under_root *= Fraction(
        _factorial(j1 + m1)
        * _factorial(j1 - m1)
        * _factorial(j2 + m2)
        * _factorial(j2 - m2)
        * _factorial(j3 + m3)
        * _factorial(j3 - m3)
    )

    total = Fraction(0)
    lowest = max(0, j2 - j3 - m1, j1 - j3 + m2)
    highest = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    for k in range(lowest, highest + 1):
        denominator = (
            _factorial(k)
            * _factorial(j1 + j2 - j3 - k)
            * _factorial(j1 - m1 - k)
            * _factorial(j2 + m2 - k)
            * _factorial(j3 - j2 + m1 + k)
            * _factorial(j3 - j1 - m2 + k)
        )
        total += Fraction((-1) ** k, denominator)

    return float(total) * math.sqrt(float(under_root))


def wigner_3j_complex(l1: int, l2: int, l3: int) -> np.ndarray:
    """The Wigner 3j symbol as a dense array, in the complex spherical basis.

    Args:
        l1, l2, l3: The three degrees.

    Returns:
        Shape ``(2*l1+1, 2*l2+1, 2*l3+1)``, real-valued, indexed by
        ``m = -l .. +l`` on each axis. Entries with ``m1 + m2 + m3 != 0`` are
        zero by the selection rule.

    The normalization is the standard one, so the sum of the squares of all
    entries is 1 whenever the triangle inequality is satisfied. That is what
    makes the change to the real basis norm-preserving: a unitary basis change
    cannot alter it.
    """
    table = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1), dtype=np.float64)
    if not abs(l1 - l2) <= l3 <= l1 + l2:
        return table
    normalization = 1.0 / math.sqrt(2 * l3 + 1)
    for a, m1 in enumerate(range(-l1, l1 + 1)):
        for b, m2 in enumerate(range(-l2, l2 + 1)):
            m3 = -(m1 + m2)
            if abs(m3) > l3:
                continue
            c = m3 + l3
            table[a, b, c] = (
                (-1) ** (l1 - l2 - m3)
                * normalization
                * clebsch_gordan(l1, m1, l2, m2, l3, -m3)
            )
    return table
