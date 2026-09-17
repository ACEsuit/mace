"""The Wigner 3j table in the real basis the models actually use.

The coefficients in :mod:`mace_core.clebsch_gordan.coefficients` are in the
complex spherical basis. Everything downstream works over real features, so
they have to be carried across, and which real basis that is is a convention.

**The convention is the textbook one**, stated here once because two tickets
depend on it agreeing: components run ``m = -l .. +l``, and the real
combinations are

    m < 0:   (i/sqrt2) ( Y_l^m - (-1)^m Y_l^-m )
    m = 0:   Y_l^0
    m > 0:   (1/sqrt2) ( Y_l^-m + (-1)^m Y_l^m )

This is *not* e3nn's basis, and the difference is not a relabelling. Measured
against ``o3.spherical_harmonics`` on random directions, e3nn agrees up to a
signed permutation at l = 0 and l = 1 and then diverges: at l = 2 the
transformation mixes m = 0 with m = +2 through a rotation. Reproducing e3nn
element for element would mean reproducing its construction, which is a
different orthogonal basis of the same space carrying no justification beyond
being the one that library chose.

So v1 states its own, and the legacy converter absorbs the difference, exactly
as it already has to absorb the node embedding's factor of sqrt(num_elements).
Equivalence is what gets tested: the spans agree and the change of basis is
orthogonal. Nothing observable depends on which of the two is used, because the
weights that multiply the basis are learned.
"""

from __future__ import annotations

import math

import numpy as np

from mace_core.clebsch_gordan.coefficients import wigner_3j_complex

__all__ = ["real_basis_change", "wigner_3j_real"]


def real_basis_change(degree: int) -> np.ndarray:
    """The unitary carrying the complex spherical basis to the real one.

    Args:
        degree: The rotation order ``l``.

    Returns:
        Shape ``(2l+1, 2l+1)``, complex. Row ``i`` holds the complex
        coefficients of real component ``m = i - l``.
    """
    matrix = np.zeros((2 * degree + 1, 2 * degree + 1), dtype=np.complex128)
    half = 1 / math.sqrt(2)
    for index, m in enumerate(range(-degree, degree + 1)):
        if m < 0:
            matrix[index, degree + m] = 1j * half
            matrix[index, degree - m] = -1j * half * (-1) ** m
        elif m == 0:
            matrix[index, degree] = 1.0
        else:
            matrix[index, degree - m] = half
            matrix[index, degree + m] = half * (-1) ** m
    return matrix


def wigner_3j_real(l1: int, l2: int, l3: int) -> np.ndarray:
    """The 3j table of three degrees, in the real basis.

    Args:
        l1, l2, l3: The three degrees.

    Returns:
        Shape ``(2*l1+1, 2*l2+1, 2*l3+1)``, real, with unit sum of squares
        whenever the triangle inequality holds. Zero otherwise.

    The factor of ``i**(l1+l2+l3)`` is what makes the result real. The raw
    change of basis leaves a tensor that is either real or purely imaginary
    depending on the parity of the degree sum, and that single factor covers
    both cases; the assertion below states it rather than trusting it, because
    a silently complex basis would surface much later as a wrong gradient.
    """
    complex_table = wigner_3j_complex(l1, l2, l3).astype(np.complex128)
    carried = np.einsum(
        "ai,bj,ck,ijk->abc",
        real_basis_change(l1),
        real_basis_change(l2),
        real_basis_change(l3),
        complex_table,
    )
    carried = (1j) ** (l1 + l2 + l3) * carried
    residue = float(np.abs(carried.imag).max())
    if residue > 1e-12:
        raise AssertionError(
            f"the real 3j table for degrees ({l1}, {l2}, {l3}) came out "
            f"complex, with a largest imaginary part of {residue:.3e}. The "
            f"phase convention in this module is wrong for this triple."
        )
    return np.ascontiguousarray(carried.real)
