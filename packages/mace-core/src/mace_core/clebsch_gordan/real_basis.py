"""The Wigner 3j table in the real basis, and the one statement of that basis.

The coefficients in :mod:`mace_core.clebsch_gordan.coefficients` are in the
complex spherical basis. Everything downstream works over real features, so
they are carried to a real basis here. This docstring is the only place the
convention is stated; the package docstring and the tests point here.

**The combinations.** Components of degree ``l`` run ``m = -l .. +l``, and the
real combinations are the textbook ones, with the Condon-Shortley phase in the
complex harmonics:

    m < 0:   (i/sqrt2) ( Y_l^m - (-1)^m Y_l^-m )
    m = 0:   Y_l^0
    m > 0:   (1/sqrt2) ( Y_l^-m + (-1)^m Y_l^m )

**Relation to e3nn.** The harmonics the models evaluate are e3nn's, with
component normalization and ``y`` as the polar axis; the v1 implementation
reproduces them (``tests/parity/test_spherical_harmonics_parity.py``). Those
are exactly the textbook real harmonics above evaluated at the permuted point
``(z, x, y)``, times ``sqrt(4 pi)``. That permutation is a proper rotation, so
it acts on each degree by an orthogonal matrix, and a 3j table is an
intertwiner: it is unchanged when one rotation acts on all three of its
indices. The tables are therefore correct in either frame, and
:func:`wigner_3j_real` equals ``e3nn.o3.wigner_3j`` up to one overall sign per
triple ``(l1, l2, l3)``. Over every triple with degrees up to 5, 26 triples
differ by that sign and none differ in any other way. A sign per triple is
absorbed by the weight that multiplies the path, so it is a gauge choice and
not a numerical difference. ``tests/parity/test_clebsch_gordan_parity.py``
pins both facts, including which triples flip.

Comparing the harmonics themselves against e3nn's on random directions shows a
signed permutation at ``l = 0`` and ``l = 1`` and, at ``l = 2``, a rotation that
mixes ``m = 0`` with ``m = +2``. That is the change of frame acting on the
harmonics, not a property of the tables.
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
