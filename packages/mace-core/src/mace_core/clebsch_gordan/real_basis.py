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
from functools import cache

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


AXIS_PERMUTATION = (2, 0, 1)


@cache
def textbook_harmonics(degree: int, directions: tuple) -> np.ndarray:
    """Real spherical harmonics in the textbook basis, on given directions.

    Built by the same recursion the model's harmonics use: each degree is the
    previous one coupled with the l = 1 block through the 3j of this module.
    It exists here, rather than only in the framework packages, because
    :func:`induced_rotation` needs it and has to stay framework-free.
    """
    points = np.asarray(directions, dtype=float).reshape(-1, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    blocks = [np.ones((points.shape[0], 1))]
    if degree == 0:
        return blocks[0]
    first = points[:, [1, 2, 0]]
    blocks.append(first)
    for order in range(2, degree + 1):
        coupling = wigner_3j_real(order, order - 1, 1)
        raw = np.einsum("oab,na,nb->no", coupling, blocks[-1], first)
        blocks.append(raw / np.linalg.norm(raw, axis=1, keepdims=True).mean())
    return blocks[degree]


@cache
def induced_rotation(degree: int) -> np.ndarray:
    """The orthogonal map from the textbook basis to e3nn's, at one degree.

    Solved for from the textbook harmonics alone: evaluating them on a set of
    directions and on the same directions with the axes permuted gives
    ``Y(Pr) = Y(r) @ Q``, and ``Q`` is that matrix. No e3nn is consulted, and
    none can be: this module is imported by a package that must not have it.

    A fixed set of directions rather than a random one, so the result is the
    same on every machine and every run. The system is heavily overdetermined,
    so the particular directions do not matter as long as they are generic.
    """
    if degree == 0:
        return np.ones((1, 1))
    angles = np.linspace(0.13, 3.01, 4 * (2 * degree + 1))
    points = np.stack(
        [np.cos(angles) * 0.9, np.sin(angles) * 0.8, np.cos(2.0 * angles) * 0.7], axis=1
    )
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    permuted = points[:, list(AXIS_PERMUTATION)]
    here = textbook_harmonics(degree, tuple(map(tuple, points)))
    there = textbook_harmonics(degree, tuple(map(tuple, permuted)))
    rotation, *_ = np.linalg.lstsq(here, there, rcond=None)
    return np.ascontiguousarray(rotation)


def wigner_d_real(degree: int, rotation: np.ndarray) -> np.ndarray:
    """How one irrep's components move when space is rotated.

    The representation matrix ``D`` of the rotation at this degree, in the
    project's own basis, so that a feature row ``f`` carrying this irrep goes
    to ``f @ D`` when the positions go to ``r @ rotation.T``.

    Solved from the project's own harmonics rather than from a closed form in
    Euler angles: the basis is the one this module defines, so deriving ``D``
    from anything else would be asserting a convention instead of reading it.
    The system is overdetermined by a factor of four, which is what makes the
    least squares an identification rather than a fit.

    Args:
        degree: The irrep's degree.
        rotation: A ``(3, 3)`` proper rotation.

    Returns:
        A ``(2 * degree + 1, 2 * degree + 1)`` orthogonal matrix.
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(
            f"a rotation is a (3, 3) matrix, and this one has shape {rotation.shape}."
        )
    if degree == 0:
        return np.ones((1, 1))

    angles = np.linspace(0.13, 3.01, 4 * (2 * degree + 1))
    points = np.stack(
        [np.cos(angles) * 0.9, np.sin(angles) * 0.8, np.cos(2.0 * angles) * 0.7], axis=1
    )
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    turned = points @ rotation.T

    change = induced_rotation(degree)
    here = textbook_harmonics(degree, tuple(map(tuple, points))) @ change
    there = textbook_harmonics(degree, tuple(map(tuple, turned))) @ change
    wigner, *_ = np.linalg.lstsq(here, there, rcond=None)

    residue = np.abs(wigner @ wigner.T - np.eye(2 * degree + 1)).max()
    if residue > 1e-9:
        raise AssertionError(
            f"the Wigner matrix for degree {degree} came out non-orthogonal by "
            f"{residue:.3e}, which means the argument was not a proper "
            f"rotation. Its determinant is {np.linalg.det(rotation):.6f}."
        )
    return np.ascontiguousarray(wigner)


@cache
def symmetric_matrix_basis() -> np.ndarray:
    """A symmetric 3x3 matrix as ``0e+2e``, and back.

    Row ``m`` is the matrix component ``m`` of the six spherical components
    stands for, so a spherical row ``t`` is the matrix
    ``einsum("mij,m->ij", basis, t)``, and since the rows are orthonormal the
    same table read the other way takes a symmetric matrix to its six
    components. The first row is the trace part, ``I / sqrt(3)``.

    These are the 3j tables coupling two vectors to degrees 0 and 2, scaled to
    unit norm per row: the matrix ``v w^T`` of two vectors is exactly that
    coupling. The components are therefore in the project's basis, the one the
    harmonics and every readout use, which is also the frozen tree's.

    Returns:
        ``(6, 3, 3)``, read only.
    """
    rows = [
        np.moveaxis(wigner_3j_real(1, 1, degree), -1, 0) * math.sqrt(2 * degree + 1)
        for degree in (0, 2)
    ]
    basis = np.concatenate(rows, axis=0)
    basis.setflags(write=False)
    return basis


__all__ += [
    "AXIS_PERMUTATION",
    "induced_rotation",
    "symmetric_matrix_basis",
    "textbook_harmonics",
    "wigner_d_real",
]
