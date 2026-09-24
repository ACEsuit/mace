"""A symmetric 3x3 matrix as ``0e+2e``: the table and what it has to satisfy."""

from __future__ import annotations

import numpy as np
from mace_core.clebsch_gordan import symmetric_matrix_basis
from mace_core.clebsch_gordan.real_basis import wigner_d_real

#: The frozen tree's table, which e3nn builds as the change of basis of a
#: symmetric rank-two Cartesian tensor. Written out so this package need not
#: import it; the values are the closed forms, rounded by nothing.
S2, S3, S6 = np.sqrt(2.0), np.sqrt(3.0), np.sqrt(6.0)
FROZEN_TREE = np.array(
    [
        np.eye(3) / S3,
        [[0, 0, 1 / S2], [0, 0, 0], [1 / S2, 0, 0]],
        [[0, 1 / S2, 0], [1 / S2, 0, 0], [0, 0, 0]],
        [[-1 / S6, 0, 0], [0, 2 / S6, 0], [0, 0, -1 / S6]],
        [[0, 0, 0], [0, 0, 1 / S2], [0, 1 / S2, 0]],
        [[-1 / S2, 0, 0], [0, 0, 0], [0, 0, 1 / S2]],
    ]
)


def test_the_table_is_the_frozen_tree_s():
    np.testing.assert_allclose(symmetric_matrix_basis(), FROZEN_TREE, atol=1e-15)


def test_the_rows_are_orthonormal_symmetric_matrices():
    basis = symmetric_matrix_basis()
    gram = np.einsum("mij,nij->mn", basis, basis)
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-14)
    np.testing.assert_allclose(basis, np.transpose(basis, (0, 2, 1)), atol=1e-15)
    np.testing.assert_allclose(np.trace(basis[1:], axis1=1, axis2=2), 0.0, atol=1e-14)


def test_a_rotated_matrix_is_its_components_rotated():
    """``R M R^T`` has the components ``t @ D``, degree by degree: the matrix is
    the same object as its spherical components, in the project's basis."""
    basis = symmetric_matrix_basis()
    generator = np.random.default_rng(3)
    axis = generator.normal(size=3)
    angle = 0.8
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
    components = generator.normal(size=6)
    matrix = np.einsum("mij,m->ij", basis, components)
    turned = components.copy()
    turned[1:] = components[1:] @ wigner_d_real(2, rotation)
    np.testing.assert_allclose(
        np.einsum("mij,m->ij", basis, turned),
        rotation @ matrix @ rotation.T,
        atol=1e-13,
    )


def test_the_table_is_read_only():
    basis = symmetric_matrix_basis()
    assert not basis.flags.writeable
