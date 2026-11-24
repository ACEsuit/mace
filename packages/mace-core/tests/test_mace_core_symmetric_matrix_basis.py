"""The basis a symmetric matrix is read out in: six orthonormal matrices, the
trace part first."""

import numpy as np
from mace_core.clebsch_gordan import symmetric_matrix_basis


def test_the_six_matrices_are_orthonormal_and_symmetric():
    basis = symmetric_matrix_basis()
    gram = np.einsum("kij,lij->kl", basis, basis)
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-15)
    np.testing.assert_allclose(basis, basis.transpose(0, 2, 1), atol=0)


def test_the_first_is_the_trace_and_the_others_are_traceless():
    basis = symmetric_matrix_basis()
    np.testing.assert_allclose(basis[0], np.eye(3) / np.sqrt(3.0), atol=1e-15)
    np.testing.assert_allclose(np.trace(basis[1:], axis1=1, axis2=2), 0.0, atol=1e-15)


def test_a_matrix_and_its_components_turn_into_each_other():
    matrix = np.array([[1.0, 0.2, -0.3], [0.2, 2.0, 0.5], [-0.3, 0.5, -0.7]])
    basis = symmetric_matrix_basis()
    components = np.einsum("kij,ij->k", basis, matrix)
    np.testing.assert_allclose(np.einsum("kij,k->ij", basis, components), matrix)
