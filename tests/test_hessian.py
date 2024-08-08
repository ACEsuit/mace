import numpy as np
import pytest
from ase.build import fcc111

from mace.calculators import mace_mp


@pytest.fixture(name="setup_calculator_")
def setup_calculator():
    calc = mace_mp(
        model="medium", dispersion=False, default_dtype="float64", device="cpu"
    )
    return calc


@pytest.fixture(name="setup_structure_")
def setup_structure(setup_calculator_):
    initial = fcc111("Pt", size=(4, 4, 1), vacuum=10.0, orthogonal=True)
    initial.calc = setup_calculator_
    return initial


def test_potential_energy_and_hessian(setup_structure_):
    initial = setup_structure_
    h_autograd = initial.calc.get_hessian(atoms=initial)
    assert h_autograd.shape == (len(initial) * 3, len(initial), 3)


def test_finite_difference_hessian(setup_structure_):
    initial = setup_structure_
    indicies = list(range(len(initial)))
    delta, ndim = 1e-4, 3
    hessian = np.zeros((len(indicies) * ndim, len(indicies) * ndim))
    atoms_h = initial.copy()
    for i, index in enumerate(indicies):
        for j in range(ndim):
            atoms_i = atoms_h.copy()
            atoms_i.positions[index, j] += delta
            atoms_i.calc = initial.calc
            forces_i = atoms_i.get_forces()

            atoms_j = atoms_h.copy()
            atoms_j.positions[index, j] -= delta
            atoms_j.calc = initial.calc
            forces_j = atoms_j.get_forces()

            hessian[:, i * ndim + j] = -(forces_i - forces_j)[indicies].flatten() / (
                2 * delta
            )

    hessian = hessian.reshape((-1, len(initial), 3))
    h_autograd = initial.calc.get_hessian(atoms=initial)
    is_close = np.allclose(h_autograd, hessian, atol=1e-6)
    assert is_close
