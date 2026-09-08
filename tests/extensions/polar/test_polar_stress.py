"""Finite-difference stress checks for the published medium Polar model.

Adapted from mace_scf's LocalSplitCharges tests (b7697fb): compact cubic and
triclinic cells, a nonzero normal/shear signal, and periodic-image invariance.
Unlike large dilute boxes, these geometries expose missing electrostatic strain
derivatives. All stress tolerances are in eV/Angstrom**3.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.fd import calculate_numerical_stress

from mace.calculators import mace_polar

pytestmark = [pytest.mark.polar, pytest.mark.network]

CELLS = [
    np.diag([7.0, 7.0, 7.0]),
    np.array([[7.0, 0.0, 0.0], [1.3, 6.8, 0.0], [0.9, -1.1, 6.9]]),
]
IMAGE_SHIFTS = np.array(
    [[1, 0, 0], [1, 0, 0], [0, -1, 0], [-1, 1, 0], [0, 0, 1], [200, 0, -1]]
)
STRESS_ATOL = 1e-9
FD_STEP = 1e-5


def water_dimer(cell):
    atoms = Atoms(
        "OHHOHH",
        positions=[
            [3.10, 3.05, 3.12],
            [3.65, 3.80, 2.95],
            [2.31, 3.36, 3.55],
            [4.85, 5.10, 4.60],
            [5.25, 4.35, 5.05],
            [4.25, 5.55, 5.22],
        ],
        cell=cell,
        pbc=True,
    )
    atoms.info.update(charge=0, spin=1, external_field=[0.0, 0.0, 0.0])
    return atoms


@pytest.fixture(scope="module")
def polar_calculator():
    return mace_polar(
        model="polar-1-m", device="cpu", default_dtype="float64", pbc_handling="pbc"
    )


def check_finite_difference(atoms):
    analytic = atoms.get_stress(voigt=True)
    numerical = calculate_numerical_stress(
        atoms, eps=FD_STEP, voigt=True, force_consistent=False
    )
    assert np.abs(numerical[:3]).max() > 1e-6
    assert np.abs(numerical[3:]).max() > 1e-6
    error = np.abs(analytic - numerical)
    print(f"max stress error={error.max():.12e}; component errors={error}")
    np.testing.assert_allclose(analytic, numerical, rtol=0, atol=STRESS_ATOL)


@pytest.mark.parametrize("cell", CELLS, ids=["cubic", "triclinic"])
def test_polar_stress_matches_finite_difference(polar_calculator, cell):
    atoms = water_dimer(cell)
    atoms.calc = polar_calculator
    check_finite_difference(atoms)


@pytest.mark.parametrize("cell", CELLS, ids=["cubic", "triclinic"])
def test_disabling_stress_preserves_energy_and_forces(
    polar_calculator, cell, monkeypatch
):
    without_stress = mace_polar(
        model="polar-1-m",
        device="cpu",
        default_dtype="float64",
        pbc_handling="pbc",
        compute_stress=False,
    )
    model = without_stress.models[0]
    forward = model.forward
    calls = []

    def checked_forward(*args, **kwargs):
        calls.append(kwargs["compute_stress"])
        out = forward(*args, **kwargs)
        assert out["stress"] is None
        return out

    monkeypatch.setattr(model, "forward", checked_forward)
    atoms = water_dimer(cell)
    atoms.calc = polar_calculator
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = without_stress
    np.testing.assert_allclose(atoms.get_potential_energy(), energy, rtol=0, atol=1e-10)
    np.testing.assert_allclose(atoms.get_forces(), forces, rtol=0, atol=1e-10)
    assert calls == [False]
    assert "stress" not in without_stress.results
    assert "stress" not in without_stress.implemented_properties
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()


@pytest.mark.parametrize("cell", CELLS, ids=["cubic", "triclinic"])
def test_polar_stress_under_periodic_image_shifts(polar_calculator, cell):
    atoms = water_dimer(cell)
    atoms.calc = polar_calculator
    reference = atoms.get_stress(voigt=True)
    atoms.positions += IMAGE_SHIFTS @ cell
    shifted = atoms.get_stress(voigt=True)
    print(f"max image-shift stress error={np.abs(shifted - reference).max():.12e}")
    np.testing.assert_allclose(shifted, reference, rtol=0, atol=1e-12)
    check_finite_difference(atoms)
