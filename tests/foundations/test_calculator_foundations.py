"""Tests of the foundation-model constructors (mace_mp / mace_off / mace_omol /
mace_polar) through the ASE calculator. Everything here downloads pretrained
models (capability `network`, auto-marked by directory).

Split out of tests/test_calculator.py; the trained-from-scratch calculator
tests live in tests/workflows/test_calculator.py."""

import numpy as np
import pytest
import torch
from ase import build
from ase.atoms import Atoms
from ase.filters import FrechetCellFilter

import ase.io

from mace.calculators import mace_mp, mace_off
from mace.calculators.foundations_models import mace_omol, mace_polar
from mace.calculators.mace import MACECalculator
from mace.modules.models import ScaleShiftMACE
from mace.tools.torch_tools import default_dtype
from tests.helpers import CUET_AVAILABLE


def write_extxyz_test(tmp_path, atoms):
    assert isinstance(
        atoms, Atoms
    ), "write_extxyz_test only working for Atoms, not anything else such as list(Atoms)"
    ase.io.write(tmp_path / "test.extxyz", atoms)
    atoms_written = ase.io.read(tmp_path / "test.extxyz")

    nonstd_fields = set(
        [
            "node_energy",
            "energy_var",
            "energy_comm",
            "stress_var",
            "stress_comm",
            "forces_var",
            "forces_comm",
            "virials",
        ]
    )
    # everything that we expect has been written
    assert set(atoms.calc.results.keys()) - nonstd_fields == set(
        atoms_written.calc.results.keys()
    )
    # everything that was written was correct
    assert all(
        np.allclose(atoms.calc.results[k], atoms_written.calc.results[k])
        for k in atoms_written.calc.results
    )


@pytest.mark.network
def test_mace_mp(capsys: pytest.CaptureFixture):
    mp_mace = mace_mp()
    assert isinstance(mp_mace, MACECalculator)
    assert mp_mace.model_type == "MACE"
    assert len(mp_mace.models) == 1
    assert isinstance(mp_mace.models[0], ScaleShiftMACE)

    _, stderr = capsys.readouterr()
    assert stderr == ""


@pytest.mark.network
@pytest.mark.polar
def test_mace_polar_constructor():
    try:
        import graph_longrange  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pytest.skip("graph_longrange is not installed")
    model_name = "polar-1-m"
    try:
        polar_calc = mace_polar(model=model_name, device="cpu")
    except (FileNotFoundError, RuntimeError):
        pytest.skip(f"Missing Polar foundation model file: {model_name}")

    assert isinstance(polar_calc, MACECalculator)
    assert len(polar_calc.models) == 1
    assert polar_calc.models[0].__class__.__name__ == "PolarMACE"


@pytest.mark.network
def test_mace_off(tmp_path):
    mace_off_model = mace_off(model="small", device="cpu")
    assert isinstance(mace_off_model, MACECalculator)
    assert mace_off_model.model_type == "MACE"
    assert len(mace_off_model.models) == 1
    assert isinstance(mace_off_model.models[0], ScaleShiftMACE)

    atoms = build.molecule("H2O")
    atoms.calc = mace_off_model

    E = atoms.get_potential_energy()

    assert np.allclose(E, -2081.116128586803, atol=1e-9)
    write_extxyz_test(tmp_path, atoms)



@pytest.mark.network
@pytest.mark.cueq
@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_mace_off_cueq(tmp_path, model="medium", device="cpu"):
    mace_off_model = mace_off(model=model, device=device, enable_cueq=True)
    assert isinstance(mace_off_model, MACECalculator)
    assert mace_off_model.model_type == "MACE"
    assert len(mace_off_model.models) == 1
    assert isinstance(mace_off_model.models[0], ScaleShiftMACE)

    atoms = build.molecule("H2O")
    atoms.calc = mace_off_model

    E = atoms.get_potential_energy()

    assert np.allclose(E, -2081.116128586803, atol=1e-9)
    write_extxyz_test(tmp_path, atoms)


@pytest.mark.network
def test_mace_mp_stresses(tmp_path, model="medium", device="cpu"):
    atoms = build.bulk("Al", "fcc", a=4.05, cubic=True)
    atoms = atoms.repeat((2, 2, 2))
    mace_mp_model = mace_mp(model=model, device=device, compute_atomic_stresses=True)
    atoms.set_calculator(mace_mp_model)
    stress = atoms.get_stress()
    stresses = atoms.get_stresses()
    assert stress.shape == (6,)
    assert stresses.shape == (32, 6)
    assert np.allclose(stress, stresses.sum(axis=0), atol=1e-6)
    write_extxyz_test(tmp_path, atoms)


@pytest.mark.network
def test_mace_mp_energies(tmp_path, model="medium", device="cpu"):
    atoms = build.bulk("Al", "fcc", a=4.05, cubic=True)
    atoms = atoms.repeat((2, 2, 2))
    mace_mp_model = mace_mp(model=model, device=device)
    atoms.set_calculator(mace_mp_model)
    energy = atoms.get_potential_energy()
    energies = atoms.get_potential_energies()
    assert energies.shape == (len(atoms),)
    assert np.allclose(energy, energies.sum(), atol=1e-6)
    write_extxyz_test(tmp_path, atoms)


@pytest.mark.network
@pytest.mark.cueq
@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_mace_mh_1_cueq(tmp_path, device="cpu"):

    calc = mace_mp(
        model="mh-1", device=device, default_dtype="float64", head="omat_pbe"
    )
    mol = build.molecule("H2O")
    mol.set_calculator(calc)
    energy = mol.get_potential_energy()
    forces = mol.get_forces()

    # reset the calculator to test CUEQ
    mol.calc.reset()
    calc_cueq = mace_mp(
        model="mh-1",
        device=device,
        default_dtype="float64",
        head="omat_pbe",
        enable_cueq=True,
    )
    mol.set_calculator(calc_cueq)
    energy_cueq = mol.get_potential_energy()
    forces_cueq = mol.get_forces()
    assert np.allclose(energy, energy_cueq, atol=1e-6)
    assert np.allclose(forces, forces_cueq, atol=1e-6)
    write_extxyz_test(tmp_path, mol)


@pytest.mark.network
@pytest.mark.cueq
@pytest.mark.gpu
@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_mace_mh_1_cueq_cuda(tmp_path):
    test_mace_mh_1_cueq(tmp_path, device="cuda")


@pytest.mark.network
@pytest.mark.cueq
@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_mace_omol_cueq(tmp_path, device="cpu"):

    calc = mace_omol(device=device, default_dtype="float64")
    mol = build.molecule("H2O")
    mol.set_calculator(calc)
    energy = mol.get_potential_energy()
    forces = mol.get_forces()

    # reset the calculator to test CUEQ
    mol.calc.reset()
    calc_cueq = mace_omol(device=device, enable_cueq=True, default_dtype="float64")
    mol.set_calculator(calc_cueq)
    energy_cueq = mol.get_potential_energy()
    forces_cueq = mol.get_forces()
    assert np.allclose(energy, energy_cueq, atol=1e-6)
    assert np.allclose(forces, forces_cueq, atol=1e-6)
    assert np.allclose(energy, -2079.863496758961, atol=1e-9)
    write_extxyz_test(tmp_path, mol)

