import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.test import gradient_test
from ase.constraints import ExpCellFilter
from ase import build

from mace.calculators import mace_mp, mace_off
from mace.calculators.foundations_models import local_model_path
from mace.calculators.mace import MACECalculator
from mace.modules.models import ScaleShiftMACE

pytest_mace_dir = Path(__file__).parent.parent
run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"


@pytest.fixture(scope="module", name="fitting_configs")
def fitting_configs_fixture():
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    fit_configs = [
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3),
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3),
    ]
    fit_configs[0].info["REF_energy"] = 0.0
    fit_configs[0].info["config_type"] = "IsolatedAtom"
    fit_configs[1].info["REF_energy"] = 0.0
    fit_configs[1].info["config_type"] = "IsolatedAtom"

    np.random.seed(5)
    for _ in range(20):
        c = water.copy()
        c.positions += np.random.normal(0.1, size=c.positions.shape)
        c.info["REF_energy"] = np.random.normal(0.1)
        c.info["REF_dipole"] = np.random.normal(0.1, size=3)
        c.new_array("REF_forces", np.random.normal(0.1, size=c.positions.shape))
        c.new_array("Qs", np.random.normal(0.1, size=c.positions.shape[0]))
        c.info["REF_stress"] = np.random.normal(0.1, size=6)
        fit_configs.append(c)

    return fit_configs


@pytest.fixture(scope="module", name="trained_model")
def trained_model_fixture(tmp_path_factory, fitting_configs):
    _mace_params = {
        "name": "MACE",
        "valid_fraction": 0.05,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "MACE",
        "hidden_irreps": "128x0e",
        "r_max": 3.5,
        "batch_size": 5,
        "max_num_epochs": 10,
        "swa": None,
        "start_swa": 5,
        "ema": None,
        "ema_decay": 0.99,
        "amsgrad": None,
        "restart_latest": None,
        "device": "cpu",
        "seed": 5,
        "loss": "stress",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
    }

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    # make sure run_train.py is using the mace that is currently being tested
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

    cmd = (
        sys.executable
        + " "
        + str(run_train)
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), env=run_env, check=True)

    assert p.returncode == 0

    return MACECalculator(tmp_path / "MACE.model", device="cpu")


@pytest.fixture(scope="module", name="trained_equivariant_model")
def trained_model_equivariant_fixture(tmp_path_factory, fitting_configs):
    _mace_params = {
        "name": "MACE",
        "valid_fraction": 0.05,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "MACE",
        "hidden_irreps": "16x0e+16x1o",
        "r_max": 3.5,
        "batch_size": 5,
        "max_num_epochs": 10,
        "swa": None,
        "start_swa": 5,
        "ema": None,
        "ema_decay": 0.99,
        "amsgrad": None,
        "restart_latest": None,
        "device": "cpu",
        "seed": 5,
        "loss": "stress",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
    }

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    # make sure run_train.py is using the mace that is currently being tested
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

    cmd = (
        sys.executable
        + " "
        + str(run_train)
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), env=run_env, check=True)

    assert p.returncode == 0

    return MACECalculator(tmp_path / "MACE.model", device="cpu")


@pytest.fixture(scope="module", name="trained_dipole_model")
def trained_dipole_fixture(tmp_path_factory, fitting_configs):
    _mace_params = {
        "name": "MACE",
        "valid_fraction": 0.05,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "AtomicDipolesMACE",
        "num_channels": 8,
        "max_L": 2,
        "r_max": 3.5,
        "batch_size": 5,
        "max_num_epochs": 10,
        "ema": None,
        "ema_decay": 0.99,
        "amsgrad": None,
        "restart_latest": None,
        "device": "cpu",
        "seed": 5,
        "loss": "dipole",
        "energy_key": "",
        "forces_key": "",
        "stress_key": "",
        "dipole_key": "REF_dipole",
        "error_table": "DipoleRMSE",
    }

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    # make sure run_train.py is using the mace that is currently being tested
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

    cmd = (
        sys.executable
        + " "
        + str(run_train)
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), env=run_env, check=True)

    assert p.returncode == 0

    return MACECalculator(
        tmp_path / "MACE.model", device="cpu", model_type="DipoleMACE"
    )


@pytest.fixture(scope="module", name="trained_energy_dipole_model")
def trained_energy_dipole_fixture(tmp_path_factory, fitting_configs):
    _mace_params = {
        "name": "MACE",
        "valid_fraction": 0.05,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "EnergyDipolesMACE",
        "num_channels": 32,
        "max_L": 1,
        "r_max": 3.5,
        "batch_size": 5,
        "max_num_epochs": 10,
        "ema": None,
        "ema_decay": 0.99,
        "amsgrad": None,
        "restart_latest": None,
        "device": "cpu",
        "seed": 5,
        "loss": "energy_forces_dipole",
        "energy_key": "REF_energy",
        "forces_key": "",
        "stress_key": "",
        "dipole_key": "REF_dipole",
        "error_table": "EnergyDipoleRMSE",
    }

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    # make sure run_train.py is using the mace that is currently being tested
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

    cmd = (
        sys.executable
        + " "
        + str(run_train)
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), env=run_env, check=True)

    assert p.returncode == 0

    return MACECalculator(
        tmp_path / "MACE.model", device="cpu", model_type="EnergyDipoleMACE"
    )


@pytest.fixture(scope="module", name="trained_committee")
def trained_committee_fixture(tmp_path_factory, fitting_configs):
    _seeds = [5, 6, 7]
    _model_paths = []
    for seed in _seeds:
        _mace_params = {
            "name": f"MACE{seed}",
            "valid_fraction": 0.05,
            "energy_weight": 1.0,
            "forces_weight": 10.0,
            "stress_weight": 1.0,
            "model": "MACE",
            "hidden_irreps": "16x0e",
            "r_max": 3.5,
            "batch_size": 5,
            "max_num_epochs": 10,
            "swa": None,
            "start_swa": 5,
            "ema": None,
            "ema_decay": 0.99,
            "amsgrad": None,
            "restart_latest": None,
            "device": "cpu",
            "seed": seed,
            "loss": "stress",
            "energy_key": "REF_energy",
            "forces_key": "REF_forces",
            "stress_key": "REF_stress",
        }

        tmp_path = tmp_path_factory.mktemp(f"run{seed}_")

        ase.io.write(tmp_path / "fit.xyz", fitting_configs)

        mace_params = _mace_params.copy()
        mace_params["checkpoints_dir"] = str(tmp_path)
        mace_params["model_dir"] = str(tmp_path)
        mace_params["train_file"] = tmp_path / "fit.xyz"

        # make sure run_train.py is using the mace that is currently being tested
        run_env = os.environ.copy()
        sys.path.insert(0, str(Path(__file__).parent.parent))
        run_env["PYTHONPATH"] = ":".join(sys.path)
        print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

        cmd = (
            sys.executable
            + " "
            + str(run_train)
            + " "
            + " ".join(
                [
                    (f"--{k}={v}" if v is not None else f"--{k}")
                    for k, v in mace_params.items()
                ]
            )
        )

        p = subprocess.run(cmd.split(), env=run_env, check=True)

        assert p.returncode == 0

        _model_paths.append(tmp_path / f"MACE{seed}.model")

    return MACECalculator(_model_paths, device="cpu")


def test_calculator_forces(fitting_configs, trained_model):
    at = fitting_configs[2].copy()
    at.calc = trained_model

    # test just forces
    grads = gradient_test(at)

    assert np.allclose(grads[0], grads[1])


def test_calculator_stress(fitting_configs, trained_model):
    at = fitting_configs[2].copy()
    at.calc = trained_model

    # test forces and stress
    at_wrapped = ExpCellFilter(at)
    grads = gradient_test(at_wrapped)

    assert np.allclose(grads[0], grads[1])


def test_calculator_committee(fitting_configs, trained_committee):
    at = fitting_configs[2].copy()
    at.calc = trained_committee

    # test just forces
    grads = gradient_test(at)

    assert np.allclose(grads[0], grads[1])

    E = at.get_potential_energy()
    energies = at.calc.results["energies"]
    energies_var = at.calc.results["energy_var"]
    forces_var = np.var(at.calc.results["forces_comm"], axis=0)
    assert np.allclose(E, np.mean(energies))
    assert np.allclose(energies_var, np.var(energies))
    assert forces_var.shape == at.calc.results["forces"].shape


def test_calculator_dipole(fitting_configs, trained_dipole_model):
    at = fitting_configs[2].copy()
    at.calc = trained_dipole_model

    dip = at.get_dipole_moment()

    assert len(dip) == 3


def test_calculator_energy_dipole(fitting_configs, trained_energy_dipole_model):
    at = fitting_configs[2].copy()
    at.calc = trained_energy_dipole_model

    grads = gradient_test(at)
    dip = at.get_dipole_moment()

    assert np.allclose(grads[0], grads[1])
    assert len(dip) == 3


def test_calculator_descriptor(fitting_configs, trained_equivariant_model):
    at = fitting_configs[2].copy()
    at.calc = trained_equivariant_model

    desc_invariant = at.calc.get_descriptors(at, invariants_only=True)
    desc_single_layer = at.calc.get_descriptors(at, invariants_only=True, num_layers=1)
    desc = at.calc.get_descriptors(at, invariants_only=False)

    assert desc_invariant.shape[0] == 3
    assert desc_invariant.shape[1] == 32
    assert desc_single_layer.shape[0] == 3
    assert desc_single_layer.shape[1] == 16
    assert desc.shape[0] == 3
    assert desc.shape[1] == 80


def test_mace_mp(capsys: pytest.CaptureFixture):
    mp_mace = mace_mp()
    assert isinstance(mp_mace, MACECalculator)
    assert mp_mace.model_type == "MACE"
    assert len(mp_mace.models) == 1
    assert isinstance(mp_mace.models[0], ScaleShiftMACE)

    stdout, stderr = capsys.readouterr()
    assert stderr == ""


def test_mace_off():
    mace_off__model = mace_off(model="small", device="cpu")
    assert isinstance(mace_off__model, MACECalculator)
    assert mace_off__model.model_type == "MACE"
    assert len(mace_off__model.models) == 1
    assert isinstance(mace_off__model.models[0], ScaleShiftMACE)

    atoms = build.molecule("H2O")
    atoms.calc = mace_off__model

    E = atoms.get_potential_energy()

    assert np.allclose(E, -2081.116128586803, atol=1e-9)


def test_mace_off_2(capsys: pytest.CaptureFixture):
    mace_off__model = mace_off(model="small", device="cpu")
    stdout, stderr = capsys.readouterr()
    assert "Downloading" not in stdout
