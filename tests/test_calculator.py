import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from ase import build
from ase.atoms import Atoms
from ase.calculators.test import gradient_test
from ase.constraints import ExpCellFilter

from mace.calculators import mace_mp, mace_off
from mace.calculators.foundations_models import mace_omol
from mace.calculators.mace import MACECalculator
from mace.modules.models import ScaleShiftMACE

try:
    import cuequivariance as cue  # pylint: disable=unused-import

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

pytest_mace_dir = Path(__file__).parent.parent
run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"


def write_extxyz_test(tmp_path, atoms):
    assert isinstance(atoms, Atoms), "write_extxyz_test only working for Atoms, not anything else such as list(Atoms)"
    ase.io.write(tmp_path / "test.extxyz", atoms)
    atoms_written = ase.io.read(tmp_path / "test.extxyz")

    nonstd_fields = set(['node_energy', 'energy_var', 'energy_comm', 'stress_var', 'stress_comm', 'forces_var', 'forces_comm', 'virials'])
    # everything that we expect has been written
    assert set(atoms.calc.results.keys()) - nonstd_fields == set(atoms_written.calc.results.keys())
    # everything that was written was correct
    assert all(
        np.allclose(atoms.calc.results[k], atoms_written.calc.results[k])
        for k in atoms_written.calc.results
    )


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
    fit_configs[0].info["REF_energy"] = 1.0
    fit_configs[0].info["config_type"] = "IsolatedAtom"
    fit_configs[1].info["REF_energy"] = -0.5
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
        "eval_interval": 2,
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

    return MACECalculator(model_paths=tmp_path / "MACE.model", device="cpu")


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
        "eval_interval": 2,
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

    return MACECalculator(model_paths=tmp_path / "MACE.model", device="cpu")


@pytest.fixture(scope="module", name="trained_equivariant_model_cueq")
def trained_model_equivariant_fixture_cueq(tmp_path_factory, fitting_configs):
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
        "eval_interval": 2,
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

    model = torch.load(tmp_path / "MACE.model", map_location="cpu")
    print("DEBUG model", model)
    return MACECalculator(
        model_paths=tmp_path / "MACE.model", device="cpu", enable_cueq=True
    )


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
        "eval_interval": 2,
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
        model_paths=tmp_path / "MACE.model", device="cpu", model_type="DipoleMACE"
    )


@pytest.fixture(scope="module", name="trained_dipole_polarizability_model")
def trained_dipole_polar_fixture(tmp_path_factory, fitting_configs):
    _mace_params = {
        "name": "MACE",
        "valid_fraction": 0.05,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "AtomicDielectricMACE",
        "num_channels": 8,
        "max_L": 2,
        "r_max": 3.5,
        "batch_size": 5,
        "max_num_epochs": 10,
        "MLP_irreps": "16x0e+16x1o+16x2e",
        "ema": None,
        "ema_decay": 0.99,
        "amsgrad": None,
        "restart_latest": None,
        "device": "cpu",
        "seed": 5,
        "loss": "dipole_polar",
        "energy_key": "",
        "forces_key": "",
        "stress_key": "",
        "dipole_key": "REF_dipole",
        "polarizability_key": "REF_polarizability",
        "error_table": "DipolePolarRMSE",
        "eval_interval": 2,
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
        tmp_path / "MACE.model", device="cpu", model_type="DipolePolarizabilityMACE"
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
        "eval_interval": 2,
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
        model_paths=tmp_path / "MACE.model", device="cpu", model_type="EnergyDipoleMACE"
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
            "eval_interval": 2,
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

    return MACECalculator(model_paths=_model_paths, device="cpu")


def test_calculator_node_energy(fitting_configs, trained_model):
    for at in fitting_configs:
        trained_model.calculate(at)
        node_energies = trained_model.results["node_energy"]
        batch = trained_model._atoms_to_batch(at)  # pylint: disable=protected-access
        node_heads = batch["head"][batch["batch"]]
        num_atoms_arange = torch.arange(batch["positions"].shape[0])
        node_e0 = (
            trained_model.models[0].atomic_energies_fn(batch["node_attrs"]).detach()
        )
        node_e0 = node_e0[num_atoms_arange, node_heads].cpu().numpy()
        energy_via_nodes = np.sum(node_energies + node_e0)
        energy = trained_model.results["energy"]
        np.testing.assert_allclose(energy, energy_via_nodes, atol=1e-6)


def test_calculator_forces(tmp_path, fitting_configs, trained_model):
    at = fitting_configs[2].copy()
    at.calc = trained_model

    # test just forces
    grads = gradient_test(at)

    assert np.allclose(grads[0], grads[1])
    write_extxyz_test(tmp_path, at)


def test_calculator_stress(tmp_path, fitting_configs, trained_model):
    at = fitting_configs[2].copy()
    at.calc = trained_model

    # test forces and stress
    at_wrapped = ExpCellFilter(at)
    grads = gradient_test(at_wrapped)

    assert np.allclose(grads[0], grads[1])
    write_extxyz_test(tmp_path, at)


def test_calculator_committee(tmp_path, fitting_configs, trained_committee):
    at = fitting_configs[2].copy()
    at.calc = trained_committee

    # test just forces
    grads = gradient_test(at)

    assert np.allclose(grads[0], grads[1])

    E = at.get_potential_energy()
    energies = at.calc.results["energy_comm"]
    energies_var = at.calc.results["energy_var"]
    forces_var = np.var(at.calc.results["forces_comm"], axis=0)
    assert np.allclose(E, np.mean(energies))
    assert np.allclose(energies_var, np.var(energies))
    assert forces_var.shape == at.calc.results["forces"].shape
    write_extxyz_test(tmp_path, at)


def test_calculator_from_model(tmp_path, fitting_configs, trained_committee):
    # test single model
    test_calculator_forces(
        tmp_path,
        fitting_configs,
        trained_model=MACECalculator(models=trained_committee.models[0], device="cpu"),
    )

    # test committee model
    test_calculator_committee(
        tmp_path,
        fitting_configs,
        trained_committee=MACECalculator(models=trained_committee.models, device="cpu"),
    )


def test_calculator_dipole(tmp_path, fitting_configs, trained_dipole_model):
    at = fitting_configs[2].copy()
    at.calc = trained_dipole_model

    dip = at.get_dipole_moment()

    assert len(dip) == 3
    write_extxyz_test(tmp_path, at)


def test_calculator_energy_dipole(tmp_path, fitting_configs, trained_energy_dipole_model):
    at = fitting_configs[2].copy()
    at.calc = trained_energy_dipole_model

    grads = gradient_test(at)
    dip = at.get_dipole_moment()

    assert np.allclose(grads[0], grads[1])
    assert len(dip) == 3
    write_extxyz_test(tmp_path, at)


def test_calculator_descriptor(fitting_configs, trained_equivariant_model):
    at = fitting_configs[2].copy()
    at_rotated = fitting_configs[2].copy()
    at_rotated.rotate(90, "x")
    calc = trained_equivariant_model

    desc_invariant = calc.get_descriptors(at, invariants_only=True)
    desc_invariant_rotated = calc.get_descriptors(at_rotated, invariants_only=True)
    desc_invariant_single_layer = calc.get_descriptors(
        at, invariants_only=True, num_layers=1
    )
    desc_invariant_single_layer_rotated = calc.get_descriptors(
        at_rotated, invariants_only=True, num_layers=1
    )
    desc = calc.get_descriptors(at, invariants_only=False)
    desc_single_layer = calc.get_descriptors(at, invariants_only=False, num_layers=1)
    desc_rotated = calc.get_descriptors(at_rotated, invariants_only=False)
    desc_rotated_single_layer = calc.get_descriptors(
        at_rotated, invariants_only=False, num_layers=1
    )

    assert desc_invariant.shape[0] == 3
    assert desc_invariant.shape[1] == 32
    assert desc_invariant_single_layer.shape[0] == 3
    assert desc_invariant_single_layer.shape[1] == 16
    assert desc.shape[0] == 3
    assert desc.shape[1] == 80
    assert desc_single_layer.shape[0] == 3
    assert desc_single_layer.shape[1] == 16 * 4
    assert desc_rotated_single_layer.shape[0] == 3
    assert desc_rotated_single_layer.shape[1] == 16 * 4

    np.testing.assert_allclose(desc_invariant, desc_invariant_rotated, atol=1e-6)
    np.testing.assert_allclose(
        desc_invariant_single_layer, desc_invariant[:, :16], atol=1e-6
    )
    np.testing.assert_allclose(
        desc_invariant_single_layer_rotated, desc_invariant[:, :16], atol=1e-6
    )
    np.testing.assert_allclose(
        desc_single_layer[:, :16], desc_rotated_single_layer[:, :16], atol=1e-6
    )
    assert not np.allclose(
        desc_single_layer[:, 16:], desc_rotated_single_layer[:, 16:], atol=1e-6
    )
    assert not np.allclose(desc, desc_rotated, atol=1e-6)


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_calculator_descriptor_cueq(fitting_configs, trained_equivariant_model_cueq):
    at = fitting_configs[2].copy()
    at_rotated = fitting_configs[2].copy()
    at_rotated.rotate(90, "x")
    calc = trained_equivariant_model_cueq
    print("model", calc.models[0])

    desc_invariant = calc.get_descriptors(at, invariants_only=True)
    desc_invariant_rotated = calc.get_descriptors(at_rotated, invariants_only=True)
    desc_invariant_single_layer = calc.get_descriptors(
        at, invariants_only=True, num_layers=1
    )
    desc_invariant_single_layer_rotated = calc.get_descriptors(
        at_rotated, invariants_only=True, num_layers=1
    )
    desc = calc.get_descriptors(at, invariants_only=False)
    desc_single_layer = calc.get_descriptors(at, invariants_only=False, num_layers=1)
    desc_rotated = calc.get_descriptors(at_rotated, invariants_only=False)
    desc_rotated_single_layer = calc.get_descriptors(
        at_rotated, invariants_only=False, num_layers=1
    )

    assert desc_invariant.shape[0] == 3
    assert desc_invariant.shape[1] == 32
    assert desc_invariant_single_layer.shape[0] == 3
    assert desc_invariant_single_layer.shape[1] == 16
    assert desc.shape[0] == 3
    assert desc.shape[1] == 80
    assert desc_single_layer.shape[0] == 3
    assert desc_single_layer.shape[1] == 16 * 4
    assert desc_rotated_single_layer.shape[0] == 3
    assert desc_rotated_single_layer.shape[1] == 16 * 4

    np.testing.assert_allclose(desc_invariant, desc_invariant_rotated, atol=1e-6)
    np.testing.assert_allclose(
        desc_invariant_single_layer, desc_invariant[:, :16], atol=1e-6
    )
    np.testing.assert_allclose(
        desc_invariant_single_layer_rotated, desc_invariant[:, :16], atol=1e-6
    )
    np.testing.assert_allclose(
        desc_single_layer[:, :16], desc_rotated_single_layer[:, :16], atol=1e-6
    )
    assert not np.allclose(
        desc_single_layer[:, 16:], desc_rotated_single_layer[:, 16:], atol=1e-6
    )
    assert not np.allclose(desc, desc_rotated, atol=1e-6)


def test_mace_mp(tmp_path, capsys: pytest.CaptureFixture):
    mp_mace = mace_mp()
    assert isinstance(mp_mace, MACECalculator)
    assert mp_mace.model_type == "MACE"
    assert len(mp_mace.models) == 1
    assert isinstance(mp_mace.models[0], ScaleShiftMACE)

    _, stderr = capsys.readouterr()
    assert stderr == ""


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
