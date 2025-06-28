import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from ase.calculators.test import gradient_test
from ase.constraints import ExpCellFilter

from mace.calculators.mace import MACECalculator

try:
    import cuequivariance as cue  # pylint: disable=unused-import

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

run_train = "mace_run_train"


@pytest.fixture(scope="module")
def default_dtype_str():
    return "float64"


@pytest.fixture(scope="module")
def fitting_configs():
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


@pytest.fixture(scope="module")
def core_mace_params(default_dtype_str):
    return {
        "valid_fraction": 0.05,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "MACE",
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
        "default_dtype": default_dtype_str,
    }


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory, fitting_configs, core_mace_params):
    _mace_params = {
        **core_mace_params,
        "name": "MACE",
        "hidden_irreps": "128x0e",
    }

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
    mace_params["train_file"] = tmp_path / "fit.xyz"

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), check=True)

    assert p.returncode == 0

    return MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        default_dtype=_mace_params["default_dtype"],
    )


@pytest.fixture(scope="module")
def trained_equivariant_model(tmp_path_factory, fitting_configs, core_mace_params):
    _mace_params = {
        **core_mace_params,
        "name": "MACE",
        "hidden_irreps": "16x0e+16x1o",
    }

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), check=True)

    assert p.returncode == 0

    return MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        default_dtype=_mace_params["default_dtype"],
    )


@pytest.fixture(scope="module")
def trained_equivariant_model_cueq(tmp_path_factory, fitting_configs, core_mace_params):
    _mace_params = {
        **core_mace_params,
        "name": "MACE",
        "hidden_irreps": "16x0e+16x1o",
    }

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), check=True)

    assert p.returncode == 0

    return MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        enable_cueq=True,
        default_dtype=_mace_params["default_dtype"],
    )


@pytest.fixture(scope="module")
def trained_dipole_model(tmp_path_factory, fitting_configs, default_dtype_str):
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
        "default_dtype": default_dtype_str,
    }

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), check=True)

    assert p.returncode == 0

    return MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        model_type="DipoleMACE",
        default_dtype=default_dtype_str,
    )


@pytest.fixture(scope="module")
def trained_energy_dipole_model(tmp_path_factory, fitting_configs, default_dtype_str):
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
        "default_dtype": default_dtype_str,
    }

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), check=True)

    assert p.returncode == 0

    return MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        model_type="EnergyDipoleMACE",
        default_dtype=default_dtype_str,
    )


@pytest.fixture(scope="module")
def trained_committee(tmp_path_factory, fitting_configs, core_mace_params):
    _seeds = [5, 6, 7]
    _model_paths = []
    for seed in _seeds:
        _mace_params = {
            **core_mace_params,
            "name": f"MACE{seed}",
            "hidden_irreps": "16x0e",
        }

        tmp_path = tmp_path_factory.mktemp(f"run{seed}_")

        ase.io.write(tmp_path / "fit.xyz", fitting_configs)

        mace_params = _mace_params.copy()
        mace_params["checkpoints_dir"] = str(tmp_path)
        mace_params["model_dir"] = str(tmp_path)
        mace_params["train_file"] = tmp_path / "fit.xyz"

        cmd = (
            run_train
            + " "
            + " ".join(
                [
                    (f"--{k}={v}" if v is not None else f"--{k}")
                    for k, v in mace_params.items()
                ]
            )
        )

        p = subprocess.run(cmd.split(), check=True)

        assert p.returncode == 0

        _model_paths.append(tmp_path / f"MACE{seed}.model")

    return MACECalculator(
        model_paths=_model_paths,
        device="cpu",
        default_dtype=_mace_params["default_dtype"],
    )


@pytest.mark.parametrize("test_dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_calculator_node_energy(fitting_configs, trained_model, test_dtype):
    trained_model.to(dtype=test_dtype)
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
        np.testing.assert_allclose(energy, energy_via_nodes, atol=1e-7)


@pytest.mark.parametrize("test_dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_calculator_forces(fitting_configs, trained_model, test_dtype):
    at = fitting_configs[2].copy()
    at.calc = trained_model.to(dtype=test_dtype)

    # test just forces
    eps_max = 1e-8 if test_dtype == torch.float64 else 1e-4
    atol = 1e-7 if test_dtype == torch.float64 else 1e-3
    grads = gradient_test(at, eps_max=eps_max)

    np.testing.assert_allclose(grads[0], grads[1], atol=atol)


@pytest.mark.parametrize("test_dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_calculator_stress(fitting_configs, trained_model, test_dtype):
    at = fitting_configs[2].copy()
    at.calc = trained_model.to(dtype=test_dtype)

    # test forces and stress
    at_wrapped = ExpCellFilter(at)
    eps_max = 1e-8 if test_dtype == torch.float64 else 1e-4
    atol = 1e-7 if test_dtype == torch.float64 else 1e-3
    grads = gradient_test(at_wrapped, eps_max=eps_max)

    np.testing.assert_allclose(grads[0], grads[1], atol=atol)


@pytest.mark.parametrize("test_dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_calculator_committee(fitting_configs, trained_committee, test_dtype):
    at = fitting_configs[2].copy()
    at.calc = trained_committee.to(dtype=test_dtype)

    E = at.get_potential_energy()
    energies = at.calc.results["energies"]
    energies_var = at.calc.results["energy_var"]
    np.testing.assert_allclose(E, np.mean(energies))
    np.testing.assert_allclose(energies_var, np.var(energies))

    # test just forces
    eps_max = 1e-8 if test_dtype == torch.float64 else 1e-4
    atol = 1e-7 if test_dtype == torch.float64 else 1e-3
    grads = gradient_test(at, eps_max=eps_max)
    np.testing.assert_allclose(grads[0], grads[1], atol=atol)
    forces_var = np.var(at.calc.results["forces_comm"], axis=0)
    assert forces_var.shape == at.calc.results["forces"].shape


@pytest.mark.parametrize("test_dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_calculator_from_model(fitting_configs, trained_committee, test_dtype):
    # test single model
    test_calculator_forces(
        fitting_configs,
        trained_model=MACECalculator(models=trained_committee.models[0], device="cpu"),
        test_dtype=test_dtype,
    )

    # test committee model
    test_calculator_committee(
        fitting_configs,
        trained_committee=MACECalculator(models=trained_committee.models, device="cpu"),
        test_dtype=test_dtype,
    )


@pytest.mark.parametrize("test_dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_calculator_dipole(fitting_configs, trained_dipole_model, test_dtype):
    at = fitting_configs[2].copy()
    at.calc = trained_dipole_model.to(dtype=test_dtype)

    dip = at.get_dipole_moment()

    assert len(dip) == 3


@pytest.mark.parametrize("test_dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_calculator_energy_dipole(fitting_configs, trained_energy_dipole_model, test_dtype):
    at = fitting_configs[2].copy()
    at.calc = trained_energy_dipole_model.to(dtype=test_dtype)

    eps_max = 1e-8 if test_dtype == torch.float64 else 1e-4
    # NOTE: unexplained why larger atol needed here for fp32 than other models.
    atol = 1e-7 if test_dtype == torch.float64 else 5e-3
    grads = gradient_test(at, eps_max=eps_max)
    dip = at.get_dipole_moment()

    np.testing.assert_allclose(grads[0], grads[1], atol=atol)
    assert len(dip) == 3


@pytest.mark.parametrize("test_dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("test_model", ["trained_equivariant_model"] + (["trained_equivariant_model_cueq"] if CUET_AVAILABLE else []))
def test_calculator_descriptor(fitting_configs, test_model, test_dtype, request):
    at = fitting_configs[2].copy()
    at_rotated = fitting_configs[2].copy()
    at_rotated.rotate(90, "x")
    calc = request.getfixturevalue(test_model).to(dtype=test_dtype)

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
