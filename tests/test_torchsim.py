"""Tests for the MACE TorchSim model interface."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import build

try:
    import torch_sim as ts

    TORCHSIM_AVAILABLE = True
except ImportError:
    TORCHSIM_AVAILABLE = False

try:
    import cuequivariance as cue  # noqa: F401

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCHSIM_AVAILABLE, reason="torch-sim not installed"
)

pytest_mace_dir = Path(__file__).parent.parent
run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory):
    """Train a minimal MACE model and return the path to the model file."""
    from ase.atoms import Atoms

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

    np.random.seed(42)
    for _ in range(10):
        c = water.copy()
        c.positions += np.random.normal(0.1, size=c.positions.shape)
        c.info["REF_energy"] = np.random.normal(0.1)
        c.new_array("REF_forces", np.random.normal(0.1, size=c.positions.shape))
        c.info["REF_stress"] = np.random.normal(0.1, size=6)
        fit_configs.append(c)

    tmp_path = tmp_path_factory.mktemp("torchsim_model_")
    import ase.io

    ase.io.write(tmp_path / "fit.xyz", fit_configs)

    mace_params = {
        "name": "MACE",
        "valid_fraction": 0.05,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "MACE",
        "hidden_irreps": "32x0e",
        "r_max": 3.5,
        "batch_size": 5,
        "max_num_epochs": 5,
        "device": "cpu",
        "seed": 42,
        "loss": "stress",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
        "eval_interval": 2,
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "train_file": str(tmp_path / "fit.xyz"),
    }

    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)

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
    return tmp_path / "MACE.model"


@pytest.fixture(scope="module")
def water_atoms():
    from ase.atoms import Atoms

    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    return atoms


def test_torchsim_basic(trained_model_path, water_atoms):
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    model = MaceTorchSimModel(
        model=trained_model_path,
        device=torch.device("cpu"),
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=True,
    )

    state = ts.io.atoms_to_state(
        water_atoms, device=torch.device("cpu"), dtype=torch.float64
    )

    results = model(state)
    assert "energy" in results
    assert "forces" in results
    assert "stress" in results
    assert results["energy"].shape == (1,)
    assert results["forces"].shape[0] == len(water_atoms)
    assert results["forces"].shape[1] == 3


def test_torchsim_no_stress(trained_model_path, water_atoms):
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    model = MaceTorchSimModel(
        model=trained_model_path,
        device=torch.device("cpu"),
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=False,
    )

    state = ts.io.atoms_to_state(
        water_atoms, device=torch.device("cpu"), dtype=torch.float64
    )

    results = model(state)
    assert "energy" in results
    assert "forces" in results


def test_torchsim_matches_ase_calculator(trained_model_path, water_atoms):
    from ase.stress import full_3x3_to_voigt_6_stress

    from mace.calculators.mace import MACECalculator
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    ase_calc = MACECalculator(
        model_paths=trained_model_path, device="cpu", default_dtype="float64"
    )
    atoms_ase = water_atoms.copy()
    atoms_ase.calc = ase_calc
    ase_energy = atoms_ase.get_potential_energy()
    ase_forces = atoms_ase.get_forces()
    ase_stress = atoms_ase.get_stress()

    ts_model = MaceTorchSimModel(
        model=trained_model_path,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    state = ts.io.atoms_to_state(
        water_atoms, device=torch.device("cpu"), dtype=torch.float64
    )
    ts_results = ts_model(state)

    np.testing.assert_allclose(
        ts_results["energy"].item(), ase_energy, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        ts_results["forces"].detach().cpu().numpy(), ase_forces, atol=1e-5, rtol=1e-5
    )
    ts_stress_voigt = full_3x3_to_voigt_6_stress(
        ts_results["stress"].detach().cpu().numpy().reshape(3, 3)
    )
    np.testing.assert_allclose(ts_stress_voigt, ase_stress, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_torchsim_cueq(trained_model_path, water_atoms):
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    model = MaceTorchSimModel(
        model=trained_model_path,
        device=torch.device("cpu"),
        dtype=torch.float64,
        enable_cueq=True,
    )

    state = ts.io.atoms_to_state(
        water_atoms, device=torch.device("cpu"), dtype=torch.float64
    )

    results = model(state)
    assert "energy" in results
    assert "forces" in results


def test_torchsim_batched(trained_model_path, water_atoms):
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    w1 = water_atoms.copy()
    w2 = water_atoms.copy()
    w2.positions += np.random.RandomState(0).normal(0.01, size=w2.positions.shape)

    model = MaceTorchSimModel(
        model=trained_model_path,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    state = ts.io.atoms_to_state(
        [w1, w2], device=torch.device("cpu"), dtype=torch.float64
    )

    results = model(state)
    assert results["energy"].shape == (2,)
    assert results["forces"].shape == (len(w1) + len(w2), 3)
