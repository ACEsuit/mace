"""Tests for the MACE TorchSim model interface."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mace.calculators import mace_mp, mace_off

try:
    import torch_sim as ts
    from torch_sim.models.interface import validate_model_outputs
    from torch_sim.testing import (
        SIMSTATE_BULK_GENERATORS,
        SIMSTATE_MOLECULE_GENERATORS,
        assert_model_calculator_consistency,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("Skipping torch-sim tests due to ImportError", allow_module_level=True)

try:
    import cuequivariance as cue  # noqa: F401

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

from mace.calculators.mace_torchsim import MaceTorchSimModel

pytest_mace_dir = Path(__file__).parent.parent
run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"

DEVICE = torch.device("cpu")
DTYPE = torch.float64
MACE_MP_MODEL = "small-0b"
MACE_OFF_MODEL = "small"


def _to_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise ValueError(f"Unsupported dtype {dtype}")


# ---------------------------------------------------------------------------
# Fixtures for foundation-model tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_mace_mp_model():
    return mace_mp(
        model=MACE_MP_MODEL,
        device=str(DEVICE),
        default_dtype=_to_dtype_name(DTYPE),
        return_raw_model=True,
    )


@pytest.fixture(scope="module")
def raw_mace_off_model():
    return mace_off(
        model=MACE_OFF_MODEL,
        device=str(DEVICE),
        default_dtype=_to_dtype_name(DTYPE),
        return_raw_model=True,
    )


@pytest.fixture
def ase_mace_mp_calculator():
    return mace_mp(
        model=MACE_MP_MODEL,
        device=str(DEVICE),
        default_dtype=_to_dtype_name(DTYPE),
        dispersion=False,
    )


@pytest.fixture
def ase_mace_off_calculator():
    return mace_off(
        model=MACE_OFF_MODEL,
        device=str(DEVICE),
        default_dtype=_to_dtype_name(DTYPE),
    )


@pytest.fixture
def ts_mace_mp_model(raw_mace_mp_model):
    return MaceTorchSimModel(
        model=raw_mace_mp_model,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.fixture
def ts_mace_off_model(raw_mace_off_model):
    return MaceTorchSimModel(
        model=raw_mace_off_model,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=False,
    )


# ---------------------------------------------------------------------------
# Fixtures for locally-trained-model tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Foundation-model tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sim_state_name", ("si_sim_state", "rattled_si_sim_state"))
def test_torch_sim_mace_mp_consistency(
    sim_state_name, ts_mace_mp_model, ase_mace_mp_calculator
):
    sim_state = SIMSTATE_BULK_GENERATORS[sim_state_name](DEVICE, DTYPE)
    assert_model_calculator_consistency(
        model=ts_mace_mp_model,
        calculator=ase_mace_mp_calculator,
        sim_state=sim_state,
    )


@pytest.mark.parametrize("sim_state_name", ("benzene_sim_state",))
def test_torch_sim_mace_off_consistency(
    sim_state_name, ts_mace_off_model, ase_mace_off_calculator
):
    sim_state = SIMSTATE_MOLECULE_GENERATORS[sim_state_name](DEVICE, DTYPE)
    assert_model_calculator_consistency(
        model=ts_mace_off_model,
        calculator=ase_mace_off_calculator,
        sim_state=sim_state,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_torch_sim_mace_dtype_smoke(raw_mace_mp_model, dtype: torch.dtype):
    model = MaceTorchSimModel(
        model=raw_mace_mp_model,
        device=DEVICE,
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
    )
    state = SIMSTATE_BULK_GENERATORS["si_sim_state"](DEVICE, dtype)
    output = model(state)

    assert output["energy"].shape == (1,)
    assert torch.is_floating_point(output["energy"])
    assert output["forces"].shape == state.positions.shape
    assert torch.is_floating_point(output["forces"])
    assert output["stress"].shape == (1, 3, 3)


def test_torch_sim_mace_off_output_keys(ts_mace_off_model):
    state = SIMSTATE_MOLECULE_GENERATORS["benzene_sim_state"](DEVICE, DTYPE)
    output = ts_mace_off_model(state)
    assert "energy" in output
    assert "forces" in output
    assert "stress" not in output


def test_torch_sim_mace_validate_outputs(ts_mace_mp_model):
    validate_model_outputs(ts_mace_mp_model, DEVICE, DTYPE)


# ---------------------------------------------------------------------------
# Locally-trained-model tests
# ---------------------------------------------------------------------------


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
