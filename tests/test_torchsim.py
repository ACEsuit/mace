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


# ---------------------------------------------------------------------------
# PolarMACE tests
# ---------------------------------------------------------------------------

try:
    import graph_longrange  # noqa: F401

    GRAPH_LONGRANGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GRAPH_LONGRANGE_AVAILABLE = False

POLAR_MODEL_NAME = "polar-1-s"
DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _atoms_state(atoms):
    return ts.io.atoms_to_state(atoms, device=DEVICE, dtype=DTYPE)


def _skip_if_polar_unavailable(exc, model_name):
    msg = str(exc).lower()
    if "no such file" in msg or "not found" in msg or "download" in msg:
        pytest.skip(f"Missing Polar foundation model file: {model_name}")
    raise exc


@pytest.fixture(scope="module")
def polar_raw_model():
    """Load the smallest pre-trained PolarMACE foundation model."""
    if not GRAPH_LONGRANGE_AVAILABLE:
        pytest.skip("graph_longrange is not installed")
    from mace.calculators.foundations_models import mace_polar

    try:
        return mace_polar(
            model=POLAR_MODEL_NAME, device=DEVICE.type, return_raw_model=True
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _skip_if_polar_unavailable(exc, POLAR_MODEL_NAME)


@pytest.fixture(scope="module")
def water_state(water_atoms):
    """SimState for a single water molecule (no extras)."""
    return _atoms_state(water_atoms)


@pytest.fixture(scope="module")
def water_state_with_extras(water_state):
    """SimState with polar-relevant extras set."""
    state = water_state.clone()
    state.external_E_field = torch.tensor([[0.1, 0.0, 0.0]], dtype=DTYPE)
    state.charge = torch.tensor([0.0], dtype=DTYPE)
    state.spin = torch.tensor([1.0], dtype=DTYPE)
    return state


@pytest.fixture(scope="module")
def water_batched_state(water_atoms):
    """Batched SimState with 2 water molecules."""
    rng = np.random.default_rng(seed=0)
    w1, w2 = water_atoms.copy(), water_atoms.copy()
    w2.positions += rng.normal(0.01, size=w2.positions.shape)
    return _atoms_state([w1, w2])


@pytest.fixture(scope="module")
def water_batched_state_with_extras(water_batched_state):
    """Batched SimState with polar-relevant extras."""
    state = water_batched_state.clone()
    state.external_E_field = torch.tensor(
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=DTYPE
    )
    state.charge = torch.tensor([0.0, 0.0], dtype=DTYPE)
    state.spin = torch.tensor([1.0, 1.0], dtype=DTYPE)
    return state


def test_torchsim_polar_basic(polar_raw_model, water_state):
    """Forward pass with PolarMACE using defaults (no extras)."""
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    model = MaceTorchSimModel(
        model=polar_raw_model,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )

    results = model(water_state)
    assert results["energy"].shape == (1,)
    assert results["forces"].shape == (3, 3)
    assert results["stress"].shape == (1, 3, 3)
    assert "charges" in results
    assert "dipole" in results
    assert "density_coefficients" in results


def test_torchsim_polar_with_extras(polar_raw_model, water_state_with_extras):
    """Forward pass with PolarMACE using explicit extras."""
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    model = MaceTorchSimModel(
        model=polar_raw_model,
        device=DEVICE,
        dtype=DTYPE,
    )

    results = model(water_state_with_extras)
    assert results["energy"].shape == (1,)
    assert "charges" in results
    assert "dipole" in results
    assert "density_coefficients" in results


def test_torchsim_polar_no_extras_vs_zero_extras(polar_raw_model, water_state):
    """Defaults (no extras) should match explicitly passing zeros."""
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    model = MaceTorchSimModel(
        model=polar_raw_model,
        device=DEVICE,
        dtype=DTYPE,
    )

    results_no_extras = model(water_state)

    state_zero_extras = water_state.clone()
    state_zero_extras.external_E_field = torch.zeros(1, 3, dtype=DTYPE)
    results_zero_extras = model(state_zero_extras)

    np.testing.assert_allclose(
        results_no_extras["energy"].detach().cpu().numpy(),
        results_zero_extras["energy"].detach().cpu().numpy(),
        atol=1e-10,
    )


def test_torchsim_polar_batched(polar_raw_model, water_batched_state):
    """Batched PolarMACE forward pass."""
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    model = MaceTorchSimModel(
        model=polar_raw_model,
        device=DEVICE,
        dtype=DTYPE,
    )

    results = model(water_batched_state)
    assert results["energy"].shape == (2,)
    assert results["forces"].shape == (6, 3)
    assert "dipole" in results
    assert results["dipole"].shape[0] == 2


def test_torchsim_polar_batched_with_extras(
    polar_raw_model, water_batched_state_with_extras
):
    """Batched PolarMACE with per-system extras."""
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    model = MaceTorchSimModel(
        model=polar_raw_model,
        device=DEVICE,
        dtype=DTYPE,
    )

    results = model(water_batched_state_with_extras)
    assert results["energy"].shape == (2,)
    assert results["forces"].shape == (6, 3)
    assert "dipole" in results
    assert results["dipole"].shape[0] == 2


def test_torchsim_polar_matches_ase(polar_raw_model, water_atoms):
    """PolarMACE TorchSim results should match ASE calculator."""
    from ase.stress import full_3x3_to_voigt_6_stress

    from mace.calculators.foundations_models import mace_polar
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    try:
        ase_calc = mace_polar(
            model=POLAR_MODEL_NAME, device=DEVICE.type, default_dtype="float64"
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _skip_if_polar_unavailable(exc, POLAR_MODEL_NAME)

    # ASE's AtomicData.from_config() defaults total_spin to 1.0; SimState
    # defaults spin to 0.0. Pin both sides to 1.0 so they match.
    atoms_ase = water_atoms.copy()
    atoms_ase.info["spin"] = 1.0
    atoms_ase.calc = ase_calc
    ase_energy = atoms_ase.get_potential_energy()
    ase_forces = atoms_ase.get_forces()
    ase_stress = atoms_ase.get_stress()

    ts_model = MaceTorchSimModel(
        model=polar_raw_model,
        device=DEVICE,
        dtype=DTYPE,
    )
    state = _atoms_state(water_atoms)
    state.spin = torch.tensor([1.0], dtype=DTYPE)
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
