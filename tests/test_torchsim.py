"""Tests for the MACE TorchSim model interface."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mace.calculators import mace_mp, mace_off, mace_omol

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

from mace.calculators.mace import MACECalculator
from mace.calculators.mace_torchsim import MaceTorchSimModel

run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"
DEVICE = torch.device("cpu")
DTYPE = torch.float64
MACE_MP_MODEL = "small-0b"
MACE_OFF_MODEL = "small"
SKIP_OMOL_DOWNLOAD = os.getenv("CI", "").lower() in {"1", "true", "yes"}


def _to_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise ValueError(f"Unsupported dtype {dtype}")


def _atoms_state(atoms, device=DEVICE, dtype=DTYPE):
    return ts.io.atoms_to_state(atoms, device=device, dtype=dtype)


@pytest.fixture(scope="module")
def raw_mace_mp_model():
    return mace_mp(
        model=MACE_MP_MODEL,
        device=DEVICE.type,
        default_dtype=_to_dtype_name(DTYPE),
        return_raw_model=True,
    )


@pytest.fixture(scope="module")
def raw_mace_off_model():
    return mace_off(
        model=MACE_OFF_MODEL,
        device=DEVICE.type,
        default_dtype=_to_dtype_name(DTYPE),
        return_raw_model=True,
    )


@pytest.fixture
def ase_mace_mp_calculator():
    return mace_mp(
        model=MACE_MP_MODEL,
        device=DEVICE.type,
        default_dtype=_to_dtype_name(DTYPE),
        dispersion=False,
    )


@pytest.fixture
def ase_mace_off_calculator():
    return mace_off(
        model=MACE_OFF_MODEL,
        device=DEVICE.type,
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


@pytest.fixture(scope="module")
def raw_mace_omol_model():
    if SKIP_OMOL_DOWNLOAD:
        pytest.skip("MACE-OMOL checkpoint is large; skip in CI.")
    return mace_omol(
        device=DEVICE.type,
        default_dtype=_to_dtype_name(DTYPE),
        return_raw_model=True,
    )


@pytest.fixture
def ase_mace_omol_calculator():
    if SKIP_OMOL_DOWNLOAD:
        pytest.skip("MACE-OMOL checkpoint is large; skip in CI.")
    return mace_omol(
        device=DEVICE.type,
        default_dtype=_to_dtype_name(DTYPE),
    )


@pytest.fixture
def ts_mace_omol_model(raw_mace_omol_model):
    return MaceTorchSimModel(
        model=raw_mace_omol_model,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=False,
        head="omol",
    )


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory):
    """Train a minimal MACE model and return the path to the model file."""
    import ase.io
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
def test_torch_sim_mace_off_consistency(sim_state_name, ts_mace_off_model, ase_mace_off_calculator):
    sim_state = SIMSTATE_MOLECULE_GENERATORS[sim_state_name](DEVICE, DTYPE)
    assert_model_calculator_consistency(
        model=ts_mace_off_model,
        calculator=ase_mace_off_calculator,
        sim_state=sim_state,
    )
    assert "stress" not in ts_mace_off_model(sim_state)


@pytest.mark.parametrize("sim_state_name", ("benzene_sim_state",))
def test_torch_sim_mace_omol_consistency(sim_state_name, ts_mace_omol_model, ase_mace_omol_calculator):
    sim_state = SIMSTATE_MOLECULE_GENERATORS[sim_state_name](DEVICE, DTYPE)
    ion = sim_state.clone()
    ion.charge[0] = 1.0
    ion.spin[0] = 3.0
    for state in (sim_state, ion):
        assert_model_calculator_consistency(
            model=ts_mace_omol_model,
            calculator=ase_mace_omol_calculator,
            sim_state=state,
        )
        assert "stress" not in ts_mace_omol_model(state)


def test_torch_sim_mace_validate_outputs(ts_mace_mp_model):
    validate_model_outputs(ts_mace_mp_model, DEVICE, DTYPE)


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


def test_torchsim_no_stress(trained_model_path, water_atoms):
    model = MaceTorchSimModel(
        model=trained_model_path,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=False,
    )
    results = model(_atoms_state(water_atoms))
    assert "energy" in results
    assert "forces" in results
    assert "stress" not in results


def test_torchsim_matches_ase_calculator(trained_model_path, water_atoms):
    ase_calc = MACECalculator(
        model_paths=trained_model_path, device=DEVICE.type, default_dtype="float64"
    )
    ts_model = MaceTorchSimModel(
        model=trained_model_path,
        device=DEVICE,
        dtype=DTYPE,
    )
    sim_state = _atoms_state(water_atoms)
    assert_model_calculator_consistency(
        model=ts_model,
        calculator=ase_calc,
        sim_state=sim_state,
    )


def test_torchsim_buffers_reused(raw_mace_mp_model):
    """Buffers returned by _fill_padded_data must be the same objects across calls.
    Dynamic allocation would break torch.compile / CUDA graphs.
    We force the buffer path without torch.compile to avoid e3nn Dynamo issues."""
    model = MaceTorchSimModel(
        model=raw_mace_mp_model,
        device=DEVICE,
        dtype=torch.float32,
    )
    model._use_compile = True
    state = SIMSTATE_BULK_GENERATORS["si_sim_state"](DEVICE, torch.float32)
    state = state.from_state(
        state,
        charge=torch.zeros(state.n_systems, device=DEVICE, dtype=torch.float32),
        spin=torch.ones(state.n_systems, device=DEVICE, dtype=torch.float32),
    )
    _BUF_NAMES = (
        "_buf_node_attrs",
        "_buf_batch",
        "_buf_edge_index",
        "_buf_shifts",
        "_buf_unit_shifts",
        "_buf_ptr",
        "_buf_cell",
        "_buf_head",
        "_buf_total_charge",
        "_buf_total_spin",
    )
    def _buf_ptrs(m):
        return {
            name: getattr(m, name).data_ptr()
            for name in _BUF_NAMES
            if getattr(m, name, None) is not None
        }
    out1 = model(state)
    ptrs1 = _buf_ptrs(model)
    _ = model(state)
    ptrs2 = _buf_ptrs(model)
    out3 = model(state)
    ptrs3 = _buf_ptrs(model)
    assert ptrs1 == ptrs2 == ptrs3, "buffers were re-allocated between calls"
    for key in _BUF_NAMES:
        assert key in ptrs1, f"buffer {key} was never allocated"
    np.testing.assert_allclose(
        out1["energy"].detach().cpu().numpy(),
        out3["energy"].detach().cpu().numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        out1["forces"].detach().cpu().numpy(),
        out3["forces"].detach().cpu().numpy(),
        atol=1e-5,
    )


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_torchsim_cueq(trained_model_path, water_atoms):
    model = MaceTorchSimModel(
        model=trained_model_path,
        device=DEVICE,
        dtype=DTYPE,
        enable_cueq=True,
    )
    results = model(_atoms_state(water_atoms))
    assert "energy" in results
    assert "forces" in results


# ---------------------------------------------------------------------------
# PolarMACE tests
# ---------------------------------------------------------------------------

try:
    import graph_longrange  # noqa: F401

    GRAPH_LONGRANGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GRAPH_LONGRANGE_AVAILABLE = False

POLAR_MODEL_NAME = "polar-1-s"


def _skip_if_polar_unavailable(exc, model_name):
    msg = str(exc).lower()
    if "no such file" in msg or "not found" in msg or "download" in msg:
        pytest.skip(f"Missing Polar foundation model file: {model_name}")
    raise


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
    # PolarMACE-specific outputs
    assert "charges" in results
    assert "dipole" in results
    assert "density_coefficients" in results


def test_torchsim_polar_with_extras(polar_raw_model, water_state_with_extras):
    """Forward pass with PolarMACE using explicit extras."""
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
    model = MaceTorchSimModel(
        model=polar_raw_model,
        device=DEVICE,
        dtype=DTYPE,
    )

    results_no_extras = model(water_state)

    # Explicitly set zeros for all polar extras — should match defaults.
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

    try:
        ase_calc = mace_polar(
            model=POLAR_MODEL_NAME, device=DEVICE.type, default_dtype="float64"
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _skip_if_polar_unavailable(exc, POLAR_MODEL_NAME)

    # Set total_spin=1.0 explicitly so ASE and TorchSim use the same value.
    # ASE's AtomicData.from_config() defaults total_spin to 1.0, but
    # TorchSim's SimState defaults spin to 0.0.
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
