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
