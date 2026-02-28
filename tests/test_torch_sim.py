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

from mace.calculators.torch_sim import MaceTorchSimModel


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
