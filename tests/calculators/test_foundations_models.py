
import numpy as np
import pytest
from ase import build

from mace.calculators import mace_mp, mace_off
from mace.calculators.mace import MACECalculator
from mace.modules.models import ScaleShiftMACE

try:
    import cuequivariance as cue  # pylint: disable=unused-import

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False


@pytest.mark.parametrize("default_dtype", ["float32", "float64"])
def test_mace_mp(default_dtype, capsys: pytest.CaptureFixture) -> None:
    mp_mace = mace_mp(default_dtype=default_dtype)
    assert isinstance(mp_mace, MACECalculator)
    assert mp_mace.model_type == "MACE"
    assert len(mp_mace.models) == 1
    assert isinstance(mp_mace.models[0], ScaleShiftMACE)

    _, stderr = capsys.readouterr()
    assert stderr == ""


def test_mace_mp_stresses() -> None:
    model = "medium"
    device = "cpu"
    atoms = build.bulk("Al", "fcc", a=4.05, cubic=True)
    atoms = atoms.repeat((2, 2, 2))
    mace_mp_model = mace_mp(model=model, device=device, compute_atomic_stresses=True)
    atoms.set_calculator(mace_mp_model)
    stress = atoms.get_stress()
    stresses = atoms.get_stresses()
    assert stress.shape == (6,)
    assert stresses.shape == (32, 6)
    np.testing.assert_allclose(stress, stresses.sum(axis=0), rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize("default_dtype", ["float32", "float64"])
def test_mace_off(default_dtype) -> None:
    mace_off_model = mace_off(model="small", device="cpu", default_dtype=default_dtype)
    assert isinstance(mace_off_model, MACECalculator)
    assert mace_off_model.model_type == "MACE"
    assert len(mace_off_model.models) == 1
    assert isinstance(mace_off_model.models[0], ScaleShiftMACE)

    atoms = build.molecule("H2O")
    atoms.calc = mace_off_model

    E = atoms.get_potential_energy()

    np.testing.assert_allclose(E, -2081.116128586803, rtol=1e-05, atol=1e-08)


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_mace_off_cueq() -> None:
    model = "medium"
    device = "cpu"
    mace_off_model = mace_off(model=model, device=device, enable_cueq=True)
    assert isinstance(mace_off_model, MACECalculator)
    assert mace_off_model.model_type == "MACE"
    assert len(mace_off_model.models) == 1
    assert isinstance(mace_off_model.models[0], ScaleShiftMACE)

    atoms = build.molecule("H2O")
    atoms.calc = mace_off_model

    E = atoms.get_potential_energy()

    np.testing.assert_allclose(E, -2081.116128586803, rtol=1e-05, atol=1e-08)
