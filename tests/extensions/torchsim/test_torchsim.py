"""Tests for the MACE TorchSim model interface.

Uses factory functions (mirrored from the torch-sim test conftest) to build
model/calculator consistency tests and ``validate_model_outputs`` tests from
fixture names.  Unlike the upstream factories we feed them sim_state *fixtures*
(built from ASE atoms) instead of the ``SIMSTATE_GENERATORS`` registry, which
keeps the inputs hermetic to this repo and avoids needing torch-sim's bulk
crystal generators at consistency-test time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers import (
    CUET_AVAILABLE,
    GRAPH_LONGRANGE_AVAILABLE,
    TORCHSIM_AVAILABLE,
    run_mace_train,
)

if TORCHSIM_AVAILABLE:
    import torch_sim as ts

pytestmark = [
    pytest.mark.torchsim,
    pytest.mark.skipif(not TORCHSIM_AVAILABLE, reason="torch-sim not installed"),
]

DEVICE = torch.device("cpu")
DTYPE = torch.float64
POLAR_MODEL_NAME = "polar-1-s"

pytest_mace_dir = Path(__file__).parent.parent


def make_model_calculator_consistency_test(
    test_name: str,
    model_fixture_name: str,
    calculator_fixture_name: str,
    sim_state_fixture_names: tuple[str, ...],
    *,
    energy_rtol: float = 1e-5,
    energy_atol: float = 1e-5,
    force_rtol: float = 1e-5,
    force_atol: float = 1e-5,
    stress_rtol: float = 1e-5,
    stress_atol: float = 1e-5,
):
    """Build a parametrized consistency test between a model and a calculator.

    Adapted from the torch-sim conftest factory; takes sim_state *fixtures*
    rather than entries of ``SIMSTATE_GENERATORS``.
    """

    @pytest.mark.parametrize("sim_state_fixture_name", sim_state_fixture_names)
    def _test(sim_state_fixture_name: str, request: pytest.FixtureRequest) -> None:
        from torch_sim.testing import assert_model_calculator_consistency

        model = request.getfixturevalue(model_fixture_name)
        calculator = request.getfixturevalue(calculator_fixture_name)
        sim_state = request.getfixturevalue(sim_state_fixture_name)
        assert_model_calculator_consistency(
            model=model,
            calculator=calculator,
            sim_state=sim_state,
            energy_rtol=energy_rtol,
            energy_atol=energy_atol,
            force_rtol=force_rtol,
            force_atol=force_atol,
            stress_rtol=stress_rtol,
            stress_atol=stress_atol,
        )

    _test.__name__ = f"test_{test_name}_consistency"
    return _test


def make_validate_model_outputs_test(
    model_fixture_name: str,
    *,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
    check_detached: bool = True,
    state_modifier=None,
):
    """Build a ``validate_model_outputs`` test for a model fixture."""

    def _test(request: pytest.FixtureRequest) -> None:
        from torch_sim.models.interface import validate_model_outputs

        model = request.getfixturevalue(model_fixture_name)
        validate_model_outputs(
            model,
            device,
            dtype,
            check_detached=check_detached,
            state_modifier=state_modifier,
        )

    _test.__name__ = f"test_{model_fixture_name}_output_validation"
    return _test


def _ensure_periodic_cell(state, box_size: float = 20.0):
    """Force PBC and replace any zero-volume cell with a cubic *box_size* box.

    Used as a ``state_modifier`` when running ``validate_model_outputs`` on
    PolarMACE: the validator probes the model on a non-periodic benzene
    molecule (zero cell) which makes graph_longrange's k-space code blow up.
    Wrapping it in a big PBC box keeps the chemistry effectively molecular
    while giving the long-range code a well-defined reciprocal lattice.
    """
    state = state.clone()
    state.pbc = torch.tensor([True, True, True])
    cell = state.cell.clone()
    volumes = torch.linalg.det(cell).abs()
    eye = torch.eye(3, device=cell.device, dtype=cell.dtype) * box_size
    for i, vol in enumerate(volumes):
        if vol < 1e-6:
            cell[i] = eye
    state.cell = cell
    return state


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory):
    """Train a minimal MACE model and return the path to the model file."""
    from ase.atoms import Atoms

    from ase.build import bulk, molecule

    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    isolated_energies = {1: -0.5, 8: 1.0, 6: -1.3, 14: -0.7, 12: -0.2, 26: -2.1}
    fit_configs = []
    for z, e0 in isolated_energies.items():
        atom = Atoms(numbers=[z], positions=[[0, 0, 0]], cell=[6] * 3)
        atom.info["REF_energy"] = e0
        atom.info["config_type"] = "IsolatedAtom"
        fit_configs.append(atom)

    np.random.seed(42)

    def _rattled(atoms_template):
        c = atoms_template.copy()
        c.positions += np.random.normal(0.1, size=c.positions.shape)
        c.info["REF_energy"] = np.random.normal(0.1)
        c.new_array("REF_forces", np.random.normal(0.1, size=c.positions.shape))
        c.info["REF_stress"] = np.random.normal(0.1, size=6)
        return c

    extra_templates = [
        bulk("Si", "diamond", a=5.43, cubic=True),
        bulk("Mg", "hcp", a=3.21, c=5.21),
        bulk("Fe", "bcc", a=2.87, cubic=True),
        molecule("C6H6", vacuum=4.0),
    ]
    extra_templates[-1].pbc = [True] * 3  # MACE stress training requires PBC
    templates = [water, *extra_templates]
    for template in templates:
        for _ in range(3):
            fit_configs.append(_rattled(template))

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

    p = run_mace_train(mace_params)
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


@pytest.fixture(scope="module")
def water_batched_atoms(water_atoms):
    rng = np.random.default_rng(seed=0)
    w1, w2 = water_atoms.copy(), water_atoms.copy()
    w2.positions += rng.normal(0.01, size=w2.positions.shape)
    return [w1, w2]


@pytest.fixture(scope="module")
def water_sim_state(water_atoms):
    return ts.io.atoms_to_state(water_atoms, device=DEVICE, dtype=DTYPE)


@pytest.fixture(scope="module")
def water_batched_sim_state(water_batched_atoms):
    return ts.io.atoms_to_state(water_batched_atoms, device=DEVICE, dtype=DTYPE)


@pytest.fixture(scope="module")
def mace_model(trained_model_path):
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    return MaceTorchSimModel(
        model=trained_model_path,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.fixture(scope="module")
def mace_calculator(trained_model_path):
    from mace.calculators.mace import MACECalculator

    return MACECalculator(
        model_paths=trained_model_path, device=DEVICE.type, default_dtype="float64"
    )


test_mace_water = make_model_calculator_consistency_test(
    test_name="mace_water",
    model_fixture_name="mace_model",
    calculator_fixture_name="mace_calculator",
    sim_state_fixture_names=("water_sim_state",),
)

test_mace_output_validation = make_validate_model_outputs_test(
    model_fixture_name="mace_model",
    check_detached=True,
)


def test_mace_torchsim_basic(mace_model, water_sim_state, water_atoms):
    """Smoke test: forward pass returns the expected keys/shapes."""
    results = mace_model(water_sim_state)
    assert "energy" in results
    assert "forces" in results
    assert "stress" in results
    assert results["energy"].shape == (1,)
    assert results["forces"].shape == (len(water_atoms), 3)


def test_mace_torchsim_no_stress(trained_model_path, water_sim_state):
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    model = MaceTorchSimModel(
        model=trained_model_path,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=False,
    )
    results = model(water_sim_state)
    assert "energy" in results
    assert "forces" in results




@pytest.fixture(scope="module")
def mace_cueq_model(trained_model_path):
    if not CUET_AVAILABLE:
        pytest.skip("cuequivariance not installed")
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    return MaceTorchSimModel(
        model=trained_model_path,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
        enable_cueq=True,
    )


test_mace_cueq_water = pytest.mark.cueq(
    make_model_calculator_consistency_test(
        test_name="mace_cueq_water",
        model_fixture_name="mace_cueq_model",
        calculator_fixture_name="mace_calculator",
        sim_state_fixture_names=("water_sim_state",),
    )
)

test_mace_cueq_output_validation = pytest.mark.cueq(
    make_validate_model_outputs_test(
        model_fixture_name="mace_cueq_model",
        check_detached=True,
    )
)


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
def polar_model(polar_raw_model):
    from mace.calculators.mace_torchsim import MaceTorchSimModel

    return MaceTorchSimModel(
        model=polar_raw_model,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.fixture(scope="module")
def polar_calculator():
    if not GRAPH_LONGRANGE_AVAILABLE:
        pytest.skip("graph_longrange is not installed")
    from mace.calculators.foundations_models import mace_polar

    try:
        return mace_polar(
            model=POLAR_MODEL_NAME, device=DEVICE.type, default_dtype="float64"
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _skip_if_polar_unavailable(exc, POLAR_MODEL_NAME)


@pytest.fixture(scope="module")
def water_sim_state_polar(water_sim_state):
    """Single water with spin=1.0 to match ASE's AtomicData default."""
    state = water_sim_state.clone()
    state.spin = torch.tensor([1.0], dtype=DTYPE)
    return state


@pytest.fixture(scope="module")
def water_batched_sim_state_polar(water_batched_sim_state):
    state = water_batched_sim_state.clone()
    state.spin = torch.tensor([1.0, 1.0], dtype=DTYPE)
    return state


@pytest.fixture(scope="module")
def water_sim_state_with_extras(water_sim_state):
    state = water_sim_state.clone()
    state.external_E_field = torch.tensor([[0.1, 0.0, 0.0]], dtype=DTYPE)
    state.charge = torch.tensor([0.0], dtype=DTYPE)
    state.spin = torch.tensor([1.0], dtype=DTYPE)
    return state


@pytest.fixture(scope="module")
def water_batched_sim_state_with_extras(water_batched_sim_state):
    state = water_batched_sim_state.clone()
    state.external_E_field = torch.tensor(
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=DTYPE
    )
    state.charge = torch.tensor([0.0, 0.0], dtype=DTYPE)
    state.spin = torch.tensor([1.0, 1.0], dtype=DTYPE)
    return state


test_polar_water = pytest.mark.polar(
    pytest.mark.network(
        make_model_calculator_consistency_test(
            test_name="polar_water",
            model_fixture_name="polar_model",
            calculator_fixture_name="polar_calculator",
            sim_state_fixture_names=("water_sim_state_polar",),
        )
    )
)

test_polar_output_validation = pytest.mark.polar(
    pytest.mark.network(
        make_validate_model_outputs_test(
            model_fixture_name="polar_model",
            check_detached=True,
            state_modifier=_ensure_periodic_cell,  # needed because default benzene is non-periodic
        )
    )
)


@pytest.mark.polar
@pytest.mark.network
def test_polar_torchsim_basic(polar_model, water_sim_state):
    """Forward pass with PolarMACE using defaults (no extras)."""
    results = polar_model(water_sim_state)
    assert results["energy"].shape == (1,)
    assert results["forces"].shape == (3, 3)
    assert results["stress"].shape == (1, 3, 3)
    assert "charges" in results
    assert "dipole" in results
    assert "density_coefficients" in results


@pytest.mark.polar
@pytest.mark.network
def test_polar_torchsim_with_extras(polar_model, water_sim_state_with_extras):
    results = polar_model(water_sim_state_with_extras)
    assert results["energy"].shape == (1,)
    assert "charges" in results
    assert "dipole" in results
    assert "density_coefficients" in results


@pytest.mark.polar
@pytest.mark.network
def test_polar_torchsim_no_extras_vs_zero_extras(polar_model, water_sim_state):
    """Defaults (no extras) should match explicitly passing zeros."""
    results_no_extras = polar_model(water_sim_state)
    state_zero_extras = water_sim_state.clone()
    state_zero_extras.external_E_field = torch.zeros(1, 3, dtype=DTYPE)
    results_zero_extras = polar_model(state_zero_extras)
    np.testing.assert_allclose(
        results_no_extras["energy"].detach().cpu().numpy(),
        results_zero_extras["energy"].detach().cpu().numpy(),
        atol=1e-10,
    )


@pytest.mark.polar
@pytest.mark.network
def test_polar_torchsim_batched(polar_model, water_batched_sim_state):
    results = polar_model(water_batched_sim_state)
    assert results["energy"].shape == (2,)
    assert results["forces"].shape == (6, 3)
    assert "dipole" in results
    assert results["dipole"].shape[0] == 2


@pytest.mark.polar
@pytest.mark.network
def test_polar_torchsim_batched_with_extras(
    polar_model, water_batched_sim_state_with_extras
):
    results = polar_model(water_batched_sim_state_with_extras)
    assert results["energy"].shape == (2,)
    assert results["forces"].shape == (6, 3)
    assert "dipole" in results
    assert results["dipole"].shape[0] == 2
