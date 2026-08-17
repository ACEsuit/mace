import copy
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from ase import build
from ase.atoms import Atoms
from ase.calculators.test import gradient_test
from ase.filters import FrechetCellFilter

import mace.calculators.foundations_models as foundations_models
from mace.calculators import mace_mp, mace_off
from mace.calculators.foundations_models import mace_omol, mace_polar
from mace.calculators.mace import MACECalculator
from mace.modules.models import ScaleShiftMACE
from mace.tools.torch_tools import default_dtype
from tests.helpers import CUET_AVAILABLE, base_mace_params, run_mace_train

from tests.helpers import REPO_ROOT

pytest_mace_dir = REPO_ROOT

# NOTE: this file keeps its own (module-scoped) fitting_configs fixture: it
# uses different isolated-atom energies and adds REF_dipole / Qs, so it is not
# the canonical fixture from tests/conftest.py


def write_extxyz_test(tmp_path, atoms):
    assert isinstance(
        atoms, Atoms
    ), "write_extxyz_test only working for Atoms, not anything else such as list(Atoms)"
    ase.io.write(tmp_path / "test.extxyz", atoms)
    atoms_written = ase.io.read(tmp_path / "test.extxyz")

    nonstd_fields = set(
        [
            "node_energy",
            "energy_var",
            "energy_comm",
            "stress_var",
            "stress_comm",
            "forces_var",
            "forces_comm",
            "virials",
        ]
    )
    # everything that we expect has been written
    assert set(atoms.calc.results.keys()) - nonstd_fields == set(
        atoms_written.calc.results.keys()
    )
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
    # canonical params without the explicit use_reduced_cg flag
    _mace_params = base_mace_params()
    del _mace_params["use_reduced_cg"]

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    p = run_mace_train(mace_params)

    assert p.returncode == 0

    return MACECalculator(model_paths=tmp_path / "MACE.model", device="cpu")


@pytest.fixture(scope="module", name="trained_equivariant_model")
def trained_model_equivariant_fixture(tmp_path_factory, fitting_configs):
    # canonical params with equivariant irreps, without use_reduced_cg
    _mace_params = base_mace_params()
    _mace_params["hidden_irreps"] = "16x0e+16x1o"
    del _mace_params["use_reduced_cg"]

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    p = run_mace_train(mace_params)

    assert p.returncode == 0

    return MACECalculator(model_paths=tmp_path / "MACE.model", device="cpu")


@pytest.fixture(scope="module", name="trained_equivariant_model_cueq")
def trained_model_equivariant_fixture_cueq(tmp_path_factory, fitting_configs):
    # canonical params with equivariant irreps, without use_reduced_cg
    _mace_params = base_mace_params()
    _mace_params["hidden_irreps"] = "16x0e+16x1o"
    del _mace_params["use_reduced_cg"]

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    p = run_mace_train(mace_params)

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

    p = run_mace_train(mace_params)

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
    p = run_mace_train(mace_params)
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

    p = run_mace_train(mace_params)

    assert p.returncode == 0

    return MACECalculator(
        model_paths=tmp_path / "MACE.model", device="cpu", model_type="EnergyDipoleMACE"
    )


@pytest.fixture(scope="module", name="trained_committee")
def trained_committee_fixture(tmp_path_factory, fitting_configs):
    _seeds = [5, 6, 7]
    _model_paths = []
    for seed in _seeds:
        # canonical params with a small model and per-member name/seed,
        # without use_reduced_cg
        _mace_params = base_mace_params()
        _mace_params["name"] = f"MACE{seed}"
        _mace_params["hidden_irreps"] = "16x0e"
        _mace_params["seed"] = seed
        del _mace_params["use_reduced_cg"]

        tmp_path = tmp_path_factory.mktemp(f"run{seed}_")

        ase.io.write(tmp_path / "fit.xyz", fitting_configs)

        mace_params = _mace_params.copy()
        mace_params["checkpoints_dir"] = str(tmp_path)
        mace_params["model_dir"] = str(tmp_path)
        mace_params["train_file"] = tmp_path / "fit.xyz"

        p = run_mace_train(mace_params)

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
    at_wrapped = FrechetCellFilter(at)
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


def test_calculator_dtype_is_instance_local(fitting_configs, trained_model):
    atoms = fitting_configs[2].copy()

    with default_dtype(torch.float64):
        calc32 = MACECalculator(
            models=copy.deepcopy(trained_model.models[0]),
            device="cpu",
            default_dtype="float32",
        )
        assert torch.get_default_dtype() == torch.float64

        calc64 = MACECalculator(
            models=copy.deepcopy(trained_model.models[0]),
            device="cpu",
            default_dtype="float64",
        )
        assert torch.get_default_dtype() == torch.float64

        assert next(calc32.models[0].parameters()).dtype == torch.float32
        assert next(calc64.models[0].parameters()).dtype == torch.float64

        batch32 = calc32._atoms_to_batch(atoms)  # pylint: disable=protected-access
        batch64 = calc64._atoms_to_batch(atoms)  # pylint: disable=protected-access
        assert torch.get_default_dtype() == torch.float64

        assert batch32["positions"].dtype == torch.float32
        assert batch32["node_attrs"].dtype == torch.float32
        assert batch64["positions"].dtype == torch.float64
        assert batch64["node_attrs"].dtype == torch.float64

        atoms32 = atoms.copy()
        atoms32.calc = calc32
        atoms64 = atoms.copy()
        atoms64.calc = calc64

        forces32 = atoms32.get_forces()
        forces64 = atoms64.get_forces()
        assert torch.get_default_dtype() == torch.float64

        assert calc32.results["forces"].dtype == np.float32
        assert calc64.results["forces"].dtype == np.float64
        assert np.isfinite(forces32).all()
        assert np.isfinite(forces64).all()


def test_calculator_dipole(tmp_path, fitting_configs, trained_dipole_model):
    at = fitting_configs[2].copy()
    at.calc = trained_dipole_model

    dip = at.get_dipole_moment()

    assert len(dip) == 3
    write_extxyz_test(tmp_path, at)


def test_calculator_energy_dipole(
    tmp_path, fitting_configs, trained_energy_dipole_model
):
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


@pytest.mark.cueq
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



def test_calculator_padding(trained_model, fitting_configs):
    """Calculator with graph padding should give the same results as without."""
    water = fitting_configs[2].copy()

    calc_no_pad = MACECalculator(
        models=trained_model.models[0], device="cpu", default_dtype="float64"
    )
    water_no_pad = water.copy()
    water_no_pad.calc = calc_no_pad
    e_no_pad = water_no_pad.get_potential_energy()
    f_no_pad = water_no_pad.get_forces()
    s_no_pad = water_no_pad.get_stress()

    calc_pad = MACECalculator(
        models=trained_model.models[0],
        device="cpu",
        default_dtype="float64",
        pad_num_atoms=10,
        pad_num_edges=128,
    )
    water_pad = water.copy()
    water_pad.calc = calc_pad
    e_pad = water_pad.get_potential_energy()
    f_pad = water_pad.get_forces()
    s_pad = water_pad.get_stress()

    assert np.allclose(e_no_pad, e_pad, atol=1e-6)
    assert np.allclose(f_no_pad, f_pad, atol=1e-6)
    assert np.allclose(s_no_pad, s_pad, atol=1e-6)
