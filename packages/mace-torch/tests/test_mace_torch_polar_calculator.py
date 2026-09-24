"""The charge-aware model behind the ASE calculator.

Its results under the frozen tree's names and shapes, read off the same
evaluation the model gives directly; charge, spin and field from
``atoms.info``; the Hessian as the derivative of the forces; units converted by
dimension; and dielectric derivatives refused, as they are the dipole family's.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from conftest import fp64_only
from mace_torch.calculators import MACECalculator
from mace_torch.calculators.ase_calculator import POLAR_RESULTS
from mace_torch.calculators.padding import PaddingPolicy
from mace_torch.deploy.loader import load_deployed
from mace_torch.train import run_train_stage
from test_mace_torch_polar_training import built, polar_configuration

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


@pytest.fixture(scope="module", name="checkpoints")
def fixture_checkpoints(tmp_path_factory):
    """Two charge-aware models, a committee's worth."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        paths = []
        for seed in (1, 2):
            directory = tmp_path_factory.mktemp(f"polar{seed}")
            config = polar_configuration(directory)
            config = config.model_copy(
                update={"runtime": config.runtime.model_copy(update={"seed": seed})}
            )
            _, model = built(config)
            run_train_stage(config, model, checkpoint_path=directory / "polar")
            paths.append(directory / "polar.json")
        return paths
    finally:
        torch.set_default_dtype(previous)


def dimer(**info) -> Atoms:
    atoms = Atoms(
        "OH2OH2",
        positions=np.vstack([WATER, WATER + np.array([2.8, 0.2, 0.3])]) + 3.0,
    )
    atoms.info.update(info)
    return atoms


def evaluate(calculator, atoms):
    atoms = atoms.copy()
    atoms.calc = calculator
    atoms.get_potential_energy()
    return dict(calculator.results)


@fp64_only
def test_every_polar_result_has_the_frozen_tree_s_name_and_shape(checkpoints):
    calculator = MACECalculator(checkpoints[0])
    results = evaluate(calculator, dimer())
    atoms = 6
    shapes = {
        "dipole": (3,),
        "charges": (atoms,),
        "spins": (atoms,),
        "density_coefficients": (atoms, 4),
        "spin_charge_density": (atoms, 2, 4),
        "fukui_functions": (atoms, 2),
    }
    for key, shape in shapes.items():
        assert np.shape(results[key]) == shape, key
    for key in ("interaction_energy", "electrostatic_energy", "electron_energy"):
        assert isinstance(results[key], float), key
    assert set(results) <= set(calculator.implemented_properties)
    assert set(POLAR_RESULTS) <= set(results)


@fp64_only
def test_the_results_are_the_model_s_own_evaluation(checkpoints):
    calculator = MACECalculator(checkpoints[0])
    atoms = dimer(charge=-1.0, spin=2.0)
    results = evaluate(calculator, atoms)
    graph, _ = calculator._graph(atoms, padded=False)
    direct = load_deployed(checkpoints[0]).compute(graph, compute=("forces",))
    assert direct.total_energy is not None and direct.dipole is not None
    np.testing.assert_allclose(results["energy"], float(direct.total_energy[0]))
    np.testing.assert_allclose(results["dipole"], direct.dipole[0].detach().numpy())
    np.testing.assert_allclose(
        results["charges"], direct.extras["charges"].detach().numpy()
    )
    np.testing.assert_allclose(results["charges"].sum(), -1.0, atol=1e-12)


@fp64_only
def test_charge_spin_and_field_are_read_from_the_structure(checkpoints):
    calculator = MACECalculator(checkpoints[0])
    neutral = evaluate(calculator, dimer())
    explicit = evaluate(calculator, dimer(charge=0.0, spin=1.0))
    charged = evaluate(calculator, dimer(charge=1.0, spin=2.0))
    field = evaluate(calculator, dimer(external_field=[0.05, 0.0, 0.0]))
    assert neutral["energy"] == explicit["energy"]
    assert charged["energy"] != neutral["energy"]
    assert field["energy"] != neutral["energy"]


@fp64_only
def test_a_field_given_to_the_calculator_is_every_structure_s(checkpoints):
    given = MACECalculator(checkpoints[0], external_field=[0.05, 0.0, 0.0])
    read = MACECalculator(checkpoints[0])
    assert (
        evaluate(given, dimer())["energy"]
        == evaluate(read, dimer(external_field=[0.05, 0.0, 0.0]))["energy"]
    )


@fp64_only
def test_units_are_converted_by_dimension(checkpoints):
    plain = evaluate(MACECalculator(checkpoints[0]), dimer())
    scaled = evaluate(
        MACECalculator(checkpoints[0], energy_units_to_eV=2.0, length_units_to_A=3.0),
        dimer(),
    )
    for key in ("energy", "electrostatic_energy", "electron_energy"):
        np.testing.assert_allclose(scaled[key], 2.0 * plain[key])
    np.testing.assert_allclose(scaled["dipole"], 3.0 * plain["dipole"])
    np.testing.assert_allclose(scaled["charges"], plain["charges"])


@fp64_only
def test_a_committee_reports_the_dipole_per_model_and_as_a_spread(checkpoints):
    calculator = MACECalculator(checkpoints)
    results = evaluate(calculator, dimer())
    assert np.shape(results["dipole_comm"]) == (2, 3)
    np.testing.assert_allclose(
        results["dipole_var"], results["dipole_comm"].var(axis=0)
    )


@fp64_only
def test_the_hessian_is_the_derivative_of_the_forces(checkpoints):
    calculator = MACECalculator(checkpoints[0])
    atoms = dimer()
    hessian = calculator.get_hessian(atoms)
    assert isinstance(hessian, np.ndarray) and hessian.shape == (18, 6, 3)
    step = 1e-5
    numerical = np.zeros((18, 6, 3))
    for atom in range(6):
        for axis in range(3):
            forces = []
            for sign in (1.0, -1.0):
                moved = atoms.copy()
                moved.positions[atom, axis] += sign * step
                forces.append(evaluate(calculator, moved)["forces"])
            numerical[3 * atom + axis] = -(forces[0] - forces[1]) / (2 * step)
    np.testing.assert_allclose(hessian, numerical, atol=1e-6)


@fp64_only
def test_padding_leaves_every_result_unchanged(checkpoints):
    plain = evaluate(MACECalculator(checkpoints[0]), dimer(charge=-1.0, spin=2.0))
    padded = evaluate(
        MACECalculator(
            checkpoints[0], padding=PaddingPolicy.requested(16, 256, environ={})
        ),
        dimer(charge=-1.0, spin=2.0),
    )
    for key, value in plain.items():
        np.testing.assert_allclose(padded[key], value, rtol=1e-12, atol=1e-12)


@fp64_only
@pytest.mark.parametrize("periodic", [False, True])
def test_padding_is_clean_where_the_fake_structure_is_summed_in_k_space(
    tmp_path, periodic
):
    """Under the mixed profile every structure goes through reciprocal space,
    the fake one included, with all its atoms on one point."""
    config = polar_configuration(tmp_path, profile="partial")
    _, model = built(config)
    run_train_stage(config, model, checkpoint_path=tmp_path / "polar")
    atoms = dimer(charge=-1.0, spin=2.0)
    if periodic:
        atoms.set_cell(np.eye(3) * 9.0)
        atoms.set_pbc(True)
    plain = evaluate(MACECalculator(tmp_path / "polar.json"), atoms)
    padded = evaluate(
        MACECalculator(
            tmp_path / "polar.json",
            padding=PaddingPolicy.requested(16, 256, environ={}),
        ),
        atoms,
    )
    for key, value in plain.items():
        np.testing.assert_allclose(padded[key], value, rtol=1e-12, atol=1e-12)


@fp64_only
def test_dielectric_derivatives_are_refused(checkpoints):
    calculator = MACECalculator(checkpoints[0])
    with pytest.raises(NotImplementedError, match="dipole"):
        calculator.get_dielectric_derivatives(dimer())
