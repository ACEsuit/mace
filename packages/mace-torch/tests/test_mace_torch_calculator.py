"""The ASE calculator on small trained models.

Its numbers are checked against the model evaluated directly on the same
graph, which is what agreeing with an evaluation means when both read one
checkpoint; against the frozen tree's calculator they are checked in the parity
suite. What is pinned here is the calculator's own surface: the result keys and
shapes, the per-atom energy split, units, the committee statistics, padding,
the inputs it passes through, and the refusals.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress
from conftest import fp64_only
from mace_torch.calculators import MACECalculator, PaddingPolicy
from mace_torch.deploy.loader import load_deployed
from mace_torch.train import run_train_stage
from test_mace_torch_full_batch import built_task

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


@pytest.fixture(scope="module", name="checkpoints")
def fixture_checkpoints(tmp_path_factory):
    """Two members of a committee, and one with another cutoff."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        paths = {}
        for name, seed, r_max in (("a", 1, 5.0), ("b", 2, 5.0), ("far", 1, 4.5)):
            directory = tmp_path_factory.mktemp(name)
            config, built = built_task(directory, max_num_epochs=1)
            config = config.model_copy(
                update={
                    "runtime": config.runtime.model_copy(update={"seed": seed}),
                    "model": config.model.model_copy(update={"r_max": r_max}),
                }
            )
            from mace_core.observables import load_default_catalogue
            from mace_torch.train import run_data_stage, run_model_stage

            catalogue = load_default_catalogue()
            data = run_data_stage(config, catalogue)
            built = run_model_stage(config, data, catalogue)
            run_train_stage(config, built, checkpoint_path=directory / f"mace_{name}")
            paths[name] = directory / f"mace_{name}.json"
        return paths
    finally:
        torch.set_default_dtype(previous)


def water(periodic: bool = False) -> Atoms:
    atoms = Atoms("OH2", positions=WATER + np.array([0.3, 0.2, 0.1]))
    if periodic:
        atoms.set_cell(np.eye(3) * 4.0)
        atoms.set_pbc(True)
    return atoms


def one(value) -> np.ndarray:
    """A single model's array, which a committee would return as a list."""
    assert isinstance(value, np.ndarray)
    return value


def evaluate(calculator, atoms: Atoms) -> dict:
    atoms = atoms.copy()
    atoms.calc = calculator
    atoms.get_potential_energy()
    return dict(calculator.results)


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize("periodic", [False, True])
def test_the_results_have_the_frozen_tree_s_names_and_shapes(checkpoints, periodic):
    results = evaluate(MACECalculator(model_paths=checkpoints["a"]), water(periodic))
    assert set(results) == {
        "energy",
        "free_energy",
        "energies",
        "node_energy",
        "forces",
        "stress",
    }
    assert isinstance(results["energy"], float)
    assert results["free_energy"] == results["energy"]
    assert results["forces"].shape == (3, 3) and results["forces"].dtype == np.float64
    assert results["stress"].shape == (6,)
    assert results["energies"].shape == results["node_energy"].shape == (3,)
    if not periodic:
        assert not np.any(results["stress"])
    else:
        assert np.any(results["stress"])


@fp64_only
@pytest.mark.parametrize("periodic", [False, True])
def test_the_results_are_the_model_s_own_on_the_same_graph(checkpoints, periodic):
    calculator = MACECalculator(model_paths=checkpoints["a"])
    atoms = water(periodic)
    results = evaluate(calculator, atoms)
    graph, _ = calculator._graph(atoms, padded=False)
    direct = load_deployed(checkpoints["a"]).compute(
        graph, compute=("forces", "stress")
    )
    assert direct.total_energy is not None
    assert direct.forces is not None and direct.stress is not None
    assert results["energy"] == float(direct.total_energy[0])
    np.testing.assert_array_equal(results["forces"], direct.forces.detach().numpy())
    np.testing.assert_array_equal(
        results["stress"], full_3x3_to_voigt_6_stress(direct.stress[0].detach().numpy())
    )


@fp64_only
def test_the_per_atom_energies_come_with_and_without_the_isolated_atom_energies(
    checkpoints,
):
    calculator = MACECalculator(model_paths=checkpoints["a"])
    results = evaluate(calculator, water())
    e0 = calculator.models[0].e0s["default"]
    references = np.array([e0[8], e0[1], e0[1]])
    np.testing.assert_allclose(
        results["energies"] - results["node_energy"], references, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(results["energies"].sum(), results["energy"], rtol=1e-12)


@fp64_only
def test_each_result_is_converted_by_its_own_dimension(checkpoints):
    plain = evaluate(
        MACECalculator(model_paths=checkpoints["a"], compute_atomic_stresses=True),
        water(True),
    )
    energy, length = 2.0, 0.5
    scaled = evaluate(
        MACECalculator(
            model_paths=checkpoints["a"],
            energy_units_to_eV=energy,
            length_units_to_A=length,
            compute_atomic_stresses=True,
        ),
        water(True),
    )
    factors = {
        "energy": energy,
        "free_energy": energy,
        "energies": energy,
        "node_energy": energy,
        "forces": energy / length,
        "stress": energy / length**3,
        "stresses": energy / length**3,
        "virials": energy,
    }
    assert set(factors) == set(plain)
    for key, factor in factors.items():
        np.testing.assert_allclose(
            scaled[key], np.asarray(plain[key]) * factor, err_msg=key
        )


@fp64_only
def test_the_atoms_stresses_add_up_to_the_stress(checkpoints):
    results = evaluate(
        MACECalculator(model_paths=checkpoints["a"], compute_atomic_stresses=True),
        water(True),
    )
    assert results["stresses"].shape == (3, 6)
    assert results["virials"].shape == (3, 3, 3)
    np.testing.assert_allclose(
        results["stresses"].sum(axis=0), results["stress"], rtol=1e-10, atol=1e-14
    )


# ---------------------------------------------------------------------------
# A committee
# ---------------------------------------------------------------------------


@fp64_only
def test_a_committee_reports_the_mean_and_the_population_variance(checkpoints):
    members = [
        evaluate(MACECalculator(model_paths=checkpoints[name]), water(True))
        for name in ("a", "b")
    ]
    committee = MACECalculator(
        model_paths=[checkpoints["a"], checkpoints["b"]], energy_units_to_eV=3.0
    )
    results = evaluate(committee, water(True))
    for key in ("energy", "forces", "stress"):
        values = np.stack([np.asarray(member[key]) for member in members]) * 3.0
        np.testing.assert_allclose(results[key], values.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(results[f"{key}_comm"], values, rtol=1e-12)
        # A spread of two nearly equal numbers keeps few of their digits, so
        # the two orders of arithmetic agree to far fewer than fp64 carries.
        np.testing.assert_allclose(
            results[f"{key}_var"], values.var(axis=0, ddof=0), rtol=1e-6, atol=1e-24
        )
    assert results["stress_comm"].shape == (2, 6)
    assert set(results) <= set(committee.implemented_properties)


@fp64_only
def test_a_pattern_names_a_committee_of_models_and_not_their_run_checkpoints(
    checkpoints,
):
    """A training run leaves its run checkpoints beside the model, and
    ``mace_*.json`` matches their records too. They are not models."""
    directory = Path(checkpoints["a"]).parent
    assert list(directory.glob("mace_a.run-*.json"))
    for pattern in ("mace_*.json", "mace_*"):
        assert len(MACECalculator(model_paths=str(directory / pattern)).models) == 1
    with pytest.raises(ValueError, match="no model file matches"):
        MACECalculator(model_paths=str(Path(checkpoints["a"]).parent / "none_*.json"))


def test_a_committee_with_two_cutoffs_is_refused(checkpoints):
    with pytest.raises(ValueError, match=r"\[5\.0, 4\.5\]"):
        MACECalculator(model_paths=[checkpoints["a"], checkpoints["far"]])


def test_every_key_a_single_model_writes_is_declared(checkpoints):
    calculator = MACECalculator(
        model_paths=checkpoints["a"], compute_atomic_stresses=True
    )
    assert set(evaluate(calculator, water(True))) == set(
        calculator.implemented_properties
    )


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def test_a_calculator_pads_nothing_by_default(checkpoints, monkeypatch):
    monkeypatch.delenv("MACE_ASE_PAD_NUM_ATOMS", raising=False)
    monkeypatch.delenv("MACE_ASE_PAD_NUM_EDGES", raising=False)
    assert MACECalculator(model_paths=checkpoints["a"]).padding.mode == "none"


def test_the_environment_selects_a_fixed_budget(checkpoints, monkeypatch):
    monkeypatch.setenv("MACE_ASE_PAD_NUM_ATOMS", "8")
    monkeypatch.setenv("MACE_ASE_PAD_NUM_EDGES", "256")
    policy = MACECalculator(model_paths=checkpoints["a"]).padding
    assert (policy.mode, policy.nodes_budget, policy.edges_budget) == ("fixed", 8, 256)


def test_a_compiled_calculator_pads_automatically(checkpoints, monkeypatch):
    monkeypatch.delenv("MACE_ASE_PAD_NUM_ATOMS", raising=False)
    monkeypatch.delenv("MACE_ASE_PAD_NUM_EDGES", raising=False)
    compiled = []
    monkeypatch.setattr(
        torch, "compile", lambda model, **settings: compiled.append(settings) or model
    )
    calculator = MACECalculator(model_paths=checkpoints["a"], compile_mode="default")
    assert calculator.padding.mode == "auto"
    assert compiled == [{"mode": "default", "dynamic": False}]


@fp64_only
@pytest.mark.parametrize("periodic", [False, True])
def test_padding_leaves_the_real_structure_s_results_as_they_were(
    checkpoints, periodic
):
    plain = evaluate(MACECalculator(model_paths=checkpoints["a"]), water(periodic))
    padded = evaluate(
        MACECalculator(
            model_paths=checkpoints["a"],
            padding=PaddingPolicy(mode="fixed", nodes_budget=8, edges_budget=3000),
        ),
        water(periodic),
    )
    assert set(padded) == set(plain)
    for key, value in plain.items():
        np.testing.assert_allclose(
            padded[key], value, rtol=1e-12, atol=1e-14, err_msg=key
        )


class _Shapes:
    """Stands in for an engine and records the batch sizes it was given."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.seen: list[tuple[int, int]] = []

    def __call__(self, graph, **settings):
        self.seen.append(
            (int(graph["positions"].shape[0]), int(graph["edge_index"].shape[1]))
        )
        return self.engine(graph, **settings)


@fp64_only
def test_a_growing_structure_changes_the_batch_shape_once_per_growth(
    checkpoints, caplog
):
    calculator = MACECalculator(
        model_paths=checkpoints["a"], padding=PaddingPolicy(mode="auto")
    )
    recorder = _Shapes(calculator._engines[0])
    calculator._engines = [recorder]
    small = water(True)
    large = water(True)
    large.set_cell(np.eye(3) * 3.0)
    with caplog.at_level(logging.WARNING):
        for atoms in (small, small, large, large, small):
            # Past ASE's cache, which would skip a structure seen just before.
            calculator.reset()
            evaluate(calculator, atoms)
    assert recorder.seen[0] == recorder.seen[1]
    assert recorder.seen[2] != recorder.seen[1]
    assert recorder.seen[2] == recorder.seen[3] == recorder.seen[4]
    assert caplog.text.count("changes the batch shape once") == 1


# ---------------------------------------------------------------------------
# Inputs and cache
# ---------------------------------------------------------------------------


@fp64_only
def test_passed_through_entries_reach_the_graph(checkpoints):
    calculator = MACECalculator(
        model_paths=checkpoints["a"],
        padding=PaddingPolicy(mode="fixed", nodes_budget=6, edges_budget=512),
    )
    atoms = water()
    atoms.info["charge"] = 1.0
    atoms.arrays["Qs"] = np.array([0.5, -0.25, -0.25])
    graph, info = calculator._graph(atoms, padded=True)
    np.testing.assert_array_equal(graph["total_charge"].numpy(), [1.0, 0.0])
    np.testing.assert_array_equal(
        graph["charges"][: info.nodes].numpy(), [0.5, -0.25, -0.25]
    )
    assert not torch.any(graph["charges"][info.nodes :])


@fp64_only
def test_changing_a_passed_through_entry_invalidates_the_cache(checkpoints):
    atoms = water()
    atoms.info["charge"] = 0.0
    atoms.calc = MACECalculator(model_paths=checkpoints["a"])
    atoms.get_potential_energy()
    assert atoms.calc.check_state(atoms) == []
    atoms.info["charge"] = 1.0
    assert atoms.calc.check_state(atoms) == ["info"]


# ---------------------------------------------------------------------------
# Beyond ASE
# ---------------------------------------------------------------------------


@fp64_only
def test_the_hessian_and_the_descriptors_have_the_frozen_tree_s_shapes(checkpoints):
    single = MACECalculator(model_paths=checkpoints["a"])
    committee = MACECalculator(model_paths=[checkpoints["a"], checkpoints["b"]])
    atoms = water()
    assert one(single.get_hessian(atoms)).shape == (9, 3, 3)
    hessians = committee.get_hessian(atoms)
    assert isinstance(hessians, list) and len(hessians) == 2
    descriptors = one(single.get_descriptors(atoms))
    assert descriptors.ndim == 2 and descriptors.shape[0] == 3
    first = one(single.get_descriptors(atoms, num_layers=1))
    assert first.shape[1] < descriptors.shape[1]
    assert len(committee.get_descriptors(atoms)) == 2


@fp64_only
def test_the_hessian_is_converted_by_energy_over_length_squared(checkpoints):
    atoms = water()
    plain = one(MACECalculator(model_paths=checkpoints["a"]).get_hessian(atoms))
    scaled = MACECalculator(
        model_paths=checkpoints["a"], energy_units_to_eV=2.0, length_units_to_A=0.5
    ).get_hessian(atoms)
    np.testing.assert_allclose(scaled, plain * 8.0)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_head_the_model_does_not_have_is_refused(checkpoints):
    with pytest.raises(ValueError, match="no head 'DFT'"):
        MACECalculator(model_paths=checkpoints["a"], head="DFT")


def test_a_calculator_with_no_model_is_refused():
    with pytest.raises(ValueError, match="model_paths or models"):
        MACECalculator()
