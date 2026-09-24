"""Giving a trained model an element it was not trained on.

The foundation here is trained on waters, so it knows hydrogen and oxygen, and
carbon is added. Carbon sits between the two in the table, so every per-element
tensor has its rows reordered as well as extended, which is the case a rule
that only appended would get wrong.

What is pinned: waters compute exactly what they did, to the last bit; every
row an old element had is carried byte for byte; the new rows are the spec's,
reproducibly; the record says what was added and how, and keeps the parent's
own record unchanged; and a fine-tune on data holding carbon starts from it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.calculators import MACECalculator
from mace_torch.deploy.loader import load_deployed
from mace_torch.finetune.extend import (
    ElementExtensionError,
    NewSpeciesInit,
    extend_elements,
)
from mace_torch.finetune.foundation import read_foundation
from mace_torch.serialization import canonical_state
from mace_torch.train import (
    run_data_stage,
    run_model_stage,
    run_train_stage,
    write_model,
)

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
CARBON = -1030.0


def waters(count=12, seed=0):
    generator = np.random.default_rng(seed)
    frames = []
    for number, energy in ((1, -13.6), (8, -2040.0)):
        atom = Atoms(numbers=[number], positions=[[0.0, 0.0, 0.0]])
        atom.info.update(REF_energy=energy, config_type="IsolatedAtom")
        atom.arrays["REF_forces"] = np.zeros((1, 3))
        frames.append(atom)
    for index in range(count):
        atoms = Atoms(
            "OH2", positions=WATER + generator.normal(scale=0.05, size=(3, 3))
        )
        atoms.info["REF_energy"] = -2067.2 + 0.02 * index
        atoms.arrays["REF_forces"] = generator.normal(scale=0.1, size=(3, 3))
        frames.append(atoms)
    return frames


def methanol(offset=0.0):
    atoms = Atoms(
        "COHHHH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.42, 0.0, 0.0],
            [1.75, 0.9, 0.1],
            [-0.36, 1.03, 0.05],
            [-0.36, -0.52, 0.89],
            [-0.36, -0.5, -0.9],
        ],
    )
    atoms.info["REF_energy"] = -3140.0 + offset
    atoms.arrays["REF_forces"] = np.zeros((6, 3))
    return atoms


@pytest.fixture(scope="module", name="foundation")
def fixture_foundation(tmp_path_factory):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        directory = tmp_path_factory.mktemp("water_foundation")
        write(directory / "train.xyz", waters())
        config = ResolvedConfig.model_validate(
            {
                "runtime": {"work_dir": str(directory), "seed": 3},
                "data": {
                    "heads": {
                        "pbe": {
                            "train_file": str(directory / "train.xyz"),
                            "e0s": {"isolated_atoms": {}},
                        }
                    },
                    "valid_fraction": 0.2,
                    "pin_memory": False,
                },
                "model": {
                    "observables": ["energy", "forces"],
                    "r_max": 3.0,
                    "num_channels": 4,
                    "max_ell": 1,
                    "correlation": 2,
                },
                "training": {"max_num_epochs": 1, "batch_size": 4},
            }
        )
        data = run_data_stage(config, DEFAULT_CATALOGUE)
        built = run_model_stage(config, data, DEFAULT_CATALOGUE)
        trained = run_train_stage(config, built)
        return write_model(directory / "model", trained.model, built.metadata)
    finally:
        torch.set_default_dtype(previous)


SEEDED = NewSpeciesInit(seed=5)


def extended(foundation, tmp_path, init=SEEDED, name="extended"):
    return extend_elements(foundation, tmp_path / name, {"pbe": {6: CARBON}}, init)


def evaluate(path, atoms):
    atoms = atoms.copy()
    atoms.calc = MACECalculator(model_paths=path)
    return atoms.get_potential_energy(), atoms.get_forces()


# ---------------------------------------------------------------------------
# The old elements are untouched
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize("kind", ["fresh", "copy"])
def test_a_structure_of_old_elements_computes_exactly_what_it_did(
    foundation, tmp_path, kind
):
    init = NewSpeciesInit(kind=kind, seed=5, donors={6: 8} if kind == "copy" else {})
    path = extended(foundation, tmp_path, init)
    for frame in waters(4, seed=9)[2:]:
        before_energy, before_forces = evaluate(foundation, frame)
        after_energy, after_forces = evaluate(path, frame)
        assert after_energy == before_energy
        assert np.array_equal(after_forces, before_forces)


@fp64_only
def test_every_row_an_old_element_had_is_carried_byte_for_byte(foundation, tmp_path):
    parent = load_deployed(foundation)
    child = load_deployed(extended(foundation, tmp_path))
    assert list(parent.z_table.zs) == [1, 8]
    assert list(child.z_table.zs) == [1, 6, 8]
    old = canonical_state(parent.model)
    new = canonical_state(child.model)
    rows = [0, 2]
    for path, tensors in old.items():
        for name, value in tensors.items():
            carried = new[path][name]
            if carried.shape == value.shape:
                assert torch.equal(carried, value), f"{path}:{name}"
            elif path.endswith("node_embedding"):
                assert torch.equal(
                    carried.reshape(-1, 3)[:, rows], value.reshape(-1, 2)
                )
            elif path.endswith("energy_head"):
                assert torch.equal(carried[:, rows], value), f"{path}:{name}"
            else:
                assert torch.equal(carried[rows], value), f"{path}:{name}"


# ---------------------------------------------------------------------------
# The new rows
# ---------------------------------------------------------------------------


def carbon_rows(path):
    state = canonical_state(load_deployed(path).model)
    rows = {}
    for op, tensors in state.items():
        for name, value in tensors.items():
            if op.endswith("node_embedding") and name == "weight":
                rows[f"{op}:{name}"] = value.reshape(-1, 3)[:, 1]
            elif (
                op.endswith(".skip") or op.endswith(".contraction")
            ) and name == "weight":
                rows[f"{op}:{name}"] = value[1]
    return rows


@fp64_only
def test_the_same_spec_and_seed_give_the_same_rows(foundation, tmp_path):
    first = carbon_rows(extended(foundation, tmp_path, name="first"))
    again = carbon_rows(extended(foundation, tmp_path, name="again"))
    other = carbon_rows(
        extended(foundation, tmp_path, NewSpeciesInit(seed=6), name="other")
    )
    assert first.keys() == again.keys() and first
    assert all(torch.equal(first[key], again[key]) for key in first)
    assert any(not torch.equal(first[key], other[key]) for key in first)


@fp64_only
def test_a_copied_element_starts_as_its_donor(foundation, tmp_path):
    path = extended(foundation, tmp_path, NewSpeciesInit(kind="copy", donors={6: 8}))
    state = canonical_state(load_deployed(path).model)
    for op, tensors in state.items():
        if "weight" not in tensors:
            continue
        value = tensors["weight"]
        if op.endswith("node_embedding"):
            grid = value.reshape(-1, 3)
            assert torch.equal(grid[:, 1], grid[:, 2])
        elif op.endswith(".skip") or op.endswith(".contraction"):
            assert torch.equal(value[1], value[2])
    assert load_deployed(path).e0s["pbe"][6] == CARBON


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


@fp64_only
def test_the_record_says_what_was_added_and_keeps_the_parent_s(foundation, tmp_path):
    parent = load_deployed(foundation).metadata
    child = load_deployed(extended(foundation, tmp_path)).metadata
    record = child.element_extension
    assert record is not None
    assert record.added == ["C"]
    assert record.initialization == "fresh"
    assert record.seed == 5
    assert child.parents[0].role == "initial_weights"
    assert child.parents[0].metadata == parent
    assert parent.element_extension is None
    assert child.heads["pbe"].e0.values["C"] == CARBON
    assert child.heads["pbe"].e0.values["H"] == parent.heads["pbe"].e0.values["H"]


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------


def test_an_element_already_in_the_table_is_refused(foundation, tmp_path):
    with pytest.raises(ElementExtensionError, match=r"\[8\] are already"):
        extend_elements(foundation, tmp_path / "x", {"pbe": {8: -2040.0}})


def test_every_head_has_to_give_the_added_energies(foundation, tmp_path):
    with pytest.raises(ElementExtensionError, match="every head"):
        extend_elements(foundation, tmp_path / "x", {"other": {6: CARBON}})


def test_a_copy_without_a_donor_is_refused(foundation, tmp_path):
    with pytest.raises(ElementExtensionError, match=r"\[6\] have none"):
        extend_elements(
            foundation, tmp_path / "x", {"pbe": {6: CARBON}}, NewSpeciesInit("copy")
        )


def test_a_donor_the_model_lacks_is_refused(foundation, tmp_path):
    with pytest.raises(ElementExtensionError, match=r"donors \[7\]"):
        extend_elements(
            foundation,
            tmp_path / "x",
            {"pbe": {6: CARBON}},
            NewSpeciesInit("copy", donors={6: 7}),
        )


def test_an_initialization_that_is_neither_is_refused():
    with pytest.raises(ElementExtensionError, match="'fresh'"):
        NewSpeciesInit(kind="noise")  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# A fine-tune from it
# ---------------------------------------------------------------------------


@fp64_only
def test_a_fine_tune_on_the_new_element_starts_from_the_extension(foundation, tmp_path):
    """The water foundation refuses methanol; its extension takes it, with the
    carbon energy it was given."""
    write(tmp_path / "methanol.xyz", [methanol(offset) for offset in (0.0, 0.1, 0.2)])
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(tmp_path), "seed": 1},
            "finetune": {"foundation_model": str(foundation)},
            "data": {
                "heads": {
                    "pbe": {
                        "train_file": str(tmp_path / "methanol.xyz"),
                        "e0s": {"foundation": {}},
                    }
                },
                "valid_fraction": 0.34,
                "pin_memory": False,
            },
            "model": {"observables": ["energy", "forces"]},
        }
    )
    from mace_torch.train import DataStageError

    with pytest.raises(DataStageError, match=r"elements \[6\]"):
        run_data_stage(
            config,
            DEFAULT_CATALOGUE,
            foundation=read_foundation(foundation, DEFAULT_CATALOGUE).context(),
        )
    path = extended(foundation, tmp_path)
    source = read_foundation(path, DEFAULT_CATALOGUE)
    data = run_data_stage(config, DEFAULT_CATALOGUE, foundation=source.context())
    assert list(data.z_table.zs) == [1, 6, 8]
    assert data.e0s.values["pbe"][6] == CARBON
    built = run_model_stage(config, data, DEFAULT_CATALOGUE, foundation=source)
    parent = built.metadata.parents[0].metadata
    assert parent is not None and parent.element_extension is not None
