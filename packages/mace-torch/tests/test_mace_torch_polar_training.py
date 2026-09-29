"""The charge-aware model from a configuration, trained, written and read back.

The solver the long-range term comes from is part of the model. It is recorded
beside the weights with whether it reproduces the reference bit for bit, a
model is rebuilt with it, and another solver stands in for it only when
neither changes a number. Charge, spin and an applied field are read from the
data per structure, and a structure that gives none is neutral, a singlet and
in no field.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.config.e0s import E0sTable
from mace_core.config.resolved import ResolvedConfig
from mace_core.electrostatics import (
    ENTRY_POINT_GROUPS,
    SolverCapabilities,
    SolverSubstitutionError,
)
from mace_core.electrostatics import registry as registry_module
from mace_core.observables import load_default_catalogue
from mace_torch.deploy.loader import load_deployed
from mace_torch.electrostatics import ReferenceSolver
from mace_torch.models.electrostatics import PolarModel
from mace_torch.serialization import CheckpointError
from mace_torch.train import (
    ModelStageError,
    run_data_stage,
    run_model_stage,
    run_train_stage,
)
from mace_torch_engine_fixtures import build_graph
from test_mace_torch_full_batch import configuration

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

POLAR = {
    "multipole_max_l": 1,
    "multipole_width": 1.0,
    "feature_max_l": 1,
    "feature_widths": [1.0, 1.5],
    "num_recursion_steps": 1,
    "add_local_electron_energy": True,
}


def polar_configuration(tmp_path, profile="molecular", **model) -> ResolvedConfig:
    base = configuration(tmp_path, max_num_epochs=1)
    return base.model_copy(
        update={
            "model": base.model.model_copy(
                update={
                    "model": "polar",
                    "polar": base.model.polar.model_copy(update=POLAR),
                    **model,
                }
            ),
            "electrostatics": base.electrostatics.model_copy(
                update={"enabled": True, "periodicity_profile": profile}
            ),
        }
    )


def built(config):
    catalogue = load_default_catalogue()
    data = run_data_stage(config, catalogue)
    return data, run_model_stage(config, data, catalogue)


class _Accelerated:
    """A solver that is not the reference's numbers."""

    name = "accelerated"
    capabilities = SolverCapabilities(
        ops=frozenset({"long_range_energy", "long_range_features"}),
        devices=frozenset({"cpu"}),
        dtypes=frozenset({"float64", "float32"}),
        slab_normals=frozenset({2}),
        supports_double_backward=True,
        bit_parity=False,
    )

    def long_range_energy(self, descriptor):
        return ReferenceSolver().long_range_energy(descriptor)

    def long_range_features(self, descriptor):
        return ReferenceSolver().long_range_features(descriptor)


class _Twin(_Accelerated):
    """Another solver that is the reference's numbers exactly."""

    name = "twin"
    capabilities = SolverCapabilities(
        ops=_Accelerated.capabilities.ops,
        dtypes=_Accelerated.capabilities.dtypes,
        slab_normals=frozenset({2}),
        supports_double_backward=True,
        bit_parity=True,
    )


class _EntryPoint:
    def __init__(self, name, target):
        self.name, self._target = name, target

    def load(self):
        return self._target


@pytest.fixture(name="registered")
def fixture_registered(monkeypatch):
    points = [
        _EntryPoint("reference", ReferenceSolver),
        _EntryPoint("accelerated", _Accelerated),
        _EntryPoint("twin", _Twin),
    ]

    def entry_points(group: str):
        assert group == ENTRY_POINT_GROUPS["torch"]
        return points

    monkeypatch.setattr(registry_module, "entry_points", entry_points)


@pytest.fixture(name="trained")
def fixture_trained(tmp_path):
    config = polar_configuration(tmp_path)
    _, model = built(config)
    run_train_stage(config, model, checkpoint_path=tmp_path / "polar")
    return tmp_path / "polar.json"


# ---------------------------------------------------------------------------
# From the configuration
# ---------------------------------------------------------------------------


@fp64_only
def test_the_polar_model_is_built_by_its_registry_name(tmp_path):
    _, model = built(polar_configuration(tmp_path))
    assert isinstance(model.model.get_submodule("backbone"), PolarModel)


@fp64_only
def test_a_polar_model_needs_its_long_range_section(tmp_path):
    config = polar_configuration(tmp_path)
    config = config.model_copy(
        update={
            "electrostatics": config.electrostatics.model_copy(
                update={"enabled": False}
            )
        }
    )
    with pytest.raises(ModelStageError, match=r"electrostatics\.enabled"):
        built(config)


@fp64_only
def test_a_polar_model_with_a_repulsion_is_refused(tmp_path):
    """The frozen tree computes this model's repulsion and never adds it."""
    with pytest.raises(ModelStageError, match="pair_repulsion"):
        built(polar_configuration(tmp_path, pair_repulsion=True))


@fp64_only
def test_charge_spin_and_field_come_from_the_data_or_their_defaults(tmp_path):
    frames = []
    for index in range(6):
        if index % 2:
            frame = Atoms("OH", positions=WATER[:2])
            frame.info.update(
                total_charge=-1.0, total_spin=2.0, external_field=[0.1, 0.0, 0.0]
            )
        else:
            frame = Atoms("OH2", positions=WATER)
        frame.info["REF_energy"] = -1.0
        frame.arrays["REF_forces"] = np.zeros((len(frame), 3))
        frames.append(frame)
    path = tmp_path / "charged.xyz"
    write(path, frames, format="extxyz")
    config = polar_configuration(tmp_path)
    head = config.data.heads["default"].model_copy(
        update={
            "train_file": str(path),
            "e0s": E0sTable(values={1: 0.0, 8: 0.0}),
        }
    )
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={"heads": {"default": head}, "valid_fraction": 0.5}
            )
        }
    )
    data, _ = built(config)
    seen = set()
    batches = [
        *data.train_loader.batches(0, drop_last=False),
        *(batch for loader in data.valid_loaders.values() for batch in loader),
    ]
    for batch in batches:
        graph = batch.graph
        for index, size in enumerate(torch.diff(graph["ptr"]).tolist()):
            seen.add(
                (
                    size,
                    float(graph["total_charge"][index]),
                    float(graph["total_spin"][index]),
                    tuple(graph["external_field"][index].tolist()),
                )
            )
    assert seen == {(3, 0.0, 1.0, (0.0, 0.0, 0.0)), (2, -1.0, 2.0, (0.1, 0.0, 0.0))}


# ---------------------------------------------------------------------------
# The solver, as model state
# ---------------------------------------------------------------------------


@fp64_only
def test_the_checkpoint_records_the_solver_and_the_solve(trained):
    record = json.loads(trained.read_text())["config"]["electrostatics"]
    assert record["solver"] == "reference" and record["bit_parity"] is True
    assert record["descriptor"]["periodicity_profile"] == "molecular"
    assert record["descriptor"]["realspace_method"] == "finite_difference"


@fp64_only
def test_a_trained_model_reads_back_with_its_own_solver(trained):
    deployed = load_deployed(trained)
    model = deployed.model
    assert isinstance(model, PolarModel) and model.solver == "reference"


@fp64_only
def test_a_solver_that_changes_the_numbers_is_not_swapped_in(trained, registered):
    with pytest.raises(SolverSubstitutionError, match="'accelerated'"):
        load_deployed(trained, solver="accelerated")


@fp64_only
def test_two_solvers_with_the_reference_s_numbers_swap_freely(trained, registered):
    reference = load_deployed(trained)
    twin = load_deployed(trained, solver="twin")
    assert twin.model.solver == "twin"
    structure = build_graph(WATER + 5.0, [8, 1, 1])
    structure["element_index"] = torch.tensor([1, 0, 0])
    structure.update(
        total_charge=torch.tensor([0.0]),
        total_spin=torch.tensor([1.0]),
        external_field=torch.zeros(1, 3, dtype=torch.float64),
    )
    energies = [
        model.compute(dict(structure), compute=()).total_energy
        for model in (reference, twin)
    ]
    assert energies[0] is not None and energies[1] is not None
    assert torch.equal(energies[0], energies[1])


@fp64_only
def test_a_model_trained_with_another_solver_is_not_loaded_with_the_reference(
    trained, registered
):
    document = json.loads(trained.read_text())
    record = document["config"]["electrostatics"]
    record.update(solver="accelerated", bit_parity=False)
    document["config"]["config"]["resolved"]["electrostatics"]["solver"] = "accelerated"
    trained.write_text(json.dumps(document))
    with pytest.raises(SolverSubstitutionError, match="'accelerated'"):
        load_deployed(trained, solver="reference")


@fp64_only
def test_a_solve_that_rebuilds_differently_is_refused(trained):
    """What a default that moved between versions would look like."""
    document = json.loads(trained.read_text())
    document["config"]["electrostatics"]["descriptor"]["kspace_cutoff"] = 1.0
    trained.write_text(json.dumps(document))
    with pytest.raises(CheckpointError, match="kspace_cutoff"):
        load_deployed(trained)
