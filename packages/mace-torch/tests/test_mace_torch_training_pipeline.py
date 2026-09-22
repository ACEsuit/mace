"""The three stages, end to end, on a task small enough to run in a test.

What is checked is not that the numbers are right, which the parity harness
owns, but that the run does the things whose absence is silent: the E0s are
resolved once and recorded, the model is built with weights it can move, the
loss goes down, and the checkpoint carries enough to rebuild what wrote it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import load_default_catalogue
from mace_torch.train import (
    RunState,
    run_data_stage,
    run_model_stage,
    run_train_stage,
)

CATALOGUE = load_default_catalogue()

#: The two references the isolated-atom structures carry.
HYDROGEN, OXYGEN = -13.6, -2040.0

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def write_dataset(path, count: int = 12):
    """A handful of waters, plus the two isolated atoms they reference."""
    generator = np.random.default_rng(0)
    frames = []
    for number, energy in ((1, HYDROGEN), (8, OXYGEN)):
        atom = Atoms(numbers=[number], positions=[[0.0, 0.0, 0.0]])
        atom.info["REF_energy"] = energy
        atom.info["config_type"] = "IsolatedAtom"
        atom.arrays["REF_forces"] = np.zeros((1, 3))
        frames.append(atom)
    for index in range(count):
        molecule = Atoms(
            "OH2", positions=WATER + generator.normal(scale=0.03, size=(3, 3))
        )
        molecule.info["REF_energy"] = 2 * HYDROGEN + OXYGEN + 0.05 * index
        molecule.arrays["REF_forces"] = generator.normal(scale=0.1, size=(3, 3))
        frames.append(molecule)
    write(path, frames)
    return path


def configuration(tmp_path, **training):
    settings = {
        "max_num_epochs": 6,
        "batch_size": 4,
        "valid_batch_size": 4,
        "lr": 0.02,
        "scheduler": {"kind": {"kind": "constant"}},
        **training,
    }
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(tmp_path), "seed": 1},
            "data": {
                "heads": {
                    "default": {
                        "train_file": str(write_dataset(tmp_path / "train.xyz")),
                        "e0s": {"kind": "isolated_atoms"},
                    }
                },
                "valid_fraction": 0.2,
                "pin_memory": False,
            },
            "model": {
                "observables": ["energy", "forces"],
                "r_max": 5.0,
                "num_interactions": 2,
                "num_channels": 4,
                "hidden_irreps": "0e+1o",
                "max_ell": 1,
                "correlation": 2,
                "readout": {"mlp_irreps": "4x0e"},
            },
            "training": settings,
            "loss": {"weights": {"energy": 1.0, "forces": 10.0}},
        }
    )


@pytest.fixture(name="pipeline")
def fixture_pipeline(tmp_path):
    """The two stages before training, which every test below needs."""
    config = configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    return config, data, run_model_stage(config, data, CATALOGUE)


# ---------------------------------------------------------------------------
# The data stage
# ---------------------------------------------------------------------------


@fp64_only
def test_the_isolated_atom_energies_are_read_once_and_recorded(pipeline):
    """Read off the reference structures, which the stage keeps until it has."""
    _, data, _ = pipeline
    assert data.e0s.values["default"] == {1: HYDROGEN, 8: OXYGEN}
    assert data.e0_provenance["default"].kind == "isolated_atoms"


@fp64_only
def test_the_reference_structures_leave_the_training_set(pipeline):
    """They are references, not structures to fit. One left in is a molecule
    with no neighbours whose energy the model is asked to reproduce on top of
    the reference it just became."""
    _, data, _ = pipeline
    kept = data.train_loader.dataset.configurations
    assert all(len(item.atomic_numbers) > 1 for item in kept)


@fp64_only
def test_the_statistics_are_taken_against_those_energies(pipeline):
    _, data, _ = pipeline
    assert data.statistics.atomic_energies == {1: HYDROGEN, 8: OXYGEN}
    assert data.statistics.avg_num_neighbors == pytest.approx(2.0)


@fp64_only
def test_the_model_reads_the_bundle_s_statistics_rather_than_its_own(pipeline):
    """One measurement per run. Two would let the model be scaled by numbers
    the E0s were not taken against."""
    _, data, built = pipeline
    assert built.statistics is data.statistics


# ---------------------------------------------------------------------------
# The model stage
# ---------------------------------------------------------------------------


@fp64_only
def test_a_built_model_has_weights_it_can_move(pipeline):
    """The defect this guards is a model that trains and never changes: the
    ops allocate zeros, and a product of zeros has a zero gradient."""
    _, _, built = pipeline
    weighted = [
        name
        for name, parameter in built.model.named_parameters()
        if parameter.numel() and float(parameter.detach().abs().max()) > 0.0
    ]
    assert len(weighted) > 10


@fp64_only
def test_the_same_seed_builds_the_same_model(tmp_path):
    config = configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    first = run_model_stage(config, data, CATALOGUE)
    second = run_model_stage(config, data, CATALOGUE)
    for (_, left), (_, right) in zip(
        first.model.named_parameters(), second.model.named_parameters(), strict=True
    ):
        assert torch.equal(left, right)


@fp64_only
def test_the_model_produces_the_observables_that_were_requested(pipeline):
    _, data, built = pipeline
    assert built.outputs.derivatives == ("forces",)
    batch = next(iter(data.valid_loader))
    output = built.model(batch.graph, compute=("forces",))
    assert output.total_energy is not None
    assert output.forces is not None
    assert output.stress is None


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


@fp64_only
def test_the_tiny_task_trains(pipeline):
    """The whole point: a model built from a configuration gets better."""
    config, _, built = pipeline
    trained = run_train_stage(config, built)
    losses = [record.train_loss for record in trained.history]
    assert len(losses) == config.training.max_num_epochs
    assert losses[-1] < losses[0]


@fp64_only
def test_the_checkpoint_carries_what_wrote_it(tmp_path):
    config = configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    trained = run_train_stage(config, built, checkpoint_path=tmp_path / "model")
    assert trained.checkpoint_path is not None
    sidecar = json.loads(trained.checkpoint_path.read_text())
    recorded = sidecar["config"]["config"]["resolved"]
    assert recorded["model"]["r_max"] == 5.0
    assert recorded["training"]["batch_size"] == 4


@fp64_only
def test_a_resume_continues_rather_than_restarting(tmp_path):
    """A resume that silently started over would report a first epoch as if it
    carried on from somewhere."""
    config = configuration(tmp_path, max_num_epochs=2)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    run_train_stage(config, built, checkpoint_path=tmp_path / "model")

    later = configuration(tmp_path, max_num_epochs=4)
    continued = run_train_stage(
        later, built, checkpoint_path=tmp_path / "model", resume=True
    )
    assert [record.epoch for record in continued.history] == [2, 3]


@fp64_only
def test_a_resume_with_nothing_to_resume_from_says_so(tmp_path):
    config = configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    with pytest.raises(FileNotFoundError, match="no run to continue"):
        run_train_stage(config, built, checkpoint_path=tmp_path / "absent", resume=True)


@fp64_only
def test_a_dry_run_stops_before_any_epoch_and_writes_nothing(tmp_path):
    """It is exercised on the frozen tree only as `does not crash`."""
    config = configuration(tmp_path, dry_run=True)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    trained = run_train_stage(config, built, checkpoint_path=tmp_path / "model")
    assert trained.history == ()
    assert trained.checkpoint_path is None
    assert not (tmp_path / "model.safetensors").exists()


@fp64_only
def test_the_average_is_what_the_run_is_judged_and_left_with(tmp_path):
    """With EMA on, the reported validation number comes from parameters the
    optimizer never saw, and so does the model the run returns. A run where it
    did nothing would be indistinguishable from one without it."""
    config = configuration(tmp_path, ema={"enabled": True, "decay": 0.9})
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    trained = run_train_stage(config, built)
    assert all(
        record.evaluated_with_ema
        for record in trained.history
        if record.valid_loss is not None
    )

    plain_config = configuration(tmp_path)
    plain_data = run_data_stage(plain_config, CATALOGUE)
    plain = run_train_stage(
        plain_config, run_model_stage(plain_config, plain_data, CATALOGUE)
    )
    averaged = [record.valid_loss for record in trained.history]
    stepped = [record.valid_loss for record in plain.history]
    assert averaged != stepped


@fp64_only
def test_a_run_state_knows_where_it_stopped():
    assert RunState(epoch=3, best_valid_loss=0.5, best_epoch=2).epoch == 3
