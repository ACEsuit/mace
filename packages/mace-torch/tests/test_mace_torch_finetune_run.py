"""A fine-tune, end to end: foundation, replay head, new head, training.

The fine-tuning contract the frozen tree pins, in this stack's syntax: start
from a foundation model, keep a replay head beside a new one, train, and end
with a model holding both whose error on the new data went down. The foundation
is trained here on waters and methanes; the new data is waters labelled at a
different level of theory, shifted and scaled, which is what a new head exists
to absorb. The replay head reads the foundation's own training file, since the
published datasets need the network.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import load_default_catalogue
from mace_torch.finetune.foundation import read_foundation
from mace_torch.train import (
    ModelStageError,
    evaluate,
    metric_specs,
    run_data_stage,
    run_model_stage,
    run_train_stage,
    write_model,
)
from mace_torch.train.loss import build_loss

from test_mace_torch_foundation_transfer import dataset

CATALOGUE = load_default_catalogue()
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

ARCHITECTURE = {
    "observables": ["energy", "forces"],
    "r_max": 3.0,
    "num_channels": 4,
    "max_ell": 1,
    "num_interactions": 2,
    "correlation": 2,
}


@pytest.fixture(scope="module")
def foundation_path(tmp_path_factory):
    torch.set_default_dtype(torch.float64)
    directory = tmp_path_factory.mktemp("foundation")
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(directory), "seed": 3},
            "data": {
                "heads": {
                    "pbe": {
                        "train_file": str(dataset(directory / "train.xyz")),
                        "e0s": {"isolated_atoms": {}},
                    }
                },
                "valid_fraction": 0.2,
                "pin_memory": False,
            },
            "model": ARCHITECTURE,
            "training": {
                "max_num_epochs": 3,
                "batch_size": 4,
                "valid_batch_size": 4,
                "lr": 0.02,
                "scheduler": {"kind": {"kind": "constant"}},
            },
        }
    )
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    trained = run_train_stage(config, built)
    return write_model(directory / "model", trained.model, built.metadata), directory


def new_level_of_theory(path, count=16):
    """Waters at another level of theory: the same geometries, energies moved
    by a per-atom offset and forces scaled, with their own isolated atoms."""
    generator = np.random.default_rng(11)
    frames = []
    for number, energy in ((1, -13.1), (8, -2039.2)):
        atom = Atoms(numbers=[number], positions=[[0.0, 0.0, 0.0]])
        atom.info["REF_energy"] = energy
        atom.info["config_type"] = "IsolatedAtom"
        atom.arrays["REF_forces"] = np.zeros((1, 3))
        frames.append(atom)
    for index in range(count):
        molecule = Atoms("OH2", positions=WATER + generator.normal(scale=0.03, size=(3, 3)))
        molecule.info["REF_energy"] = 2 * -13.1 - 2039.2 + 0.3 + 0.08 * index
        molecule.arrays["REF_forces"] = 1.5 * generator.normal(scale=0.1, size=(3, 3))
        frames.append(molecule)
    write(path, frames)
    return path


def fine_tune_config(directory, foundation, **model):
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(directory), "seed": 5},
            "finetune": {"foundation_model": str(foundation)},
            "data": {
                "heads": {
                    "replay": {
                        "train_file": str(directory / "train.xyz"),
                        "e0s": {"foundation": {}},
                        "weight": 0.5,
                    },
                    # Its own references for the elements it has, and one for
                    # carbon, which only the replay data holds: every head
                    # needs an energy for every element of the run, and the
                    # frozen tree fails on that with a bare KeyError.
                    "target": {
                        "train_file": str(new_level_of_theory(directory / "target.xyz")),
                        "e0s": {"table": {"values": {1: -13.1, 6: -1030.0, 8: -2039.2}}},
                    },
                },
                "valid_fraction": 0.25,
                "pin_memory": False,
                "skip_evaluate_heads": ["replay"],
            },
            "model": {"observables": ["energy", "forces"], **model},
            "training": {
                "max_num_epochs": 6,
                "batch_size": 4,
                "valid_batch_size": 4,
                "lr": 0.01,
                "scheduler": {"kind": {"kind": "constant"}},
            },
        }
    )


def target_error(built, config):
    """The loss on the new head's validation set, the number that must drop."""
    loss = build_loss(built.outputs, config.loss)
    metrics = evaluate(
        built.model,
        built.data.valid_loaders["target"],
        loss,
        metric_specs(built.outputs),
    )
    return metrics["rmse_energy_per_atom"]


@fp64_only
def test_a_fine_tune_trains_both_heads_and_improves_the_new_one(foundation_path):
    path, directory = foundation_path
    foundation = read_foundation(path, CATALOGUE)
    config = fine_tune_config(directory, path)
    data = run_data_stage(config, CATALOGUE, foundation=foundation.context())
    built = run_model_stage(config, data, CATALOGUE, foundation=foundation)
    before = target_error(built, config)
    trained = run_train_stage(config, built)
    after = target_error(built, config)

    assert data.heads == ("replay", "target")
    readout = built.model.get_submodule("backbone.outputs.heads.energy")
    assert readout.num_heads == 2
    assert after < before, f"the new head's error went from {before:.4g} to {after:.4g}"
    assert trained.best_epoch is not None


@fp64_only
def test_the_fine_tune_records_its_parent(foundation_path):
    path, directory = foundation_path
    foundation = read_foundation(path, CATALOGUE)
    config = fine_tune_config(directory, path)
    data = run_data_stage(config, CATALOGUE, foundation=foundation.context())
    built = run_model_stage(config, data, CATALOGUE, foundation=foundation)
    (parent,) = built.metadata.parents
    assert parent.role == "initial_weights"
    assert parent.name == str(path)
    assert parent.metadata is not None
    assert parent.metadata.heads["pbe"].e0.values["C"] == pytest.approx(-1030.0)


@fp64_only
def test_the_architecture_is_the_foundations(foundation_path):
    """The run wrote none of it down, and gets the foundation's."""
    path, directory = foundation_path
    foundation = read_foundation(path, CATALOGUE)
    config = fine_tune_config(directory, path)
    data = run_data_stage(config, CATALOGUE, foundation=foundation.context())
    built = run_model_stage(config, data, CATALOGUE, foundation=foundation)
    recorded = built.metadata.config.resolved["model"]
    assert recorded["num_channels"] == 4
    assert recorded["r_max"] == 3.0


@fp64_only
def test_a_setting_that_contradicts_the_foundation_is_refused(foundation_path):
    """A run that asked for a cutoff of five and trained at three would never
    be told, so it is told now."""
    path, directory = foundation_path
    foundation = read_foundation(path, CATALOGUE)
    config = fine_tune_config(directory, path, r_max=5.0)
    data = run_data_stage(config, CATALOGUE, foundation=foundation.context())
    with pytest.raises(ModelStageError, match="r_max"):
        run_model_stage(config, data, CATALOGUE, foundation=foundation)


@fp64_only
def test_a_setting_that_agrees_with_the_foundation_is_accepted(foundation_path):
    path, directory = foundation_path
    foundation = read_foundation(path, CATALOGUE)
    config = fine_tune_config(directory, path, r_max=3.0)
    data = run_data_stage(config, CATALOGUE, foundation=foundation.context())
    run_model_stage(config, data, CATALOGUE, foundation=foundation)
