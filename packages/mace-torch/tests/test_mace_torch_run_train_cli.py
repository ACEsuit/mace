"""The command line, and the seam it is not allowed to break.

Two things are checked here. That a configuration file runs a training end to
end, which is what the entry point is for; and that the stages take typed
objects rather than a namespace, which is the property the whole three-stage
shape exists to have and the one that erodes quietly.
"""

from __future__ import annotations

import argparse
import inspect
import json
import typing

import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.stages import BuiltModel, DataBundle, TrainedModel
from mace_torch.cli import run_train
from mace_torch.train import run_data_stage, run_model_stage, run_train_stage

HYDROGEN, OXYGEN = -13.6, -2040.0
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

#: The file spelling, which is the kinds-as-key one: a kind is the key its
#: settings sit under, so a setting cannot be given to a kind that has none.
CONFIG = """
runtime: {{work_dir: {work_dir}, name: tiny, seed: 1}}
data:
  heads:
    default: {{train_file: {train_file}, e0s: {{isolated_atoms: {{}}}}}}
  valid_fraction: 0.2
  pin_memory: false
model:
  observables: [energy, forces]
  r_max: 5.0
  num_interactions: 2
  num_channels: 4
  hidden_irreps: 0e+1o
  max_ell: 1
  correlation: 2
  readout: {{mlp_irreps: 4x0e}}
training:
  max_num_epochs: 3
  batch_size: 4
  valid_batch_size: 4
  lr: 0.02
  scheduler: {{kind: {{constant: {{}}}}}}
loss: {{weights: {{energy: 1.0, forces: 10.0}}}}
"""


@pytest.fixture(name="run_directory")
def fixture_run_directory(tmp_path):
    """A tiny dataset and the configuration file that names it."""
    generator = np.random.default_rng(0)
    frames = []
    for number, energy in ((1, HYDROGEN), (8, OXYGEN)):
        atom = Atoms(numbers=[number], positions=[[0.0, 0.0, 0.0]])
        atom.info["REF_energy"] = energy
        atom.info["config_type"] = "IsolatedAtom"
        atom.arrays["REF_forces"] = np.zeros((1, 3))
        frames.append(atom)
    for index in range(12):
        molecule = Atoms(
            "OH2", positions=WATER + generator.normal(scale=0.03, size=(3, 3))
        )
        molecule.info["REF_energy"] = 2 * HYDROGEN + OXYGEN + 0.05 * index
        molecule.arrays["REF_forces"] = generator.normal(scale=0.1, size=(3, 3))
        frames.append(molecule)
    write(tmp_path / "train.xyz", frames)
    (tmp_path / "run.yaml").write_text(
        CONFIG.format(work_dir=tmp_path, train_file=tmp_path / "train.xyz")
    )
    return tmp_path


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


@fp64_only
def test_a_configuration_file_trains_a_model(run_directory):
    assert run_train.main(["--config", str(run_directory / "run.yaml")]) == 0
    assert (run_directory / "tiny.safetensors").is_file()
    assert (run_directory / "tiny.json").is_file()


@fp64_only
def test_the_checkpoint_carries_the_configuration_that_wrote_it(run_directory):
    run_train.main(["--config", str(run_directory / "run.yaml")])
    sidecar = json.loads((run_directory / "tiny.json").read_text())
    assert sidecar["config"]["config"]["resolved"]["model"]["r_max"] == 5.0


@fp64_only
def test_a_dotted_override_reaches_the_field_it_names(run_directory):
    config = run_train.parse(
        ["--config", str(run_directory / "run.yaml"), "--training.max_num_epochs", "2"]
    )
    assert config.training.max_num_epochs == 2


@fp64_only
def test_an_unknown_key_is_refused_rather_than_ignored(run_directory):
    """A misspelled setting that ran anyway would train something else."""
    with pytest.raises(Exception, match="training"):
        run_train.parse(
            ["--config", str(run_directory / "run.yaml"), "--training.lr_rate", "0.1"]
        )


@fp64_only
def test_the_run_returns_what_the_training_produced(run_directory):
    config = run_train.parse(["--config", str(run_directory / "run.yaml")])
    trained = run_train.run(config)
    assert isinstance(trained, TrainedModel)
    assert len(trained.history) == 3


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------

#: What each stage takes first and gives back. The orchestrator is the only
#: thing that sees all three, so this is where the chain is stated.
CHAIN = (
    (run_data_stage, ResolvedConfig, DataBundle),
    (run_model_stage, ResolvedConfig, BuiltModel),
    (run_train_stage, ResolvedConfig, TrainedModel),
)


def annotations_of(function):
    return typing.get_type_hints(function, include_extras=False)


def test_no_namespace_crosses_a_stage_boundary():
    """The frozen tree's `run()` reads 74 attributes off one and assigns back
    to it at 43 sites, so what a run does depends on the order of assignments.
    A namespace in any of these signatures would be that again."""
    for stage, _, _ in (*CHAIN, (run_train.run, None, None)):
        for name, annotation in annotations_of(stage).items():
            assert annotation is not argparse.Namespace, f"{stage.__name__}.{name}"
            assert "Namespace" not in str(annotation), f"{stage.__name__}.{name}"


def test_every_stage_takes_the_configuration_and_returns_its_own_object():
    for stage, first, returns in CHAIN:
        hints = annotations_of(stage)
        parameters = list(inspect.signature(stage).parameters)
        assert hints[parameters[0]] is first, stage.__name__
        assert typing.get_origin(hints["return"]) is returns, stage.__name__


def test_the_orchestrator_takes_a_resolved_config_and_nothing_else():
    """It connects the stages; a second input would be a decision made here."""
    parameters = inspect.signature(run_train.run).parameters
    assert list(parameters) == ["config"]
    assert annotations_of(run_train.run)["config"] is ResolvedConfig


def test_the_orchestrator_is_short_enough_to_read():
    """The frozen tree's is about 1,150 lines. The number is not the point; a
    body that grows past this is holding logic a stage should own."""
    body = inspect.getsource(run_train.run).splitlines()
    statements = [
        line
        for line in body
        if line.strip() and not line.strip().startswith(("#", '"', "'"))
    ]
    assert len(statements) < 40


def test_a_legacy_flag_says_it_has_not_moved_rather_than_failing_on_a_key():
    """Until the flag port lands, a legacy command line on this engine has not
    been tried rather than tried and broken, and the suite skips on it."""
    with pytest.raises(SystemExit) as caught:
        run_train.parse(["--name", "tiny", "--train_file", "none.xyz"])
    message = str(caught.value)
    assert run_train.NOT_MIGRATED in message
    assert "--name" in message and "--train_file" in message
    assert "--engine legacy" in message


def test_a_dotted_override_is_not_mistaken_for_a_legacy_flag(run_directory):
    """Every field sits inside a section, so a dot is what tells them apart."""
    config = run_train.parse(
        ["--config", str(run_directory / "run.yaml"), "--runtime.seed", "9"]
    )
    assert config.runtime.seed == 9
