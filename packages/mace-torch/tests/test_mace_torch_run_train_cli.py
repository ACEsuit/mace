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
from mace_core import cli
from mace_core.config.base import ConfigError
from mace_core.config.resolved import ResolvedConfig
from mace_core.stages import BuiltModel, DataBundle, TrainedModel
from mace_torch.cli import run_train
from mace_torch.cli.commands import TRAIN_FLAGS
from mace_torch.train import run_data_stage, run_model_stage, run_train_stage

HYDROGEN, OXYGEN = -13.6, -2040.0
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

#: The file spelling, which is the kinds-as-key one: a kind is the key its
#: settings sit under, so a setting cannot be given to a kind that has none.
CONFIG = """
runtime: {{work_dir: {work_dir}, name: tiny, seed: 1}}
data:
  heads:
    default: {{train_file: {train_file}, e0s: {{kind: isolated_atoms}}}}
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
  scheduler: {{kind: {{kind: constant}}}}
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


def test_a_flag_sets_the_field_it_names(run_directory):
    config = run_train.parse(
        ["--config", str(run_directory / "run.yaml"), "--max_num_epochs", "2"]
    )
    assert config.training.max_num_epochs == 2
    assert config.training.lr == 0.02


def test_a_head_s_file_flag_goes_to_the_head_the_file_declares(run_directory):
    config = run_train.parse(
        ["--config", str(run_directory / "run.yaml"), "--valid_file", "valid.xyz"]
    )
    assert list(config.data.heads) == ["default"]
    assert str(config.data.heads["default"].valid_file) == "valid.xyz"
    assert config.data.heads["default"].train_file == run_directory / "train.xyz"


def test_an_unknown_key_is_refused_rather_than_ignored(run_directory):
    """A misspelled setting that ran anyway would train something else."""
    path = run_directory / "misspelled.yaml"
    path.write_text("training: {lr_rate: 0.1}\n")
    with pytest.raises(Exception, match=r"training\.lr_rate"):
        run_train.parse(["--config", str(path)])


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


def test_legacy_flags_reach_the_typed_configuration():
    config = run_train.parse(
        ["--name", "tiny", "--r_max", "5.0", "--num_channels", "8"]
    )
    assert config.runtime.name == "tiny"
    assert config.model.r_max == 5.0 and config.model.num_channels == 8


def test_a_legacy_yaml_reaches_the_typed_configuration(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("num_channels: 8\nr_max: 5.0\nE0s: average\n")
    config = run_train.parse(["--config", str(path), "--work_dir", str(tmp_path)])
    assert config.model.num_channels == 8
    assert config.data.heads["default"].e0s.kind == "average"
    assert config.runtime.work_dir == tmp_path


@pytest.mark.parametrize("per_head", [False, True])
def test_json_e0_flags_replace_the_previous_source(tmp_path, per_head):
    table = tmp_path / "e0s.json"
    table.write_text(json.dumps({1: -0.5, 8: -1.0}))
    head = {
        "train_file": "train.xyz",
        "readout_from": "source",
        "e0s": {"kind": "foundation", "head": "source", "missing": "zero"},
    }
    document = {
        "finetune": {"foundation_model": "foundation.safetensors"},
        "data": {"heads": {"target": head}},
    }
    if per_head:
        document["data"]["heads"]["memory"] = head
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document))
    flags = (
        ["--heads", json.dumps({"target": {"E0s": str(table)}})]
        if per_head
        else ["--E0s", str(table)]
    )
    config = run_train.parse(["--config", str(path), *flags])
    assert config.data.heads["target"].e0s.model_dump() == {
        "kind": "table",
        "values": {1: -0.5, 8: -1.0},
    }
    assert config.data.heads["target"].readout_from == "source"
    assert str(config.data.heads["target"].train_file) == "train.xyz"
    if per_head:
        assert config.data.heads["memory"].e0s.kind == "foundation"
        assert config.data.heads["memory"].e0s.head == "source"
    assert json.loads(path.read_text()) == document


@pytest.mark.parametrize("empty", ["{}", "{ }", {}])
def test_empty_foundation_options_in_legacy_yaml_are_compatibility_only(
    tmp_path, empty
):
    import yaml

    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump({"name": "fit", "foundation_model_kwargs": empty}))
    actual = run_train.parse(["--config", str(path)])
    expected = run_train.parse(["--name", "fit"])
    assert actual == expected


def test_a_dotted_override_is_refused_rather_than_skipped(run_directory):
    """There is no dotted override syntax. A dotted path is not a legacy flag
    either, so it is an error to fix, not a capability to wait for."""
    with pytest.raises(SystemExit) as caught:
        run_train.parse(
            ["--config", str(run_directory / "run.yaml"), "--runtime.seed", "9"]
        )
    message = str(caught.value)
    assert "--runtime.seed" in message
    assert "configuration file" in message
    assert run_train.NOT_MIGRATED not in message


# ---------------------------------------------------------------------------
# The mace program
# ---------------------------------------------------------------------------


def test_this_package_declares_train_and_export_config():
    found = cli.discover()
    for path in (("train",), ("model", "export-config")):
        assert found[path].command is not None, found[path].reason
        assert found[path].distribution == "mace-torch-v1"


def test_the_training_flags_are_the_legacy_spellings_and_few():
    """Every other setting is set in the file."""
    assert [flag.option for flag in TRAIN_FLAGS] == [
        "--name",
        "--seed",
        "--work_dir",
        "--device",
        "--train_file",
        "--valid_file",
        "--foundation_model",
        "--max_num_epochs",
    ]


def test_a_head_s_file_flag_names_a_default_head_when_the_file_has_none(tmp_path):
    config = run_train.parse(["--train_file", "train.xyz", "--name", "run"])
    assert list(config.data.heads) == ["default"]
    assert str(config.data.heads["default"].train_file) == "train.xyz"
    assert config.runtime.name == "run"


def test_a_head_s_file_flag_is_refused_when_the_file_has_several(tmp_path, capsys):
    path = tmp_path / "two.yaml"
    path.write_text("data: {heads: {pbe: {}, r2scan: {}}}\n")
    with pytest.raises(ConfigError, match=r"\['pbe', 'r2scan'\]"):
        run_train.parse(["--config", str(path), "--train_file", "train.xyz"])
    assert cli.main(["train", "--config", str(path), "--train_file", "x.xyz"]) == 2
    assert "Set train_file under the head" in capsys.readouterr().err


@fp64_only
def test_mace_train_trains_and_export_config_gives_back_its_configuration(
    run_directory,
):
    """The stored configuration, written out and read back, resolves to itself."""
    arguments = ["train", "--config", str(run_directory / "run.yaml")]
    assert cli.main([*arguments, "--name", "flagged", "--max_num_epochs", "1"]) == 0
    model = run_directory / "flagged.safetensors"
    assert model.is_file()
    stored = json.loads((run_directory / "flagged.json").read_text())
    resolved = stored["config"]["config"]["resolved"]
    assert resolved["runtime"]["name"] == "flagged"
    assert resolved["training"]["max_num_epochs"] == 1

    for suffix in (".yaml", ".json"):
        output = run_directory / f"exported{suffix}"
        command = ["model", "export-config", str(model), "--output", str(output)]
        assert cli.main(command) == 0
        again = ResolvedConfig.load(output)
        assert again.to_resolved_dict() == resolved
        assert again == ResolvedConfig.from_dict(resolved)

    with pytest.raises(SystemExit, match=r"cannot write '\.toml'"):
        cli.main(["model", "export-config", str(model), "--output", "out.toml"])


def test_export_config_names_a_file_that_is_not_a_model(tmp_path):
    stray = tmp_path / "stray.safetensors"
    stray.write_bytes(b"")
    with pytest.raises(SystemExit, match="no sidecar"):
        cli.main(["model", "export-config", str(stray), "--output", "out.yaml"])
