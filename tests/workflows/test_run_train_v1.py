"""The tiny task, trained through the console script on the v1 engine.

The stage tests exercise the pipeline in process. This is the same run reached
the way a user reaches it: one console script, one configuration file, a
subprocess. It is the last thing between the three stages and a person.

It is deliberately not a comparison against legacy. The two engines do not
produce the same numbers yet and are not meant to at this point; what is
checked is that the v1 side runs at all from a command line, writes a
checkpoint that carries its configuration, and gets better while it does.
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from tests.helpers import cli_command, run_train

#: Both halves of the dispatch have to be importable: the launcher chooses the
#: engine and the v1 stack is what it chooses. The legacy-only CI jobs install
#: neither, and there these tests are not a failure, they are absent.
needs_the_v1_engine = pytest.mark.skipif(
    importlib.util.find_spec("mace_launcher") is None
    or importlib.util.find_spec("mace_torch") is None,
    reason="the launcher or the v1 stack is not installed in this environment",
)

#: The two references the isolated atoms carry, and the geometry the molecules
#: are jittered around.
HYDROGEN, OXYGEN = -13.6, -2040.0
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

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
  max_num_epochs: 4
  batch_size: 4
  valid_batch_size: 4
  lr: 0.02
  scheduler: {{kind: {{kind: constant}}}}
loss: {{weights: {{energy: 1.0, forces: 10.0}}}}
"""


def tiny_task(directory: Path) -> Path:
    """A dozen waters and the two isolated atoms they reference."""
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
    write(directory / "train.xyz", frames)
    config = directory / "run.yaml"
    config.write_text(
        CONFIG.format(work_dir=directory, train_file=directory / "train.xyz")
    )
    return config


def variant(config: Path, settings: dict) -> Path:
    """The configuration with some values changed, as a file beside it.

    The engine takes a file and a few flags, with no dotted overrides, so a
    run that differs from another in a setting without a flag gets a file of
    its own. The run name and directory are the original's, which is what
    lets a restart find the first run's checkpoints.
    """
    import yaml
    from mace_core.cli import set_value

    document = yaml.safe_load(config.read_text())
    for dotted_path, value in settings.items():
        set_value(document, dotted_path, value)
    written = config.with_name(
        f"{config.stem}-{len(list(config.parent.glob('*.yaml')))}.yaml"
    )
    written.write_text(yaml.safe_dump(document))
    return written


def on_v1(*arguments: str) -> subprocess.CompletedProcess:
    """One command line on the v1 engine, launched the way every test does.

    The argv prefix comes from the shared runner rather than being written
    here, so the launcher stays known in one module. The engine is named
    explicitly instead of taken from `MACE_ENGINE`: this file is about the v1
    side specifically, not about whichever side the suite is being re-run on.
    """
    return subprocess.run(
        [*cli_command(run_train), "--engine", "v1", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def train_on_v1(directory: Path) -> subprocess.CompletedProcess:
    """One run of the tiny task."""
    return on_v1("--config", str(tiny_task(directory)))


@needs_the_v1_engine
def test_the_tiny_task_trains_on_the_v1_engine(tmp_path):
    finished = train_on_v1(tmp_path)
    assert finished.returncode == 0, finished.stderr
    assert (tmp_path / "tiny.safetensors").is_file()
    assert (tmp_path / "tiny.json").is_file()


@needs_the_v1_engine
def test_the_v1_checkpoint_carries_the_configuration_that_wrote_it(tmp_path):
    """A checkpoint holding only weights needs its command line to be rebuilt,
    and that is the thing least likely to still exist."""
    assert train_on_v1(tmp_path).returncode == 0
    sidecar = json.loads((tmp_path / "tiny.json").read_text())
    resolved = sidecar["config"]["config"]["resolved"]
    assert resolved["model"]["r_max"] == 5.0
    assert resolved["training"]["max_num_epochs"] == 4


@needs_the_v1_engine
def test_the_v1_run_reports_the_epoch_it_chose(tmp_path):
    """The run says which epoch the checkpoint holds. A run that wrote one and
    never said so leaves nothing to tell two checkpoints apart."""
    finished = train_on_v1(tmp_path)
    assert finished.returncode == 0
    assert "Best epoch" in finished.stderr


@needs_the_v1_engine
def test_a_legacy_command_line_on_v1_says_it_has_not_moved(tmp_path):
    """Until the flag port lands. It is a refusal rather than a crash, which
    is what lets the black-box suite skip instead of failing."""
    finished = on_v1(
        "--name",
        "tiny",
        "--train_file",
        str(tmp_path / "absent.xyz"),
        "--r_max",
        "5.0",
    )
    assert finished.returncode != 0
    assert "not yet available on v1 engine" in finished.stderr
    assert "--r_max" in finished.stderr


MULTIHEAD_CONFIG = """
runtime: {{work_dir: {work_dir}, name: tiny, seed: 1, error_table: PerAtomRMSE}}
data:
  heads:
    small: {{train_file: {small_file}, e0s: {{kind: isolated_atoms}}}}
    large: {{train_file: {large_file}, e0s: {{kind: isolated_atoms}}}}
  valid_fraction: 0.25
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
  max_num_epochs: 2
  batch_size: 2
  valid_batch_size: 2
  lr: 0.02
  head_balancing: balanced
  scheduler: {{kind: {{kind: constant}}}}
loss: {{weights: {{energy: 1.0, forces: 10.0}}}}
"""


def write_frames(path: Path, count: int, seed: int) -> Path:
    """A few waters and the two isolated atoms they reference."""
    generator = np.random.default_rng(seed)
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


def multihead_task(directory: Path) -> Path:
    """Two heads whose training files differ in size by a factor of four."""
    config = directory / "multihead.yaml"
    config.write_text(
        MULTIHEAD_CONFIG.format(
            work_dir=directory,
            small_file=write_frames(directory / "small.xyz", 4, seed=0),
            large_file=write_frames(directory / "large.xyz", 16, seed=1),
        )
    )
    return config


@needs_the_v1_engine
def test_a_multihead_task_trains_on_the_v1_engine(tmp_path):
    finished = on_v1("--config", str(multihead_task(tmp_path)))
    assert finished.returncode == 0, finished.stderr
    assert (tmp_path / "tiny.safetensors").is_file()


@needs_the_v1_engine
def test_a_multihead_run_reports_every_head_separately(tmp_path):
    """One averaged row cannot say which head got worse, so there is a row per
    head per split and the epoch lines name the head they are about."""
    finished = on_v1("--config", str(multihead_task(tmp_path)))
    assert finished.returncode == 0, finished.stderr
    for row in ("train_small", "train_large", "valid_small", "valid_large"):
        assert row in finished.stderr
    assert "head small" in finished.stderr
    assert "head large" in finished.stderr


@needs_the_v1_engine
def test_a_multihead_run_prints_an_error_table(tmp_path):
    finished = on_v1("--config", str(multihead_task(tmp_path)))
    assert finished.returncode == 0, finished.stderr
    assert "RMSE E / meV / atom" in finished.stderr


FINETUNE_CONFIG = """
runtime: {{work_dir: {work_dir}, name: tuned, seed: 2, error_table: PerAtomRMSE}}
finetune: {{foundation_model: {foundation}}}
data:
  heads:
    replay: {{train_file: {replay_file}, e0s: {{kind: foundation}}, weight: 0.5}}
    target: {{train_file: {target_file}, e0s: {{kind: isolated_atoms}}}}
  valid_fraction: 0.25
  pin_memory: false
  skip_evaluate_heads: [replay]
model:
  observables: [energy, forces]
training:
  max_num_epochs: 2
  batch_size: 2
  valid_batch_size: 2
  lr: 0.01
  scheduler: {{kind: {{kind: constant}}}}
"""


def fine_tune(directory: Path) -> subprocess.CompletedProcess:
    """A foundation trained through the console script, then a fine-tune of
    it through the same script, from a configuration file naming it."""
    foundation = directory / "foundation"
    foundation.mkdir()
    trained = on_v1("--config", str(tiny_task(foundation)))
    assert trained.returncode == 0, trained.stderr
    config = directory / "tune.yaml"
    config.write_text(
        FINETUNE_CONFIG.format(
            work_dir=directory,
            foundation=foundation / "tiny.safetensors",
            replay_file=foundation / "train.xyz",
            target_file=write_frames(directory / "target.xyz", 8, seed=4),
        )
    )
    return on_v1("--config", str(config))


@needs_the_v1_engine
def test_a_fine_tune_runs_from_the_console_script(tmp_path):
    finished = fine_tune(tmp_path)
    assert finished.returncode == 0, finished.stderr
    sidecar = json.loads((tmp_path / "tuned.json").read_text())
    record = sidecar["config"]
    assert [parent["role"] for parent in record["parents"]] == ["initial_weights"]
    assert set(record["heads"]) == {"replay", "target"}


@needs_the_v1_engine
def test_a_fine_tune_leaves_the_replay_head_out_of_its_table(tmp_path):
    finished = fine_tune(tmp_path)
    assert finished.returncode == 0, finished.stderr
    assert "valid_target" in finished.stderr
    assert "valid_replay" not in finished.stderr


#: A run that continues the newest checkpoint of the same name.
RESTART = {"runtime.restart_latest": True}


def evaluated_epochs(stderr: str) -> list[int]:
    return sorted(
        {
            int(line.split("epoch ")[1].split(",")[0])
            for line in stderr.splitlines()
            if ", head " in line and "epoch " in line
        }
    )


@needs_the_v1_engine
def test_restart_latest_continues_the_run(tmp_path):
    """Two epochs, then the same run asked for four and told to restart from
    its newest checkpoint: it evaluates epochs two and three only."""
    config = tiny_task(tmp_path)
    first = on_v1("--config", str(config), "--max_num_epochs", "2")
    assert first.returncode == 0, first.stderr
    assert evaluated_epochs(first.stderr) == [0, 1]

    resumed = on_v1("--config", str(variant(config, RESTART)))
    assert resumed.returncode == 0, resumed.stderr
    assert "Resuming from" in resumed.stderr
    assert evaluated_epochs(resumed.stderr) == [2, 3]


@needs_the_v1_engine
def test_restart_latest_with_nothing_to_restart_from_starts_fresh(tmp_path):
    """The frozen tree's contract: a restart with no checkpoint is a new run,
    and it says so."""
    finished = on_v1("--config", str(variant(tiny_task(tmp_path), RESTART)))
    assert finished.returncode == 0, finished.stderr
    assert "starts at epoch 0" in finished.stderr
    assert evaluated_epochs(finished.stderr) == [0, 1, 2, 3]


@needs_the_v1_engine
def test_a_run_keeps_only_its_newest_run_checkpoint_by_default(tmp_path):
    finished = train_on_v1(tmp_path)
    assert finished.returncode == 0, finished.stderr
    assert sorted(path.name for path in tmp_path.glob("tiny.run-*")) == [
        "tiny.run-000004.json",
        "tiny.run-000004.safetensors",
    ]


#: The full-batch regime, named where the optimizer is.
LBFGS = {"training.optimizer": {"kind": "lbfgs"}}


@needs_the_v1_engine
def test_an_lbfgs_run_trains_from_the_console_script(tmp_path):
    """The v1 counterpart of the frozen tree's L-BFGS workflow test, which
    also turns on an average and a second stage. Both are refused beside
    L-BFGS here, so this runs the regime alone."""
    finished = on_v1("--config", str(variant(tiny_task(tmp_path), LBFGS)))
    assert finished.returncode == 0, finished.stderr
    assert evaluated_epochs(finished.stderr) == [0, 1, 2, 3]
    assert (tmp_path / "tiny.safetensors").is_file()


@needs_the_v1_engine
def test_a_mini_batch_run_continued_under_lbfgs_says_its_optimizer_is_new(tmp_path):
    """The frozen tree's way of finishing a run with L-BFGS: train, then
    restart with the flag. The restart reports that the optimizer's state did
    not carry over rather than loading one optimizer's state into another."""
    config = tiny_task(tmp_path)
    first = on_v1("--config", str(config), "--max_num_epochs", "2")
    assert first.returncode == 0, first.stderr

    resumed = on_v1("--config", str(variant(config, {**LBFGS, **RESTART})))
    assert resumed.returncode == 0, resumed.stderr
    assert "Resumed with a new optimizer" in resumed.stderr
    assert "written by Adam and the run resumes with LBFGS" in resumed.stderr
    assert evaluated_epochs(resumed.stderr) == [2, 3]


@needs_the_v1_engine
def test_an_averaged_run_continued_under_lbfgs_starts_from_the_average(tmp_path):
    """The frozen tree's usual pair: an averaged mini-batch run, then L-BFGS,
    which refuses an average, from the model the first run ended on."""
    config = tiny_task(tmp_path)
    averaged = variant(config, {"training.ema.enabled": True})
    first = on_v1("--config", str(averaged), "--max_num_epochs", "2")
    assert first.returncode == 0, first.stderr

    resumed = on_v1("--config", str(variant(config, {**LBFGS, **RESTART})))
    assert resumed.returncode == 0, resumed.stderr
    assert "training continues from the averaged weights" in resumed.stderr
    assert evaluated_epochs(resumed.stderr) == [2, 3]
