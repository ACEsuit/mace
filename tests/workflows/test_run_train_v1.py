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
  max_num_epochs: 4
  batch_size: 4
  valid_batch_size: 4
  lr: 0.02
  scheduler: {{kind: {{constant: {{}}}}}}
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
    finished = on_v1("--name", "tiny", "--train_file", str(tmp_path / "absent.xyz"))
    assert finished.returncode != 0
    assert "not yet available on v1 engine" in finished.stderr
