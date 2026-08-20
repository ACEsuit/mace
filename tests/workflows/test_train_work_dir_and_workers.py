"""`--work_dir` and `--num_workers`, neither of which any run passed.

`--work_dir` is not one of the four output directories; it is where the training
split records the validation indices it chose, and it only writes that file when
the split is large enough to be worth recording (ten or more configurations).
Every existing fixture trains on a handful, so the branch never ran.

`--num_workers` sizes the DataLoader worker pool at four places in `run_train`.
A smoke run with workers is the only end-to-end statement available, so it is
paired with a source check that no loader is built without it: a fifth loader
added later is exactly how a knob like this stops applying to half the run.
"""

import ast
import inspect
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms

from mace.cli import run_train as run_train_module
from tests.helpers import base_mace_params, run_mace_train


@pytest.fixture(name="many_configs")
def fixture_many_configs():
    """Enough configurations that a 10% validation split is ten or more, which is
    the condition for the indices being written to `--work_dir` at all."""
    rng = np.random.default_rng(3)
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        cell=[6.0] * 3,
        pbc=True,
    )
    configs = []
    for numbers in ([8], [1]):
        isolated = Atoms(numbers=numbers, positions=[[0, 0, 0]], cell=[6] * 3)
        isolated.info["REF_energy"] = 0.0
        isolated.info["config_type"] = "IsolatedAtom"
        configs.append(isolated)
    for _ in range(120):
        frame = water.copy()
        frame.positions += rng.normal(0.1, size=frame.positions.shape)
        frame.info["REF_energy"] = float(rng.normal(0.1))
        frame.new_array("REF_forces", rng.normal(0.1, size=frame.positions.shape))
        configs.append(frame)
    return configs


def train(tmp_path, configs, **extra):
    ase.io.write(tmp_path / "fit.xyz", configs)
    params = base_mace_params()
    params.update(
        {
            "name": "wd",
            "hidden_irreps": "8x0e",
            "checkpoints_dir": str(tmp_path / "ckpt"),
            "model_dir": str(tmp_path / "model"),
            "results_dir": str(tmp_path / "results"),
            "log_dir": str(tmp_path / "logs"),
            "train_file": str(tmp_path / "fit.xyz"),
            "max_num_epochs": 1,
            "valid_fraction": 0.1,
            "seed": 123,
            "loss": "weighted",
        }
    )
    params.pop("swa", None)
    params.pop("start_swa", None)
    params.update(extra)
    return run_mace_train(params)


def test_the_validation_indices_land_in_the_work_dir(tmp_path, many_configs):
    """The observable effect of the flag: the file appears where it points."""
    work = tmp_path / "elsewhere"
    work.mkdir()

    train(tmp_path, many_configs, work_dir=str(work))

    written = list(work.glob("*valid_indices_*.txt"))
    assert written, f"nothing in {work}: {sorted(p.name for p in work.iterdir())}"
    indices = [int(line) for line in written[0].read_text().split()]
    assert len(indices) >= 10
    assert all(0 <= i < len(many_configs) for i in indices)


def test_without_the_flag_the_indices_do_not_appear_there(tmp_path, many_configs):
    """Otherwise the test above would pass on a run that wrote them anywhere."""
    work = tmp_path / "elsewhere"
    work.mkdir()

    train(tmp_path, many_configs)

    assert not list(work.glob("*valid_indices_*.txt"))


def test_a_run_with_workers_trains(tmp_path, many_configs):
    """`--num_workers` spawns loader subprocesses, which is the part that breaks
    when the dataset or the collate function is not picklable."""
    result = train(tmp_path, many_configs, num_workers=2)

    assert result.returncode == 0
    assert (tmp_path / "model" / "wd.model").exists()


def test_every_dataloader_is_given_the_worker_count():
    """A loader built without `num_workers` silently ignores the flag for that
    part of the run, and no end-to-end assertion distinguishes that from a run
    that honoured it. Checked on the source for the same reason
    tests/unit/test_calculator_dtype_scope.py does: several of these branches
    need a distributed launch or an on-line dataset to reach.
    """
    tree = ast.parse(inspect.getsource(run_train_module))
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if not name.endswith("DataLoader"):
            continue
        keywords = {kw.arg for kw in node.keywords}
        if "num_workers" not in keywords:
            missing.append(f"{name} at line {node.lineno}")

    assert not missing, "DataLoader built without num_workers: " + ", ".join(missing)
