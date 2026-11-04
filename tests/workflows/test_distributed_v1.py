"""Two processes on Gloo, through the console script, on the v1 engine.

Both ranks are started directly with a static rendezvous on the loopback
address and a free port, and Gloo is told which interface to use, so the test
never depends on how the host resolves its own name. The run crosses into its
second stage, and then a second pair of processes resumes it.

What is checked is the rank discipline: both processes finish, rank zero alone
writes, one set of checkpoints is left behind, and a distributed run resumes
from them.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.golden.harness import FP64_CPU_REFERENCE
from tests.helpers import cli_command, run_train
from tests.workflows.test_run_train_v1 import (
    RESTART,
    evaluated_epochs,
    needs_the_v1_engine,
    on_v1,
    optimizer_steps,
    tiny_task,
    training_records,
    variant,
    write_frames,
)

#: Written into the parsed file rather than appended to it: a second
#: ``training:`` block in the YAML would replace the first one whole.
DISTRIBUTED = {
    "runtime.distributed": True,
    "runtime.launcher": "torchrun",
    "runtime.log_level": "DEBUG",
    "training.stage_two.enabled": True,
    "training.stage_two.start_epoch": 2,
    "training.stage_two.lr": 0.005,
}


#: Seconds a rank may take. The whole run takes a few.
RANK_TIMEOUT = 180


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def both_ranks(
    config: Path, *arguments: str, settings: dict | None = None
) -> list[subprocess.CompletedProcess]:
    """Start the two ranks together and wait for both."""
    config = variant(config, {**DISTRIBUTED, **(settings or {})})
    port = free_port()
    processes = []
    for rank in (0, 1):
        environment = {
            **os.environ,
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
            "OMP_NUM_THREADS": "1",
        }
        processes.append(
            subprocess.Popen(
                [
                    *cli_command(run_train),
                    "--engine",
                    "v1",
                    "--config",
                    str(config),
                    *arguments,
                ],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    # A rank that dies leaves the other waiting at the next collective for as
    # long as the process group's own timeout, so both are killed together
    # rather than one being left behind after the test gives up.
    finished = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=RANK_TIMEOUT)
            finished.append(
                subprocess.CompletedProcess(
                    process.args, process.returncode, stdout, stderr
                )
            )
    except subprocess.TimeoutExpired:
        for process in processes:
            process.kill()
        outputs = [process.communicate()[1] for process in processes]
        raise AssertionError(
            f"the ranks did not finish within {RANK_TIMEOUT} s, which is one "
            f"rank waiting on another that stopped:\n" + "\n---\n".join(outputs)
        ) from None
    return finished


def assert_same_state(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_state(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_same_state(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(
            expected, abs=FP64_CPU_REFERENCE.atol, rel=FP64_CPU_REFERENCE.rtol
        )
    else:
        assert actual == expected


@needs_the_v1_engine
def test_two_ranks_train_and_only_rank_zero_writes(tmp_path):
    ranks = both_ranks(tiny_task(tmp_path))
    for rank, finished in enumerate(ranks):
        assert finished.returncode == 0, f"rank {rank}:\n{finished.stderr}"
    assert "Rank 0 wrote" in ranks[0].stderr
    assert "Rank 1 wrote" not in ranks[1].stderr
    assert evaluated_epochs(ranks[0].stderr) == [0, 1, 2, 3]
    assert optimizer_steps(tmp_path) == {0: 1, 1: 1, 2: 1, 3: 1}
    for finished in ranks:
        assert "Epoch 2 starts the stage 'stage_two'" in finished.stderr
    assert sorted(path.name for path in tmp_path.glob("tiny*")) == [
        "tiny.json",
        "tiny.run-000004.json",
        "tiny.run-000004.safetensors",
        "tiny.safetensors",
    ]


@needs_the_v1_engine
@pytest.mark.parametrize("stop_epoch", [1, 3])
def test_a_distributed_run_resumes_from_its_checkpoints(tmp_path, stop_epoch):
    """Resume before/after the stage switch preserves raw and averaged state."""
    from ase.io import read
    from mace_torch.calculators import MACECalculator
    from safetensors.torch import load_file

    config = tiny_task(tmp_path)
    settings = {
        "training.ema.enabled": True,
        "training.scheduler.kind": {"kind": "exponential", "gamma": 0.8},
    }
    first = both_ranks(config, "--max_num_epochs", str(stop_epoch), settings=settings)
    for rank, finished in enumerate(first):
        assert finished.returncode == 0, f"rank {rank}:\n{finished.stderr}"

    earlier_records = training_records(tmp_path)
    resumed = both_ranks(config, settings={**settings, **RESTART})
    for rank, finished in enumerate(resumed):
        assert finished.returncode == 0, f"rank {rank}:\n{finished.stderr}"
        assert "Resuming from" in finished.stderr
    assert evaluated_epochs(resumed[0].stderr) == list(range(stop_epoch, 4))
    assert training_records(tmp_path)[:len(earlier_records)] == earlier_records
    assert "initial validation" not in resumed[0].stderr
    assert optimizer_steps(tmp_path) == {0: 1, 1: 1, 2: 1, 3: 1}

    uninterrupted = tmp_path / "uninterrupted"
    for rank, finished in enumerate(both_ranks(
        config, settings={**settings, "runtime.work_dir": str(uninterrupted)},
    )):
        assert finished.returncode == 0, f"rank {rank}:\n{finished.stderr}"

    actual = json.loads((tmp_path / "tiny.run-000004.json").read_text())
    expected = json.loads((uninterrupted / "tiny.run-000004.json").read_text())
    assert expected["ema"] is not None and expected["scheduler"] is not None
    for key in ("state", "optimizer", "scheduler", "ema"):
        assert_same_state(actual[key], expected[key])
    assert_same_state(training_records(tmp_path), training_records(uninterrupted))
    for filename in ("tiny.run-000004.safetensors", "tiny.safetensors"):
        actual, expected = load_file(tmp_path / filename), load_file(uninterrupted / filename)
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_allclose(
                actual[key], expected[key],
                atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol,
            )
    atoms = read(tmp_path / "train.xyz", index=-1)
    atoms.calc = MACECalculator(uninterrupted / "tiny.safetensors", device="cpu")
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = MACECalculator(tmp_path / "tiny.safetensors", device="cpu")
    np.testing.assert_allclose(
        atoms.get_potential_energy(), energy,
        atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol,
    )
    np.testing.assert_allclose(
        atoms.get_forces(), forces,
        atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol,
    )


@needs_the_v1_engine
def test_two_rank_update_matches_one_concatenated_batch(tmp_path):
    """Two disjoint five-structure shards have the same update as ten at once."""
    from safetensors.torch import load_file

    config = tiny_task(tmp_path)
    settings = {
        "training.max_num_epochs": 1,
        "training.stage_two.enabled": False,
        "training.batch_size": 5,
        "training.ema.enabled": True,
    }
    for rank, finished in enumerate(both_ranks(config, settings=settings)):
        assert finished.returncode == 0, f"rank {rank}:\n{finished.stderr}"
    single_dir = tmp_path / "single"
    single = on_v1("--config", str(variant(config, {
        **settings, "runtime.work_dir": str(single_dir), "training.batch_size": 10,
    })))
    assert single.returncode == 0, single.stderr
    assert optimizer_steps(tmp_path) == optimizer_steps(single_dir) == {0: 1}
    together = [row for row in training_records(tmp_path) if row["mode"] == "valid"]
    alone = [row for row in training_records(single_dir) if row["mode"] == "valid"]
    assert len(together) == len(alone) == 2
    for actual, expected in zip(together, alone, strict=True):
        assert actual == pytest.approx(
            expected, abs=FP64_CPU_REFERENCE.atol, rel=FP64_CPU_REFERENCE.rtol
        )
    # Includes the raw weights, Adam's moments and EMA shadow, not just E/F.
    actual = load_file(tmp_path / "tiny.run-000001.safetensors")
    expected = load_file(single_dir / "tiny.run-000001.safetensors")
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(
            actual[key], expected[key],
            atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol,
        )


@needs_the_v1_engine
@pytest.mark.parametrize("optimizer,structures,valid_batch_size", [
    ("adam", 2, 4), ("lbfgs", 2, 4), ("lbfgs", 4, 1),
])
def test_a_dataset_smaller_than_the_world_size_trains(tmp_path, optimizer, structures, valid_batch_size):
    from safetensors.torch import load_file

    config = tiny_task(tmp_path)
    write_frames(tmp_path / "train.xyz", structures, seed=0)
    # One training structure across two ranks: mini-batch sampling pads;
    # full-batch training lets the empty rank contribute a zero gradient.
    settings = {
        "data.valid_fraction": (structures - 1) / structures,
        "training.batch_size": 1,
        "training.valid_batch_size": valid_batch_size,
        "training.max_num_epochs": 1,
        "training.optimizer": {"kind": optimizer},
        "training.stage_two.enabled": False,
    }
    ranks = both_ranks(config, settings=settings)
    for rank, finished in enumerate(ranks):
        assert finished.returncode == 0, f"rank {rank}:\n{finished.stderr}"
    assert optimizer_steps(tmp_path) == {0: 1}
    together = [row for row in training_records(tmp_path) if row["mode"] == "valid"]
    single_dir = tmp_path / "single"
    single = on_v1("--config", str(variant(config, {
        **settings, "runtime.work_dir": str(single_dir),
    })))
    assert single.returncode == 0, single.stderr
    assert optimizer_steps(single_dir) == {0: 1}
    actual = json.loads((tmp_path / "tiny.run-000001.json").read_text())
    expected = json.loads((single_dir / "tiny.run-000001.json").read_text())
    for key in ("state", "optimizer", "scheduler", "ema"):
        assert_same_state(actual[key], expected[key])
    # Includes optimizer state and raw weights, even when a rank has no data.
    actual = load_file(tmp_path / "tiny.run-000001.safetensors")
    expected = load_file(single_dir / "tiny.run-000001.safetensors")
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(
            actual[key], expected[key],
            atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol,
            err_msg=key,
        )
    alone = [row for row in training_records(single_dir) if row["mode"] == "valid"]
    assert [row["epoch"] for row in together] == [None, 0]
    assert [row["epoch"] for row in alone] == [None, 0]
    for distributed_row, single_row in zip(together, alone, strict=True):
        assert distributed_row.keys() == single_row.keys()
        for name, value in single_row.items():
            if isinstance(value, (int, float)):
                assert distributed_row[name] == pytest.approx(
                    value, abs=FP64_CPU_REFERENCE.atol, rel=FP64_CPU_REFERENCE.rtol
                ), name
            else:
                assert distributed_row[name] == value
    for row_name in ("train_default", "valid_default"):
        together_row = next(line for line in ranks[0].stderr.splitlines() if f"| {row_name}" in line)
        alone_row = next(line for line in single.stderr.splitlines() if f"| {row_name}" in line)
        assert together_row == alone_row
