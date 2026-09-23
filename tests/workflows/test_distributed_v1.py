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

import os
import socket
import subprocess
import sys
from pathlib import Path

from tests.helpers import cli_command, run_train
from tests.workflows.test_run_train_v1 import (
    evaluated_epochs,
    needs_the_v1_engine,
    tiny_task,
)

#: Given as dotted overrides rather than appended to the file: a second
#: ``training:`` block in the YAML would replace the first one whole.
DISTRIBUTED = (
    "--runtime.distributed", "true",
    "--runtime.launcher", "torchrun",
    "--runtime.log_level", "DEBUG",
    "--training.stage_two.enabled", "true",
    "--training.stage_two.start_epoch", "2",
    "--training.stage_two.lr", "0.005",
)  # fmt: skip


#: Seconds a rank may take. The whole run takes a few.
RANK_TIMEOUT = 180


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def both_ranks(config: Path, *arguments: str) -> list[subprocess.CompletedProcess]:
    """Start the two ranks together and wait for both."""
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
                    *DISTRIBUTED,
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


@needs_the_v1_engine
def test_two_ranks_train_and_only_rank_zero_writes(tmp_path):
    ranks = both_ranks(tiny_task(tmp_path))
    for rank, finished in enumerate(ranks):
        assert finished.returncode == 0, f"rank {rank}:\n{finished.stderr}"
    assert "Rank 0 wrote" in ranks[0].stderr
    assert "Rank 1 wrote" not in ranks[1].stderr
    assert evaluated_epochs(ranks[0].stderr) == [0, 1, 2, 3]
    for finished in ranks:
        assert "Epoch 2 starts the stage 'stage_two'" in finished.stderr
    assert sorted(path.name for path in tmp_path.glob("tiny*")) == [
        "tiny.json",
        "tiny.run-000004.json",
        "tiny.run-000004.safetensors",
        "tiny.safetensors",
    ]


@needs_the_v1_engine
def test_a_distributed_run_resumes_from_its_checkpoints(tmp_path):
    config = tiny_task(tmp_path)
    first = both_ranks(config, "--training.max_num_epochs", "3")
    for rank, finished in enumerate(first):
        assert finished.returncode == 0, f"rank {rank}:\n{finished.stderr}"

    resumed = both_ranks(config, "--runtime.restart_latest", "true")
    for rank, finished in enumerate(resumed):
        assert finished.returncode == 0, f"rank {rank}:\n{finished.stderr}"
        assert "Resuming from" in finished.stderr
    assert evaluated_epochs(resumed[0].stderr) == [3]
