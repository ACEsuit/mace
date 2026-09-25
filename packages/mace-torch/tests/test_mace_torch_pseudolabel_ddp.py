"""Replay labels in a run of two processes: made once, read by both.

The first process generates the labels and writes them; the second never
generates, and reads the first's bytes. Each process is given its own work
directory, as a run whose directory is on each node's local disk would have,
so the labels reach the second by being sent rather than by a shared
filesystem.
"""

from __future__ import annotations

import multiprocessing
import os
import socket
import sys
import traceback

import torch
from conftest import fp64_only
from mace_torch.finetune.stages import build
from mace_torch.train.ddp import init_distributed
from test_mace_torch_extend_elements import water_foundation
from test_mace_torch_pseudolabels import fine_tune

TIMEOUT = 180
FILES = ("labels.safetensors", "manifest.json", "provenance.json")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _rank(rank, port, foundation, directory, results) -> None:
    try:
        os.environ.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": "2",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
            }
        )
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        import mace_torch.finetune.pseudolabels as pseudolabels

        calls = []
        original = pseudolabels.generate_pseudolabels

        def counted(*args, **kwargs):
            calls.append(rank)
            return original(*args, **kwargs)

        vars(pseudolabels)["generate_pseudolabels"] = counted
        processes = init_distributed(True, "torchrun", "cpu")
        config = fine_tune(directory / f"rank{rank}", foundation)
        build(config, context=processes)
        artifact = directory / f"rank{rank}" / "pseudolabels" / "replay"
        results.put(
            (
                rank,
                calls,
                {name: (artifact / name).read_bytes() for name in FILES},
                None,
            )
        )
        torch.distributed.destroy_process_group()
    except BaseException:  # reported to the parent, which fails the test
        results.put((rank, [], {}, traceback.format_exc()))


@fp64_only
def test_the_first_process_labels_and_both_read_the_same_bytes(tmp_path):
    foundation = water_foundation(tmp_path)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    port = _free_port()
    ranks = [
        context.Process(target=_rank, args=(rank, port, foundation, tmp_path, results))
        for rank in (0, 1)
    ]
    for process in ranks:
        process.start()
    outcomes = {}
    try:
        for _ in ranks:
            rank, calls, files, failure = results.get(timeout=TIMEOUT)
            outcomes[rank] = (calls, files, failure)
    finally:
        for process in ranks:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
    for rank, (_, _, failure) in outcomes.items():
        assert failure is None, f"rank {rank}:\n{failure}"
    assert outcomes[0][0] == [0]
    assert outcomes[1][0] == []
    assert outcomes[0][1] == outcomes[1][1]
