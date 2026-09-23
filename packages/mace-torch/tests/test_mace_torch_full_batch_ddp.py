"""The full-batch regime on two processes, over Gloo.

L-BFGS evaluates its closure a number of times that depends on the line
search, so the ranks agree only if every evaluation happens at the same point
on all of them. What is checked is exactly that: each rank logs the loss and a
fingerprint of the exact parameter values at every evaluation, and the two
logs are the same. Then the run's final weights are compared with one process
training on the same data, which is what the two shares are meant to add up
to.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
import socket
import sys
import traceback

import pytest
import torch
from conftest import fp64_only
from mace_torch.train import run_data_stage, run_model_stage, run_train_stage
from mace_torch.train.ddp import init_distributed
from test_mace_torch_full_batch import CATALOGUE, configuration

#: Seconds both ranks may take. The run takes a few.
TIMEOUT = 180

EVALUATION = re.compile(r"closure evaluation (\d+): loss (\S+), parameters (\w+)")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _Evaluations(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.seen: list[tuple[str, ...]] = []

    def emit(self, record: logging.LogRecord) -> None:
        match = EVALUATION.search(record.getMessage())
        if match:
            self.seen.append(match.groups())


def _rank(rank: int, port: int, config, results) -> None:
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
        handler = _Evaluations()
        logger = logging.getLogger("mace_torch.train.full_batch")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        processes = init_distributed(True, "torchrun", "cpu")
        data = run_data_stage(config, CATALOGUE)
        built = run_model_stage(config, data, CATALOGUE)
        trained = run_train_stage(config, built, distributed=processes)
        results.put(
            (
                rank,
                handler.seen,
                [p.detach().clone() for p in trained.model.parameters()],
                None,
            )
        )
        torch.distributed.destroy_process_group()
    except BaseException:  # reported to the parent, which fails the test
        results.put((rank, [], [], traceback.format_exc()))


@fp64_only
def test_two_ranks_evaluate_every_closure_at_the_same_point(tmp_path):
    config = configuration(tmp_path, max_num_epochs=2, optimizer={"kind": "lbfgs"})
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    port = _free_port()
    ranks = [
        context.Process(target=_rank, args=(rank, port, config, results))
        for rank in (0, 1)
    ]
    for process in ranks:
        process.start()
    outcomes = {}
    try:
        for _ in ranks:
            rank, seen, parameters, failure = results.get(timeout=TIMEOUT)
            outcomes[rank] = (seen, parameters, failure)
    finally:
        # A rank that stopped leaves the other waiting at its next collective,
        # so both go together rather than one outliving the test.
        for process in ranks:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
    for rank, (_, _, failure) in outcomes.items():
        assert failure is None, f"rank {rank}:\n{failure}"

    first, second = outcomes[0][0], outcomes[1][0]
    assert len(first) > 2, "meant to see the line search evaluate several times"
    assert first == second

    # Not bit for bit: two shares add the set up in another order than one
    # process does, which moves the last bit of each evaluation, and the line
    # search carries that through its steps. Measured at 2.8e-12 on weights of
    # order one after two epochs; a missed broadcast or a doubled structure
    # moves them by many orders more.
    data = run_data_stage(config, CATALOGUE)
    alone = run_train_stage(config, run_model_stage(config, data, CATALOGUE))
    for together, single in zip(outcomes[0][1], alone.model.parameters(), strict=True):
        torch.testing.assert_close(together, single.detach(), rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
