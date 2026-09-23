"""Which process a distributed run is, and what share of the data it takes.

Nothing here starts a second process: these are the rules the processes follow,
checked one at a time. The two-process run is in the workflow suite.
"""

from __future__ import annotations

import pytest
import torch
from mace_torch.data.distributed_sampler import EvaluationSampler, training_shard
from mace_torch.train.ddp import (
    DistributedContext,
    broadcast_decision,
    init_distributed,
    process_group_backend,
    wrap_model,
)
from mace_torch.train.slurm import expand_nodelist, slurm_environment


@pytest.mark.parametrize(
    "nodelist,hosts",
    [
        ("node01", ["node01"]),
        ("node[01-03]", ["node01", "node02", "node03"]),
        ("a[1-2],b07", ["a1", "a2", "b07"]),
        (
            "raven[0998-1001,1005]",
            ["raven0998", "raven0999", "raven1000", "raven1001", "raven1005"],
        ),
        ("gpu[3,5]-ib", ["gpu3-ib", "gpu5-ib"]),
    ],
)
def test_a_node_list_is_written_out(nodelist, hosts):
    assert expand_nodelist(nodelist) == hosts


def test_an_unclosed_node_list_is_refused():
    with pytest.raises(ValueError, match="unclosed"):
        expand_nodelist("node[01-03")


def test_slurm_places_a_process_by_its_own_variables():
    place = slurm_environment(
        {
            "SLURM_PROCID": "5",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS": "8",
            "SLURM_JOB_NODELIST": "ravg[1002-1003]",
        }
    )
    assert (place.rank, place.local_rank, place.world_size) == (5, 1, 8)
    assert place.master_addr == "ravg1002"
    assert place.exported()["MASTER_PORT"] == "33333"


def test_slurm_derives_the_world_from_tasks_per_node_when_it_has_to():
    place = slurm_environment(
        {
            "SLURM_PROCID": "0",
            "SLURM_LOCALID": "0",
            "SLURM_NTASKS_PER_NODE": "4",
            "SLURM_NNODES": "3",
            "SLURM_JOB_NODELIST": "n1",
        }
    )
    assert place.world_size == 12


def test_a_process_outside_srun_says_so():
    with pytest.raises(KeyError, match="srun"):
        slurm_environment({"SLURM_PROCID": "0"})


@pytest.mark.parametrize(
    "device,backend",
    [("cpu", "gloo"), ("cuda", "nccl"), ("cuda:1", "nccl"), ("xpu", "xccl")],
)
def test_the_collective_backend_follows_the_device(device, backend):
    assert process_group_backend(device) == backend


def test_a_run_that_is_not_distributed_is_one_process():
    context = init_distributed(False, None, "cpu")
    assert (context.rank, context.world_size, context.is_main) == (0, 1, True)
    assert init_distributed(True, "none", "cpu").world_size == 1


def test_a_distributed_run_with_no_launcher_is_refused():
    with pytest.raises(ValueError, match="names no launcher"):
        init_distributed(True, None, "cpu")


def test_an_unknown_launcher_is_refused():
    with pytest.raises(ValueError, match="not a launcher"):
        init_distributed(True, "pbs", "cpu", environ={})


def test_a_single_process_is_neither_wrapped_nor_broadcast_to():
    model = torch.nn.Linear(2, 2)
    context = DistributedContext()
    assert wrap_model(model, context) is model
    assert broadcast_decision(True, context) is True


@pytest.mark.parametrize("length,world", [(10, 2), (11, 2), (7, 3), (2, 4)])
def test_every_rank_takes_as_many_steps_and_together_they_cover_the_epoch(
    length, world
):
    order = list(range(100, 100 + length))
    shares = [training_shard(rank, world)(order) for rank in range(world)]
    assert len({len(share) for share in shares}) == 1
    assert set().union(*shares) == set(order)
    assert sum(len(share) for share in shares) == -(-length // world) * world


def test_the_shares_follow_the_epoch_s_own_order():
    """Interleaved, as DistributedSampler takes them, from the order the epoch
    already decided rather than from a fresh shuffle."""
    order = [4, 0, 3, 1, 2]
    assert training_shard(0, 2)(order) == [4, 3, 2]
    assert training_shard(1, 2)(order) == [0, 1, 4]


@pytest.mark.parametrize("length,world", [(10, 2), (11, 3), (1, 2)])
def test_evaluation_is_split_without_padding(length, world):
    """Its numbers are summed across the ranks, so each structure has to be
    counted exactly once."""
    data = list(range(length))
    shares = [list(EvaluationSampler(data, rank, world)) for rank in range(world)]  # ty: ignore[invalid-argument-type]
    flattened = sorted(index for share in shares for index in share)
    assert flattened == data


def test_a_rank_outside_the_world_is_refused():
    with pytest.raises(ValueError, match="not in a world"):
        training_shard(2, 2)
