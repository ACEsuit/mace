"""What a loss value means when a batch is split across ranks.

Under several ranks a loss is not the mean of the local elements. Each rank's
loss contributes one world-size'th of the averaged gradient, so for that
average to be the gradient of the whole batch's mean, a rank has to report its
own sum scaled by the world size over the **global** element count.

The consequence is worth stating because it reads as a bug: no rank reports the
single-process number. Their mean does. A rank holding more than its share
reports less than the whole, and one holding less reports more.

Two real processes over gloo, because the rule is an `all_reduce` and a
simulated world would be this file checking its own arithmetic.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mace_torch.train import reduce_loss

needs_gloo = pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="this build has no gloo backend",
)

#: Deliberately uneven: five elements against three. With an even split the
#: wrong rule and the right one agree, which is how a rule like this survives.
SHARDS = (
    torch.arange(5, dtype=torch.float64),
    torch.arange(5, 8, dtype=torch.float64),
)


def _worker(rank: int, store: str, reported: dict) -> None:
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{store}"
    )
    try:
        reported[rank] = float(reduce_loss(SHARDS[rank]))
    finally:
        dist.destroy_process_group()


def on_two_ranks() -> tuple[float, float]:
    with tempfile.TemporaryDirectory() as directory:
        reported = mp.Manager().dict()
        mp.spawn(
            _worker,
            args=(os.path.join(directory, "store"), reported),
            nprocs=2,
            join=True,
        )
        return reported[0], reported[1]


@needs_gloo
def test_the_ranks_average_to_the_single_process_value():
    """The property the rule exists to have."""
    first, second = on_two_ranks()
    whole = float(torch.cat(SHARDS).mean())
    assert (first + second) / 2 == pytest.approx(whole, abs=1e-12)


@needs_gloo
def test_no_single_rank_reports_the_single_process_value():
    """Which is why it looks wrong in a log, and why this is written down.

    A rank holding five of the eight elements reports 2.5 against a whole-batch
    3.5, and the one holding three reports 4.5.
    """
    first, second = on_two_ranks()
    whole = float(torch.cat(SHARDS).mean())
    assert first != pytest.approx(whole)
    assert second != pytest.approx(whole)
    assert (first, second) == pytest.approx((2.5, 4.5))


@needs_gloo
def test_a_mean_of_local_means_would_be_wrong():
    """The rule a reader would reach for. It is right only when the split is
    even, which is what makes the uneven split above the whole test."""
    naive = sum(float(shard.mean()) for shard in SHARDS) / 2
    whole = float(torch.cat(SHARDS).mean())
    assert naive != pytest.approx(whole)


def test_one_rank_is_the_plain_mean():
    """Outside a process group there is nothing to reduce over."""
    assert float(reduce_loss(torch.cat(SHARDS))) == pytest.approx(3.5)
