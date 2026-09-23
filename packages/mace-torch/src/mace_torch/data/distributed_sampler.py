"""Splitting each epoch's structures across the processes of a distributed run.

Training and evaluation split differently, for two different reasons.

**Training pads.** Every process has to take the same number of optimizer
steps, since each step averages gradients across all of them and a process
that ran out would leave the others waiting. So the epoch's order is padded, by
repeating its first structures, to a multiple of the world size, and each
process takes every ``world_size``-th structure from its own offset. That is
what ``torch.utils.data.DistributedSampler`` does, and the order it is applied
to is the one the epoch already decided, so every process splits the same
permutation.

**Evaluation does not.** Its numbers are sums reduced across the processes, so
an uneven split costs nothing, while a padded one would count the repeated
structures twice and report a different error than one process would.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from torch.utils.data import Dataset, Sampler

__all__ = ["EvaluationSampler", "training_shard"]


def training_shard(rank: int, world_size: int):
    """A function giving one process its share of an epoch's order.

    Args:
        rank: The process's index.
        world_size: How many processes share the epoch.

    Returns:
        A function from the epoch's order to this process's part of it, all
        parts of the same length.
    """
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is not in a world of {world_size}.")

    def shard(indices: Sequence[int]) -> list[int]:
        order = list(indices)
        if not order:
            return order
        total = -(-len(order) // world_size) * world_size
        padded = (order * (total // len(order) + 1))[:total]
        return padded[rank:total:world_size]

    return shard


class EvaluationSampler(Sampler[int]):
    """Every ``world_size``-th structure from this process's offset, unpadded.

    Args:
        dataset: What is evaluated.
        rank: The process's index.
        world_size: How many processes share it.
    """

    def __init__(self, dataset: Dataset, rank: int, world_size: int) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not in a world of {world_size}.")
        self._indices = list(range(rank, len(dataset), world_size))  # ty: ignore[invalid-argument-type]

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)
