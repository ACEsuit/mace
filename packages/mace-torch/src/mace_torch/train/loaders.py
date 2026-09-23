"""How several heads' structures become one epoch of batches.

The frozen tree flattens every head into one pool and shuffles it
(``mace/cli/run_train.py:754,777-783``), so a head is visited in proportion to
its size and a small one is drowned by a large one. It is not a bug in the
sense of a wrong number: it is a training schedule nobody chose, and the run
reports nothing about it.

So the mode is a decision the configuration makes, and both readings exist:

``balanced``
    an epoch is as long as the **largest** head, and the smaller ones cycle
    until it is exhausted. Every head contributes the same number of batches,
    whatever its size.

``proportional``
    the frozen tree's: one pool, one shuffle, so a batch can mix heads and a
    head is seen in proportion to its size.

**Every shuffle is derived, never drawn.** A permutation comes from a key built
out of the run seed, the pool, the epoch and the cycle, so two runs of the same
configuration step on the same batches in the same order, a resumed run
continues the sequence rather than restarting it, and the second pass over a
small head is a *different* order rather than a repeat of the first. Nothing
here touches the global generator, which is what would make the sequence depend
on whatever else drew a random number first.

**Sharding is a function, not a rank.** A head's epoch indices are handed to
:data:`IndexSharder` before they are batched, so a distributed run narrows each
head the same way and the balancing above it does not change. Ranks are not
mentioned here at all: a loader that knew about them would have to be asked
whether it had been told, and the single-rank case is the identity.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import cached_property

import torch
from mace_core.elements import AtomicNumberTable
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from mace_torch.data import GraphDataset, Item, TrainingBatch, make_collate

__all__ = [
    "BalancedLoader",
    "IndexSharder",
    "ProportionalLoader",
    "TrainingLoader",
    "build_training_loader",
    "pool_seed",
]

#: Which of a head's epoch indices this process takes. The identity is a
#: single-rank run; a distributed one hands in its own slice and the balancing
#: above is untouched.
IndexSharder = Callable[[Sequence[int]], Sequence[int]]

#: Keys are taken modulo this, which is the largest seed a torch generator
#: accepts without wrapping.
_SEED_MODULUS = 2**31


def pool_seed(seed: int, pool: str, epoch: int, cycle: int) -> int:
    """The generator key for one shuffle.

    A hash rather than an arithmetic combination: ``seed + epoch + cycle``
    gives the same key to three different shuffles, and the collision is
    invisible because each of them on its own still looks random.
    """
    digest = hashlib.blake2b(f"{pool}|{epoch}|{cycle}".encode(), digest_size=8).digest()
    return (seed + int.from_bytes(digest, "big")) % _SEED_MODULUS


def _permutation(length: int, key: int) -> list[int]:
    """A permutation of ``range(length)`` from a key, on a private generator."""
    generator = torch.Generator().manual_seed(key)
    return torch.randperm(length, generator=generator).tolist()


def _batch_count(length: int, batch_size: int, drop_last: bool) -> int:
    """How many batches ``length`` samples make."""
    if drop_last:
        return length // batch_size
    return -(-length // batch_size)


class TrainingLoader:
    """What the loop steps on: batches, for an epoch, at a stage's ``drop_last``.

    The two modes share everything except which indices an epoch is made of,
    so that is the one method a subclass writes.

    ``drop_last`` is an argument rather than a field because it belongs to the
    stage rather than to the data: a full-batch stage has to see every
    structure and a mini-batch stage drops the ragged tail, and legacy carries
    the same distinction as ``drop_last=(not args.lbfgs)``. Keeping it out of
    the loader is what lets one loader serve both stages of a run.
    """

    def __init__(
        self,
        datasets: Mapping[str, GraphDataset],
        *,
        z_table: AtomicNumberTable,
        batch_size: int,
        seed: int = 0,
        float_dtype: str = "float64",
        num_workers: int = 0,
        pin_memory: bool = False,
        shard: IndexSharder | None = None,
    ) -> None:
        if not datasets:
            raise ValueError(
                "a training loader was built over no heads, so an epoch has "
                "no batches and the run would report an untrained model."
            )
        empty = sorted(name for name, data in datasets.items() if len(data) == 0)
        if empty:
            raise ValueError(
                f"heads {empty} have no training structures. A head with none "
                f"contributes no batch and its readout trains against nothing."
            )
        self.datasets = dict(datasets)
        self.heads = tuple(datasets)
        self.batch_size = batch_size
        self.seed = seed
        self.shard: IndexSharder = shard if shard is not None else list
        self._collate = make_collate(z_table, float_dtype)
        self._num_workers = num_workers
        self._pin_memory = pin_memory

    def batches(self, epoch: int, *, drop_last: bool) -> Iterator[TrainingBatch]:
        """This epoch's batches, in the order the optimizer sees them."""
        raise NotImplementedError

    def length(self, *, drop_last: bool) -> int:
        """How many batches an epoch has. The same for every epoch."""
        raise NotImplementedError

    def _loader(
        self, dataset: Dataset[Item], indices: Sequence[int], drop_last: bool
    ) -> DataLoader:
        """One pass over the given indices, in the given order.

        A sampler rather than ``shuffle=True``: the order is already decided,
        and letting the loader draw its own would put the one thing this module
        exists to make reproducible back on the global generator.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=list(indices),
            collate_fn=self._collate,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=drop_last,
        )


class BalancedLoader(TrainingLoader):
    """An epoch is as long as the largest head, and the others cycle into it.

    A head shorter than the longest is walked again from a fresh permutation
    until it has contributed as many structures as the longest, and the last
    cycle is cut rather than padded. So every head contributes the same number
    of optimizer steps, which is the property the mode exists for.

    The batches of the heads are interleaved in a shuffled order rather than
    head by head. Taking one head's whole epoch and then the next would give
    the optimizer a curriculum nobody asked for, and the per-head counts would
    be identical while the run was not.
    """

    def indices(self, head: str, epoch: int) -> list[int]:
        """One head's structures for this epoch, up-sampled to the largest."""
        length = len(self.datasets[head])
        target = max(len(data) for data in self.datasets.values())
        drawn: list[int] = []
        cycle = 0
        while len(drawn) < target:
            key = pool_seed(self.seed, head, epoch, cycle)
            drawn.extend(_permutation(length, key)[: target - len(drawn)])
            cycle += 1
        return list(self.shard(drawn))

    def length(self, *, drop_last: bool) -> int:
        return sum(
            _batch_count(len(self.indices(head, 0)), self.batch_size, drop_last)
            for head in self.heads
        )

    def batches(self, epoch: int, *, drop_last: bool) -> Iterator[TrainingBatch]:
        iterators = {}
        plan: list[str] = []
        for head in self.heads:
            indices = self.indices(head, epoch)
            loader = self._loader(self.datasets[head], indices, drop_last)
            iterators[head] = iter(loader)
            plan.extend([head] * _batch_count(len(indices), self.batch_size, drop_last))
        order = _permutation(len(plan), pool_seed(self.seed, "\x00plan", epoch, 0))
        for position in order:
            yield next(iterators[plan[position]])


class ProportionalLoader(TrainingLoader):
    """One pool, one shuffle: the frozen tree's visitation, reproduced.

    A batch can mix heads and a head is seen in proportion to its size. It is
    what the phase gate compares against, so it is a mode rather than a
    historical note.

    With a single head this is the same sequence as :class:`BalancedLoader`,
    not merely an equivalent one: both name the pool by its heads, and with one
    head the two names are the same string, so the two draw the same
    permutation from the same key.
    """

    @cached_property
    def pool(self) -> ConcatDataset[Item]:
        """The heads' datasets end to end, in the order they were given."""
        return ConcatDataset(list(self.datasets.values()))

    def indices(self, epoch: int) -> list[int]:
        """The whole pool, shuffled once, as legacy shuffles the concatenation."""
        key = pool_seed(self.seed, "+".join(self.heads), epoch, 0)
        return list(self.shard(_permutation(len(self.pool), key)))

    def length(self, *, drop_last: bool) -> int:
        return _batch_count(len(self.indices(0)), self.batch_size, drop_last)

    def batches(self, epoch: int, *, drop_last: bool) -> Iterator[TrainingBatch]:
        yield from self._loader(self.pool, self.indices(epoch), drop_last)


def build_training_loader(
    datasets: Mapping[str, GraphDataset],
    *,
    mode: str,
    z_table: AtomicNumberTable,
    batch_size: int,
    seed: int = 0,
    float_dtype: str = "float64",
    num_workers: int = 0,
    pin_memory: bool = False,
    shard: IndexSharder | None = None,
) -> TrainingLoader:
    """The loader a configured mode asks for.

    Raises:
        ValueError: Naming the modes, since a misspelled one would otherwise
            have to fall back to one of them and the run would train on a
            schedule it did not ask for.
    """
    kinds = {"balanced": BalancedLoader, "proportional": ProportionalLoader}
    if mode not in kinds:
        raise ValueError(
            f"{mode!r} is not a head-balancing mode. They are "
            f"{sorted(kinds)}: 'balanced' gives every head the same number of "
            f"batches, 'proportional' visits each in proportion to its size."
        )
    return kinds[mode](
        datasets,
        z_table=z_table,
        batch_size=batch_size,
        seed=seed,
        float_dtype=float_dtype,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shard=shard,
    )
