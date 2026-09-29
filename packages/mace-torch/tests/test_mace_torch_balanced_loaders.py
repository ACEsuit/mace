"""How several heads' structures make one epoch.

The fixture is two heads of deliberately awkward sizes, 3 and 7, so that
up-sampling needs more than one cycle of the small head and the last cycle is
cut rather than landing evenly. A pair of sizes that divided would let a bug in
the cut pass.

Nothing here builds a model. What is measured is which structures an epoch
visits and in which order.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.observables import load_default_catalogue, resolve_requested
from mace_torch.data import GraphDataset, target_specs
from mace_torch.train import (
    BalancedLoader,
    ProportionalLoader,
    build_training_loader,
)
from mace_torch.train.loaders import pool_seed

CATALOGUE = load_default_catalogue()
REQUESTED = resolve_requested(["energy", "forces"], CATALOGUE)
SPECS = target_specs(REQUESTED)
Z_TABLE = AtomicNumberTable([1])

SMALL, LARGE = 3, 7


def dataset(size: int, head: str) -> GraphDataset:
    """``size`` one-atom structures, each carrying its own energy.

    The energy is the structure's index, so a batch's energies identify exactly
    which structures it drew and in which order. That is what makes the
    up-sampling countable without reaching into the sampler.
    """
    configurations = [
        Configuration(
            atomic_numbers=np.array([1]),
            positions=np.zeros((1, 3)),
            properties={"energy": float(index), "forces": np.zeros((1, 3))},
            head=head,
        )
        for index in range(size)
    ]
    return GraphDataset(
        configurations,
        cutoff=0.5,
        z_table=Z_TABLE,
        targets=SPECS,
        heads=("small", "large"),
    )


@pytest.fixture
def datasets() -> dict:
    return {"small": dataset(SMALL, "small"), "large": dataset(LARGE, "large")}


def loader(datasets, mode: str, *, batch_size: int = 1, seed: int = 0):
    return build_training_loader(
        datasets, mode=mode, z_table=Z_TABLE, batch_size=batch_size, seed=seed
    )


def visited(loader, epoch: int, *, drop_last: bool = False) -> list[tuple[int, float]]:
    """``(head index, energy)`` for every structure the epoch visited."""
    seen = []
    for batch in loader.batches(epoch, drop_last=drop_last):
        heads = batch.graph["head"].reshape(-1).tolist()
        energies = batch.targets["energy"].reshape(-1).tolist()
        seen.extend(zip(heads, energies, strict=True))
    return seen


def per_head(seen) -> dict[int, list[float]]:
    counts: dict[int, list[float]] = {}
    for head, energy in seen:
        counts.setdefault(head, []).append(energy)
    return counts


# ---------------------------------------------------------------------------
# Balanced: every head contributes the same number of steps
# ---------------------------------------------------------------------------


def test_the_small_head_is_up_sampled_to_the_large_one(datasets):
    counts = per_head(visited(loader(datasets, "balanced"), epoch=0))
    assert len(counts[0]) == LARGE
    assert len(counts[1]) == LARGE


def test_the_up_sampled_head_still_visits_every_structure(datasets):
    """Seven draws from three structures, so each appears at least twice."""
    counts = per_head(visited(loader(datasets, "balanced"), epoch=0))
    assert sorted(set(counts[0])) == [0.0, 1.0, 2.0]


def test_the_large_head_visits_each_structure_exactly_once(datasets):
    counts = per_head(visited(loader(datasets, "balanced"), epoch=0))
    assert sorted(counts[1]) == [float(index) for index in range(LARGE)]


def test_the_batches_of_the_two_heads_are_interleaved(datasets):
    """One head's whole epoch and then the other's is a curriculum nobody asked
    for, and the per-head counts would be identical while the run was not."""
    heads = [head for head, _ in visited(loader(datasets, "balanced"), epoch=0)]
    block = [0] * LARGE + [1] * LARGE
    assert heads not in (block, block[::-1])


def test_the_length_matches_what_an_epoch_yields(datasets):
    built = loader(datasets, "balanced")
    counted = sum(1 for _ in built.batches(0, drop_last=False))
    assert built.length(drop_last=False) == counted


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_sequence(datasets):
    first = visited(loader(datasets, "balanced", seed=7), epoch=0)
    second = visited(loader(datasets, "balanced", seed=7), epoch=0)
    assert first == second


def test_a_different_seed_gives_a_different_sequence(datasets):
    first = visited(loader(datasets, "balanced", seed=7), epoch=0)
    second = visited(loader(datasets, "balanced", seed=8), epoch=0)
    assert first != second


def test_a_later_epoch_is_reshuffled(datasets):
    built = loader(datasets, "balanced", seed=7)
    assert visited(built, epoch=0) != visited(built, epoch=1)


def test_the_second_cycle_of_the_small_head_is_not_a_repeat(datasets):
    """Up-sampling by repeating one order would train on three structures in a
    fixed sequence, which is a different thing from seeing them twice."""
    built = loader(datasets, "balanced", seed=7)
    drawn = built.indices("small", epoch=0)
    assert drawn[:SMALL] != drawn[SMALL : 2 * SMALL]
    assert sorted(drawn[:SMALL]) == sorted(drawn[SMALL : 2 * SMALL])


def test_the_last_cycle_is_cut_rather_than_completed(datasets):
    """Seven from three is two whole cycles and one structure."""
    drawn = loader(datasets, "balanced", seed=7).indices("small", epoch=0)
    assert len(drawn) == LARGE


def test_two_shuffles_of_one_epoch_do_not_share_a_key():
    """A key built by adding the parts gives one shuffle to three situations."""
    keys = {
        pool_seed(0, "small", 1, 0),
        pool_seed(0, "small", 0, 1),
        pool_seed(1, "small", 0, 0),
    }
    assert len(keys) == 3


def test_nothing_here_imports_lightning():
    """The joint iteration is Lightning's design, and the dependency is not."""
    from mace_torch.train import loaders

    source = pathlib.Path(loaders.__file__).read_text(encoding="utf-8")
    assert "lightning" not in source.lower()
    assert "lightning" not in sys.modules


# ---------------------------------------------------------------------------
# Proportional: the frozen tree's visitation
# ---------------------------------------------------------------------------


def test_the_proportional_mode_visits_each_head_in_proportion_to_its_size(datasets):
    counts = per_head(visited(loader(datasets, "proportional"), epoch=0))
    assert len(counts[0]) == SMALL
    assert len(counts[1]) == LARGE


def test_the_proportional_mode_mixes_the_heads_within_one_batch(datasets):
    """One pool and one shuffle, which is what the concatenation gives."""
    built = loader(datasets, "proportional", batch_size=5)
    mixed = any(
        len(set(batch.graph["head"].reshape(-1).tolist())) > 1
        for batch in built.batches(0, drop_last=False)
    )
    assert mixed


def test_one_head_makes_the_two_modes_the_same_sequence(datasets):
    """Not merely equivalent. Both name the pool by its heads, and with one
    head the two names are the same string, so the key is the same."""
    single = {"large": datasets["large"]}
    balanced = visited(loader(single, "balanced", seed=3), epoch=0)
    proportional = visited(loader(single, "proportional", seed=3), epoch=0)
    assert balanced == proportional


# ---------------------------------------------------------------------------
# drop_last, which belongs to the stage
# ---------------------------------------------------------------------------


def test_a_full_batch_stage_sees_every_structure(datasets):
    """What a run stepping once an epoch needs, and what the ragged tail costs
    a mini-batch run."""
    built = loader(datasets, "balanced", batch_size=2)
    assert len(visited(built, epoch=0, drop_last=False)) == 2 * LARGE


def test_a_mini_batch_stage_drops_the_ragged_tail(datasets):
    built = loader(datasets, "balanced", batch_size=2)
    kept = len(visited(built, epoch=0, drop_last=True))
    assert kept == 2 * (LARGE // 2) * 2


def test_the_same_loader_serves_both_stages(datasets):
    """The setting is an argument rather than a field, so a run with a
    full-batch second stage does not rebuild its data."""
    built = loader(datasets, "balanced", batch_size=2)
    assert built.length(drop_last=True) != built.length(drop_last=False)


# ---------------------------------------------------------------------------
# The sampler seam a distributed run wraps
# ---------------------------------------------------------------------------


def test_a_sharder_narrows_each_head_without_changing_the_balancing(datasets):
    """What a two-rank run does: each rank takes every second index, and the
    heads stay balanced against each other."""
    built = BalancedLoader(
        datasets,
        z_table=Z_TABLE,
        batch_size=1,
        seed=5,
        shard=lambda indices: list(indices)[0::2],
    )
    counts = per_head(visited(built, epoch=0))
    assert len(counts[0]) == len(counts[1]) == (LARGE + 1) // 2


def test_the_two_shards_together_are_the_unsharded_epoch(datasets):
    whole = BalancedLoader(datasets, z_table=Z_TABLE, batch_size=1, seed=5)
    shards = [
        BalancedLoader(
            datasets,
            z_table=Z_TABLE,
            batch_size=1,
            seed=5,
            shard=lambda indices, rank=rank: list(indices)[rank::2],
        )
        for rank in range(2)
    ]
    assert sorted(whole.indices("large", 0)) == sorted(
        shards[0].indices("large", 0) + shards[1].indices("large", 0)
    )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_unknown_mode_is_refused(datasets):
    with pytest.raises(ValueError, match="balancing mode"):
        loader(datasets, "roundrobin")


def test_a_loader_over_no_heads_is_refused():
    with pytest.raises(ValueError, match="no heads"):
        ProportionalLoader({}, z_table=Z_TABLE, batch_size=1)


def test_a_head_with_no_structures_is_refused(datasets):
    empty = {**datasets, "empty": dataset(0, "empty")}
    with pytest.raises(ValueError, match="no training structures"):
        BalancedLoader(empty, z_table=Z_TABLE, batch_size=1)
