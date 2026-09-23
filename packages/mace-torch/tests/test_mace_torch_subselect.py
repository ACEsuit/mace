"""Keeping part of a dataset: the filter, the draw, and the padding.

The filter cases are the frozen tree's own, from
`tests/workflows/test_finetuning_select.py`, on the same six structures, and
expressed by atomic number rather than symbol. Its other two cases check that a
random selection keeps the number asked for and that a filter reads the
fine-tuning set's elements; both are here too. What that file never checks is
that the same seed gives the same structures, or anything about farthest-point
sampling, so those are added.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule
from mace_core.data.configuration import Configuration
from mace_torch.finetune import SelectionError, passes_filter, select, split_by_filter
from mace_torch.finetune.subselect import farthest_point_indices

HYDROGEN, CARBON, NITROGEN, OXYGEN, IRON = 1, 6, 7, 8, 26


def structure(numbers, tag: int) -> Configuration:
    """A structure whose energy is its position, so a selection is readable."""
    return Configuration(
        atomic_numbers=np.array(numbers),
        positions=np.zeros((len(numbers), 3)),
        properties={"energy": float(tag)},
    )


#: The frozen tree's six: H2OXYGEN, CH4, Fe2O3, CARBON, FeON, Fe.
SIX = [
    structure(molecule("H2O").get_atomic_numbers(), 0),
    structure(molecule("CH4").get_atomic_numbers(), 1),
    structure([IRON, IRON, OXYGEN, OXYGEN, OXYGEN], 2),
    structure([CARBON], 3),
    structure([IRON, OXYGEN, NITROGEN], 4),
    structure([IRON], 5),
]


def tags(configurations):
    return [int(item.properties["energy"]) for item in configurations]


# ---------------------------------------------------------------------------
# The filter, on the frozen tree's cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filtering", "expected", "elements"),
    [
        ("none", [True] * 6, []),
        ("none", [True] * 6, [CARBON, 92]),
        ("combinations", [False, False, True, False, False, True], [OXYGEN, IRON]),
        ("inclusive", [False, False, True, False, True, False], [OXYGEN, IRON]),
        ("exclusive", [False, False, True, False, False, False], [OXYGEN, IRON]),
    ],
)
def test_the_filter_allows_what_the_frozen_tree_allows(filtering, expected, elements):
    allowed = [passes_filter(s.atomic_numbers, elements, filtering) for s in SIX]
    assert allowed == expected
    passed, rest = split_by_filter(SIX, elements, filtering)
    assert len(passed) == sum(expected)
    assert len(rest) == len(SIX) - sum(expected)


def test_a_filter_with_no_elements_is_refused():
    with pytest.raises(SelectionError, match="no elements"):
        split_by_filter(SIX, [], "combinations")


def test_an_unknown_filter_is_refused():
    with pytest.raises(SelectionError, match="not a filter"):
        passes_filter([HYDROGEN], [HYDROGEN], "everything")  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# The random draw
# ---------------------------------------------------------------------------


def test_a_random_selection_keeps_the_number_asked_for():
    kept = select(SIX, num_samples=2, method="random", filtering="none")
    assert len(kept) == 2


def test_a_filtered_selection_keeps_only_what_the_filter_allows():
    """The frozen tree's `test_select_samples_ft_provided`, with the elements
    of its one fine-tuning structure, FeO."""
    kept = select(SIX, num_samples=2, method="random", elements=[IRON, OXYGEN])
    assert all(
        passes_filter(s.atomic_numbers, [IRON, OXYGEN], "combinations") for s in kept
    )


def test_the_same_seed_keeps_the_same_structures():
    first = select(SIX, num_samples=3, method="random", filtering="none", seed=7)
    second = select(SIX, num_samples=3, method="random", filtering="none", seed=7)
    assert tags(first) == tags(second)


def test_a_different_seed_can_keep_different_ones():
    picks = {
        tuple(
            tags(select(SIX, num_samples=3, method="random", filtering="none", seed=s))
        )
        for s in range(8)
    }
    assert len(picks) > 1


def test_the_draw_is_the_global_generators_after_seeding_it():
    """What the frozen tree's script draws: `np.random.seed(seed)` and then one
    `np.random.choice` over the indices. A private generator seeded the same
    way gives the same indices, without touching the global state."""
    state = np.random.get_state()
    try:
        np.random.seed(11)
        expected = np.random.choice(list(range(6)), 3, replace=False).tolist()
    finally:
        np.random.set_state(state)
    kept = select(SIX, num_samples=3, method="random", filtering="none", seed=11)
    assert tags(kept) == expected


def test_selecting_leaves_the_global_generator_alone():
    """Read by the draws that follow, since a selection in the middle of a run
    must not change what anything seeded before it draws next."""
    np.random.seed(0)
    before = np.random.random(3)
    np.random.seed(0)
    select(SIX, num_samples=3, method="random", filtering="none", seed=11)
    after = np.random.random(3)
    assert np.array_equal(before, after)


# ---------------------------------------------------------------------------
# What passes the filter, and padding
# ---------------------------------------------------------------------------


def test_asking_for_exactly_what_passes_keeps_it_all_in_order():
    kept = select(SIX, num_samples=2, elements=[OXYGEN, IRON], method="random")
    assert tags(kept) == [2, 5]


def test_no_count_keeps_everything_that_passes():
    assert tags(select(SIX, elements=[OXYGEN, IRON])) == [2, 5]


def test_a_shortfall_is_padded_from_the_rest():
    """Two pass `combinations` over Fe and OXYGEN; four are asked for, so two are
    drawn from the four that did not pass."""
    kept = select(SIX, num_samples=4, elements=[OXYGEN, IRON], method="random", seed=3)
    assert tags(kept)[:2] == [2, 5]
    assert len(kept) == 4
    assert set(tags(kept)[2:]) <= {0, 1, 3, 4}


def test_a_shortfall_without_padding_is_refused():
    """The positive setting, off. Legacy spells it as a flag that disallows,
    and reaches the same refusal through its random helper."""
    with pytest.raises(SelectionError, match="padding turned off"):
        select(
            SIX,
            num_samples=4,
            elements=[OXYGEN, IRON],
            method="random",
            allow_random_padding=False,
        )


def test_asking_for_none_at_all_is_refused():
    with pytest.raises(SelectionError, match="at least one"):
        select(SIX, num_samples=0, filtering="none")


# ---------------------------------------------------------------------------
# Farthest points
# ---------------------------------------------------------------------------


def test_farthest_points_spread_out():
    """On a line, starting from one end, the other end comes second."""
    points = np.array([[0.0], [1.0], [2.0], [10.0]])
    chosen = farthest_point_indices(points, 2, np.random.RandomState(0))
    assert 3 in chosen or 0 in chosen
    assert len(set(chosen)) == 2


def test_farthest_point_sampling_needs_descriptors():
    with pytest.raises(SelectionError, match="needs a descriptor"):
        select(SIX, num_samples=2, method="fps", filtering="none")


def test_the_descriptors_must_match_the_structures():
    with pytest.raises(SelectionError, match="descriptors were given"):
        select(
            SIX,
            num_samples=2,
            method="fps",
            filtering="none",
            descriptors=np.zeros((5, 3)),
        )


def test_farthest_point_selection_is_reproducible():
    descriptors = np.random.RandomState(0).normal(size=(6, 4))
    first = select(
        SIX, num_samples=3, method="fps", filtering="none", descriptors=descriptors
    )
    second = select(
        SIX, num_samples=3, method="fps", filtering="none", descriptors=descriptors
    )
    assert tags(first) == tags(second)
    assert len(set(tags(first))) == 3


def test_farthest_points_prefer_a_structure_with_other_elements():
    """A structure lacking an element sits `10e10` away on that element's
    columns, so the sampler reaches for element combinations first."""
    from mace_torch.finetune.subselect import ABSENT

    same = np.zeros((4, 2))
    other = np.full((1, 2), ABSENT)
    points = np.vstack([same, other])
    chosen = farthest_point_indices(points, 2, np.random.RandomState(1))
    assert 4 in chosen
