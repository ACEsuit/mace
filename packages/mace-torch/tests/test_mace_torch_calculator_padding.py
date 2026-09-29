"""Padding a structure to a budget, without touching the model.

What is pinned is the policy: which regime a calculator's arguments select,
how a budget is estimated and grown, and that the padded batch has the same
shape from one structure to the next until the budget grows. The numerical
half, that the real structure's results do not move, is in the calculator's
tests, where there is a model to evaluate.
"""

from __future__ import annotations

import numpy as np
import pytest
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_torch.calculators.padding import (
    EDGES_PER_PADDING_ATOM,
    PaddingOverflowError,
    PaddingPolicy,
    pad_batch,
    resolve_budget,
)
from mace_torch.data.graphs import graph_from_configuration

R_MAX = 3.0


def structure(count: int, seed: int = 0) -> dict[str, np.ndarray]:
    """``count`` atoms in a periodic box, dense enough to have many edges."""
    generator = np.random.default_rng(seed)
    side = (count / 0.08) ** (1 / 3)
    configuration = Configuration(
        atomic_numbers=np.full(count, 1),
        positions=generator.uniform(0.0, side, size=(count, 3)),
        cell=np.eye(3) * side,
        pbc=(True, True, True),
    )
    return graph_from_configuration(
        configuration, cutoff=R_MAX, z_table=AtomicNumberTable([1])
    )


def edges_of(graph) -> int:
    return int(graph["edge_index"].shape[1])


# ---------------------------------------------------------------------------
# Which regime
# ---------------------------------------------------------------------------


def test_nothing_is_padded_unless_asked():
    """Legacy pads nothing by default, and neither does this."""
    assert PaddingPolicy.requested(environ={}).mode == "none"


def test_a_budget_selects_the_fixed_regime():
    policy = PaddingPolicy.requested(10, 400, environ={})
    assert (policy.mode, policy.nodes_budget, policy.edges_budget) == ("fixed", 10, 400)


def test_the_environment_supplies_a_budget_the_arguments_leave_out():
    policy = PaddingPolicy.requested(
        environ={"MACE_ASE_PAD_NUM_ATOMS": "12", "MACE_ASE_PAD_NUM_EDGES": "640"}
    )
    assert (policy.mode, policy.nodes_budget, policy.edges_budget) == ("fixed", 12, 640)


def test_an_argument_wins_over_the_environment():
    policy = PaddingPolicy.requested(
        20, environ={"MACE_ASE_PAD_NUM_ATOMS": "12", "MACE_ASE_PAD_NUM_EDGES": "640"}
    )
    assert (policy.nodes_budget, policy.edges_budget) == (20, 640)


def test_a_compiled_model_with_no_budget_pads_automatically():
    assert PaddingPolicy.requested(compiled=True, environ={}).mode == "auto"


# ---------------------------------------------------------------------------
# The budget
# ---------------------------------------------------------------------------


def test_the_automatic_budget_is_the_first_structure_with_headroom():
    graph = structure(40)
    policy, changed = resolve_budget(PaddingPolicy(mode="auto"), 40, edges_of(graph))
    assert changed
    assert policy.nodes_budget == 40
    assert policy.edges_budget == -(-int(edges_of(graph) * 1.25) // 64) * 64
    assert policy.edges_budget % 64 == 0


def test_a_structure_within_the_budget_leaves_it_alone():
    first, _ = resolve_budget(PaddingPolicy(mode="auto"), 40, edges_of(structure(40)))
    again, changed = resolve_budget(first, 40, edges_of(structure(40, seed=1)))
    assert not changed and again == first


def test_an_edge_overflow_grows_the_budget_once():
    first, _ = resolve_budget(
        PaddingPolicy(mode="fixed", nodes_budget=40, edges_budget=64), 40, 64
    )
    grown, changed = resolve_budget(first, 40, 1000)
    assert changed and grown.edges_budget >= 1000
    settled, changed = resolve_budget(grown, 40, 1000)
    assert not changed and settled == grown


def test_an_overflow_is_refused_when_the_policy_does_not_grow():
    policy = PaddingPolicy(
        mode="fixed", nodes_budget=40, edges_budget=64, on_overflow="error"
    )
    with pytest.raises(PaddingOverflowError, match="1000 edges"):
        resolve_budget(policy, 40, 1000)


# ---------------------------------------------------------------------------
# The padded batch
# ---------------------------------------------------------------------------


def test_the_fake_structure_s_edges_are_beyond_the_cutoff():
    graph = structure(40)
    policy, _ = resolve_budget(PaddingPolicy(mode="auto"), 40, edges_of(graph))
    (_, fake), info = pad_batch(graph, policy, R_MAX)
    assert (info.nodes, info.edges) == (40, edges_of(graph))
    lengths = np.linalg.norm(fake["shifts"], axis=1)
    assert np.all(lengths >= R_MAX)
    # Self-loops, so the length is the shift alone.
    assert np.array_equal(fake["edge_index"][0], fake["edge_index"][1])
    # A shift is its unit shift times the cell, so a strained cell strains it.
    assert np.allclose(fake["unit_shifts"] @ fake["cell"], fake["shifts"])


def test_the_padding_edges_are_spread_over_the_fake_atoms():
    """Stacked on one atom, they hand the scatter one enormous segment."""
    graph = structure(40)
    policy, _ = resolve_budget(
        PaddingPolicy(
            mode="fixed", nodes_budget=40, edges_budget=edges_of(graph) + 5000
        ),
        40,
        edges_of(graph),
    )
    (_, fake), _ = pad_batch(graph, policy, R_MAX)
    loads = np.bincount(fake["edge_index"][0], minlength=len(fake["atomic_numbers"]))
    assert len(loads) > 1
    assert loads.max() - loads.min() <= 1
    assert loads.max() <= EDGES_PER_PADDING_ATOM


def test_the_batch_keeps_its_shape_from_one_step_to_the_next():
    """The fake atoms are counted once, with the budget, so a structure whose
    edge count moves does not move the node count, which under static shapes
    would be a recompile at every step."""
    policy = PaddingPolicy(mode="auto")
    shapes = set()
    for seed in range(6):
        graph = structure(40, seed=seed)
        policy, _ = resolve_budget(policy, 40, edges_of(graph))
        structures, _ = pad_batch(graph, policy, R_MAX)
        nodes = sum(len(item["atomic_numbers"]) for item in structures)
        edges = sum(edges_of(item) for item in structures)
        shapes.add((nodes, edges))
    assert len({edges_of(structure(40, seed=seed)) for seed in range(6)}) > 1
    assert len(shapes) == 1


def test_the_shape_changes_only_when_the_budget_grows():
    policy = PaddingPolicy(mode="auto")
    shapes = []
    for count in (40, 40, 80, 80, 40):
        graph = structure(count)
        policy, _ = resolve_budget(policy, count, edges_of(graph))
        structures, _ = pad_batch(graph, policy, R_MAX)
        shapes.append(
            (
                sum(len(item["atomic_numbers"]) for item in structures),
                sum(edges_of(item) for item in structures),
            )
        )
    assert shapes[0] == shapes[1]
    assert shapes[2] != shapes[1]
    assert shapes[2] == shapes[3] == shapes[4]
