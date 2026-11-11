"""The padding policy and its budgets, and the padding structure, in numpy."""

from __future__ import annotations

import numpy as np
import pytest
from mace_core.data import Configuration
from mace_core.graph import (
    AutoPadding,
    FixedPadding,
    NoPadding,
    PaddingBudget,
    PaddingInfo,
    PaddingOverflowError,
    PaddingPolicy,
    check_batch_invariants,
    collate,
    default_layout,
    graph_sample,
    padding_budget,
    padding_length,
    padding_sample,
    padding_samples,
    structure_graph,
)
from mace_core.neighbors import get_neighbor_backend
from mace_core.observables import InputSpec
from pydantic import TypeAdapter, ValidationError


def test_the_policy_is_a_union_discriminated_on_kind():
    adapter = TypeAdapter(PaddingPolicy)
    assert isinstance(adapter.validate_python({"kind": "none"}), NoPadding)
    assert isinstance(adapter.validate_python({"kind": "auto"}), AutoPadding)
    fixed = adapter.validate_python(
        {"kind": "fixed", "nodes_budget": 8, "edges_budget": 64, "graphs_budget": 2}
    )
    assert isinstance(fixed, FixedPadding)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "auto", "nodes_budget": 8})


def test_no_padding_has_no_budget():
    assert padding_budget(NoPadding(), n_nodes=5, n_edges=7, n_graphs=2) is None


def test_auto_padding_matches_the_legacy_calculator_rounding():
    budget = padding_budget(AutoPadding(), n_nodes=5, n_edges=100, n_graphs=2)
    assert budget == PaddingBudget(nodes=6, edges=128, graphs=3)
    no_edges = padding_budget(AutoPadding(), n_nodes=1, n_edges=0, n_graphs=1)
    assert no_edges == PaddingBudget(nodes=2, edges=0, graphs=2)


def test_a_fixed_budget_that_fits_is_kept():
    policy = FixedPadding(nodes_budget=10, edges_budget=64, graphs_budget=4)
    assert padding_budget(policy, n_nodes=5, n_edges=7, n_graphs=2) == PaddingBudget(
        10, 64, 4
    )


def test_a_fixed_budget_grows_to_the_smallest_that_fits():
    policy = FixedPadding(nodes_budget=4, edges_budget=8, graphs_budget=2)
    assert padding_budget(policy, n_nodes=5, n_edges=7, n_graphs=2) == PaddingBudget(
        6, 8, 3
    )


def test_a_fixed_budget_can_refuse_to_grow():
    policy = FixedPadding(
        nodes_budget=4, edges_budget=8, graphs_budget=2, on_overflow="error"
    )
    with pytest.raises(
        PaddingOverflowError, match=r"in nodes \(6 needed, budget 4\), structures"
    ):
        padding_budget(policy, n_nodes=5, n_edges=7, n_graphs=2)


LAYOUT = default_layout([1, 8], cutoff=2.5)


def _real():
    water = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
    )
    backend = get_neighbor_backend("all_pairs")
    return [graph_sample(water, structure_graph(water, 2.5, backend), LAYOUT)[0]]


def test_a_padding_structure_is_aperiodic_at_the_origin_with_z_zero():
    sample = padding_sample(4, 6, 2.5, LAYOUT)
    assert not sample["positions"].any()
    assert not sample["node_atomic_numbers"].any()
    assert not sample["pbc"].any()
    assert np.array_equal(sample["cell"], np.eye(3) * padding_length(2.5))
    assert np.all(sample["edge_index"] == 3)
    assert np.all(sample["shifts"][:, 0] == padding_length(2.5))
    assert sample["total_spin"] == 1.0
    assert sample["total_charge"] == 0.0


def test_a_padding_edge_is_longer_than_the_cutoff():
    assert padding_length(2.5) == 5.0
    assert padding_length(0.1) == 1.0


def test_a_padded_batch_is_the_collation_of_real_and_padding_structures():
    real = _real()
    _, info = collate(real, LAYOUT)
    budget = PaddingBudget(nodes=8, edges=20, graphs=3)
    padding = padding_samples(
        budget,
        PaddingInfo(info.num_nodes, info.num_edges, info.num_graphs),
        2.5,
        LAYOUT,
    )
    padded, padded_info = collate([*real, *padding], LAYOUT, validate=True)
    assert padded_info.structure_sizes == (3, 0, 5)
    assert padded["edge_index"].shape == (2, 20)
    assert np.all(padded["edge_index"][:, info.num_edges :] == 7)
    check_batch_invariants(padded, padded_info)


def test_padding_structures_without_edges_leave_the_edges_to_a_device_search():
    samples = padding_samples(
        PaddingBudget(nodes=5, edges=9, graphs=2),
        PaddingInfo(3, 0, 1),
        2.5,
        LAYOUT,
        with_edges=False,
    )
    assert len(samples) == 1
    assert "edge_index" not in samples[0]


def test_edges_without_an_atom_to_loop_on_are_refused():
    with pytest.raises(ValueError, match="self loop"):
        padding_sample(0, 3, 2.5, LAYOUT)


def test_an_embedding_feature_pads_at_the_trailing_sizes_the_batch_gives():
    layout = default_layout([1, 8], cutoff=2.5, features={"spin": "atom"})
    with pytest.raises(ValueError, match="feature_shapes"):
        padding_sample(2, 1, 2.5, layout)
    sample = padding_sample(2, 1, 2.5, layout, feature_shapes={"spin": (4,)})
    assert sample["spin"].shape == (2, 4)
    assert not sample["spin"].any()


def test_a_declared_input_outside_the_defaults_pads_with_its_absent_value():
    site_field = InputSpec(name="site_field", irreps="1o", per_atom=True, units="V/Å")
    layout = default_layout([1, 8], cutoff=2.5, inputs=[site_field])
    water = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        properties={"site_field": np.full((3, 3), 0.5)},
    )
    backend = get_neighbor_backend("all_pairs")
    real = [graph_sample(water, structure_graph(water, 2.5, backend), layout)[0]]
    _, info = collate(real, layout)
    padding = padding_samples(
        PaddingBudget(nodes=6, edges=info.num_edges + 4, graphs=3),
        PaddingInfo(info.num_nodes, info.num_edges, info.num_graphs),
        2.5,
        layout,
    )
    padded, padded_info = collate([*real, *padding], layout, validate=True)
    assert padded["site_field"].shape == (6, 3)
    assert np.all(padded["site_field"][:3] == 0.5)
    assert not padded["site_field"][3:].any()
    assert padded_info.structure_sizes == (3, 0, 3)
