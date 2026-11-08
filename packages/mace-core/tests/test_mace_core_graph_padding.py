"""The padding policy: a discriminated config union, and the budgets it gives."""

from __future__ import annotations

import pytest
from mace_core.graph import (
    AutoPadding,
    FixedPadding,
    NoPadding,
    PaddingBudget,
    PaddingOverflowError,
    PaddingPolicy,
    padding_budget,
)
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
    with pytest.raises(PaddingOverflowError, match="needs 6, 8 and 3"):
        padding_budget(policy, n_nodes=5, n_edges=7, n_graphs=2)


def test_unpadding_diagnostics_can_echo_a_differentiable_input():
    import numpy as np
    from mace_core.graph import PaddingInfo, unpad_outputs
    from mace_core.observables import InputSpec, ObservableCatalogue, ObservableSpec
    from mace_core.outputs import MACEOutput

    catalogue = ObservableCatalogue(
        inputs=(
            InputSpec(
                name="external_field",
                irreps="1o",
                per_atom=False,
                units="eV/(e*Å)",
                differentiable=True,
            ),
        ),
        observables=(
            ObservableSpec(
                name="energy",
                irreps="0e",
                per_atom=False,
                units="eV",
                derivatives=("external_field",),
            ),
        ),
    )
    diagnostics = (
        ObservableSpec(
            name="external_field", irreps="1o", per_atom=False, units="eV/(e*Å)"
        ),
    )
    values = np.arange(9).reshape(3, 3)
    result = unpad_outputs(
        MACEOutput(
            extras={"external_field": values, "d_energy_d_external_field": values + 1}
        ),
        PaddingInfo(n_real_nodes=2, n_real_edges=2, n_real_graphs=1),
        catalogue=catalogue,
        extra_observables=diagnostics,
    )
    np.testing.assert_array_equal(result.extras["external_field"], values[:1])
    np.testing.assert_array_equal(
        result.extras["d_energy_d_external_field"], values[:1] + 1
    )


@pytest.mark.parametrize(
    "policy",
    [AutoPadding(), FixedPadding(nodes_budget=41, edges_budget=128, graphs_budget=2)],
)
def test_retained_budget_grows_once_and_never_shrinks(policy):
    previous = None
    budgets = []
    for nodes, edges in ((40, 100), (40, 90), (80, 500), (80, 450), (40, 90)):
        previous = padding_budget(
            policy, n_nodes=nodes, n_edges=edges, n_graphs=1, previous=previous
        )
        budgets.append(previous)
    assert budgets[0] == budgets[1]
    assert budgets[1] != budgets[2]
    assert budgets[2] == budgets[3] == budgets[4]
    if isinstance(policy, AutoPadding):
        assert budgets[2].edges == 640


def test_retained_auto_budget_can_refuse_growth_even_from_zero_edges():
    policy = AutoPadding(on_overflow="error")
    first = padding_budget(policy, n_nodes=1, n_edges=0, n_graphs=1)
    with pytest.raises(PaddingOverflowError):
        padding_budget(policy, n_nodes=2, n_edges=2, n_graphs=1, previous=first)


@pytest.mark.parametrize(
    "raw",
    [
        {"kind": "typo"},
        {"kind": "auto", "edge_multiple": 0},
        {"kind": "auto", "headroom": 0.5},
    ],
)
def test_invalid_padding_policies_fail_at_construction(raw):
    with pytest.raises(ValidationError):
        TypeAdapter(PaddingPolicy).validate_python(raw)


def test_unpadding_engine_edges_uses_the_edge_count():
    import numpy as np
    from mace_core.graph import PaddingInfo, unpad_outputs
    from mace_core.outputs import MACEOutput

    values = np.arange(30).reshape(10, 3)
    output = unpad_outputs(
        MACEOutput(extras={"edge_forces": values}),
        PaddingInfo(2, 5, 1),
        extra_rows={"edge_forces": "edge"},
    )
    np.testing.assert_array_equal(output.extras["edge_forces"], values[:5])
    with pytest.raises(KeyError):
        unpad_outputs(MACEOutput(extras={"undeclared": values}), PaddingInfo(2, 5, 1))
