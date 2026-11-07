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
