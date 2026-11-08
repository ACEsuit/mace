"""Padding a structure to a budget, without touching the model.

What is pinned is the policy: which regime a calculator's arguments select,
how a budget is estimated and grown, and that the padded batch has the same
shape from one structure to the next until the budget grows. The numerical
half, that the real structure's results do not move, is in the calculator's
tests, where there is a model to evaluate.
"""

from __future__ import annotations

import pytest
from mace_torch.calculators.padding import requested_padding

# ---------------------------------------------------------------------------
# Which regime
# ---------------------------------------------------------------------------


def test_nothing_is_padded_unless_asked():
    """Legacy pads nothing by default, and neither does this."""
    assert requested_padding(environ={}).kind == "none"


def test_a_budget_selects_the_fixed_regime():
    policy = requested_padding(10, 400, environ={})
    assert (policy.kind, policy.nodes_budget, policy.edges_budget) == ("fixed", 11, 400)


def test_the_environment_supplies_a_budget_the_arguments_leave_out():
    policy = requested_padding(
        environ={"MACE_ASE_PAD_NUM_ATOMS": "12", "MACE_ASE_PAD_NUM_EDGES": "640"}
    )
    assert (policy.kind, policy.nodes_budget, policy.edges_budget) == ("fixed", 13, 640)


def test_an_argument_wins_over_the_environment():
    policy = requested_padding(
        20, environ={"MACE_ASE_PAD_NUM_ATOMS": "12", "MACE_ASE_PAD_NUM_EDGES": "640"}
    )
    assert (policy.nodes_budget, policy.edges_budget) == (21, 640)


def test_a_compiled_model_with_no_budget_pads_automatically():
    assert requested_padding(compiled=True, environ={}).kind == "auto"


@pytest.mark.parametrize(
    "options",
    [{"nodes": -1}, {"edges": -1}, {"environ": {"MACE_ASE_PAD_NUM_ATOMS": "-1"}}],
)
def test_negative_ase_budgets_are_rejected(options):
    with pytest.raises(ValueError, match="nonnegative"):
        requested_padding(**options)
