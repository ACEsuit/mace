"""Resolved isolated-atom energies: what they accept and the table they give."""

import pytest
from mace_core.elements import ResolvedE0s
from pydantic import ValidationError

VALUES = {"pbe": {1: -13.6, 8: -2041.3}, "mp2": {1: -13.2, 8: -2040.1, 6: -1030.4}}


def test_the_table_has_the_given_head_and_element_order():
    resolved = ResolvedE0s(values=VALUES, provenance={"pbe": "average", "mp2": "table"})
    assert resolved.to_array(["mp2", "pbe"], [8, 1]) == [
        [-2040.1, -13.2],
        [-2041.3, -13.6],
    ]


@pytest.mark.parametrize(
    ("heads", "atomic_numbers", "message"),
    [
        (["pbe", "ccsd"], [1], "head 'ccsd' has no isolated-atom energies"),
        (["pbe"], [1, 6, 8], r"head 'pbe' has no isolated-atom energy .* \[6\]"),
    ],
)
def test_a_missing_head_or_element_is_an_error_not_a_zero(
    heads, atomic_numbers, message
):
    resolved = ResolvedE0s(values=VALUES, provenance={"pbe": "average", "mp2": "table"})
    with pytest.raises(KeyError, match=message):
        resolved.to_array(heads, atomic_numbers)


@pytest.mark.parametrize(
    ("values", "provenance", "message"),
    [
        (VALUES, {"pbe": "average"}, "every head has exactly one"),
        (VALUES, {"pbe": "average", "mp2": "guess"}, "mp2"),
        ({}, {}, "at least 1 item"),
        ({" ": {1: -13.6}}, {" ": "table"}, "pattern"),
        ({"pbe": {0: -13.6}}, {"pbe": "table"}, "greater than 0"),
        ({"pbe": {1: float("nan")}}, {"pbe": "table"}, "finite"),
    ],
)
def test_what_cannot_be_a_table_is_refused(values, provenance, message):
    with pytest.raises(ValidationError, match=message):
        ResolvedE0s(values=values, provenance=provenance)
