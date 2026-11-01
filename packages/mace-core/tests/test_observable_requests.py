"""Turning configured names into what a model produces.

The names in a configuration are of two kinds and read as one list. The point
of resolving them here is that both frameworks then agree about which
derivatives were asked for, and a misspelling is refused rather than becoming
an output the model silently does not have.
"""

from __future__ import annotations

import pytest
from mace_core.observables import (
    DEFAULT_CATALOGUE,
    UnknownObservableError,
    resolve_requested,
)

CATALOGUE = DEFAULT_CATALOGUE


def test_an_observable_alone_asks_for_no_derivative():
    """The inference case: an energy model that computes only an energy."""
    requested = resolve_requested(["energy"], CATALOGUE)
    assert [spec.name for spec in requested.observables] == ["energy"]
    assert requested.derivatives == ()


def test_a_derivative_brings_its_observable_in_with_it():
    """Asking for forces is asking for the energy they are the gradient of."""
    requested = resolve_requested(["forces"], CATALOGUE)
    assert [spec.name for spec in requested.observables] == ["energy"]
    assert requested.derivatives == ("forces",)


def test_naming_both_means_the_same_as_naming_the_derivative():
    assert resolve_requested(["energy", "forces"], CATALOGUE) == resolve_requested(
        ["forces"], CATALOGUE
    )


def test_a_declared_derivative_nobody_asked_for_stays_out():
    """The energy declares a stress whether or not a dataset carries one."""
    requested = resolve_requested(["energy", "forces"], CATALOGUE)
    assert "stress" not in requested.derivatives


def test_the_order_of_the_request_is_kept():
    requested = resolve_requested(["energy", "stress", "forces"], CATALOGUE)
    assert requested.derivatives == ("stress", "forces")


def test_an_unknown_name_says_what_is_on_offer():
    with pytest.raises(UnknownObservableError, match="energy"):
        resolve_requested(["enrgy"], CATALOGUE)


def test_asking_twice_is_refused():
    """Two heads under one name is not a model that can exist."""
    with pytest.raises(ValueError, match="twice"):
        resolve_requested(["energy", "energy"], CATALOGUE)


def test_every_requested_name_is_reported_back():
    requested = resolve_requested(["energy", "forces"], CATALOGUE)
    assert requested.names == ("energy", "forces")
