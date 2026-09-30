"""The record of the published foundation models."""

from __future__ import annotations

import pytest
from mace_core.foundation_roster import ROSTER, RosterEntry, roster_entry


def test_each_published_model_is_listed_once():
    keys = [(entry.loader, entry.name) for entry in ROSTER]
    assert len(keys) == len(set(keys))
    assert len({entry.url for entry in ROSTER}) == len(ROSTER)


def test_the_roster_is_the_twenty_four_converted_and_one_dropped():
    converted = [entry for entry in ROSTER if entry.family is not None]
    dropped = [entry for entry in ROSTER if entry.family is None]
    assert len(converted) == 24
    assert [(entry.loader, entry.replaced_by) for entry in dropped] == [
        ("mace_anicc", "MACE-OFF23")
    ]


def test_a_dropped_model_says_why_and_what_instead():
    entry = roster_entry("mace_anicc", "default")
    assert entry.family is None
    assert entry.reason


@pytest.mark.parametrize(
    ("family", "replaced_by", "reason"),
    [(None, None, None), ("scale_shift", "MACE-OFF23", "superseded")],
    ids=["dropped-without-a-reason", "converted-with-a-replacement"],
)
def test_an_entry_is_either_converted_or_dropped_with_its_reason(
    family, replaced_by, reason
):
    with pytest.raises(ValueError, match="dropped artifact names its replacement"):
        RosterEntry(
            "mace_mp", "x", "https://example.org/x", family, replaced_by, reason
        )


def test_an_unknown_model_names_the_ones_there_are():
    with pytest.raises(KeyError, match="mace_mp small,"):
        roster_entry("mace_mp", "tiny")
