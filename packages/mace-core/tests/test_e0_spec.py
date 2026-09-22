"""Where a head's isolated-atom energies come from, as five kinds of one field.

Legacy carries this as one string meaning five different things, so the extra
settings each kind needs became separate flags. The point of the union is that
a kind and its settings are written together, and that the two fallbacks which
keep a run going on silently wrong energies are gone.
"""

import pytest
from mace_core.config.base import ConfigError, ConfigSection, ReforgeBaseConfig
from mace_core.config.e0s import (
    FOUNDATION_E0_KINDS,
    E0sAverage,
    E0sEstimated,
    E0sFromFoundation,
    E0sIsolatedAtoms,
    E0Spec,
    E0sTable,
)
from pydantic import ValidationError


class Head(ConfigSection):
    e0s: E0Spec = E0sIsolatedAtoms()


class Root(ReforgeBaseConfig):
    head: Head = Head()


#: Every kind, with a file body that satisfies it. The `table` body is not
#: empty because its `values` is required.
KINDS: dict[str, tuple[type[ConfigSection], dict]] = {
    "table": (E0sTable, {"values": {1: -13.6}}),
    "isolated_atoms": (E0sIsolatedAtoms, {}),
    "average": (E0sAverage, {}),
    "foundation": (E0sFromFoundation, {}),
    "estimated": (E0sEstimated, {}),
}


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_kind_parses_through_the_discriminator(kind):
    expected, body = KINDS[kind]
    config = Root.model_validate({"head": {"e0s": {kind: body}}})
    assert isinstance(config.head.e0s, expected)
    assert config.head.e0s.kind == kind


def test_the_default_is_the_one_legacy_reaches_for_first():
    assert Root().head.e0s == E0sIsolatedAtoms()
    assert Root().head.e0s.on_missing_energy == "error"


def test_a_kind_is_written_as_its_own_key():
    """`[e0s.foundation]` rather than a `kind =` line beside the settings."""
    config = Root.load(
        cli_overrides=[
            "--head.e0s",
            '{"foundation": {"head": "mp", "missing": "zero"}}',
        ]
    )
    assert config.head.e0s == E0sFromFoundation(head="mp", missing="zero")
    assert config.to_resolved_dict()["head"]["e0s"] == {
        "foundation": {"head": "mp", "missing": "zero"}
    }


def test_an_unknown_kind_names_the_kinds_there_are():
    with pytest.raises(ConfigError, match=r"kinds of head\.e0s"):
        Root.load(cli_overrides=["--head.e0s", '{"from_thin_air": {}}'])


def test_a_setting_that_belongs_to_another_kind_is_refused():
    """The reason the settings live with the kind rather than beside it."""
    with pytest.raises(ConfigError, match=r"head\.e0s\.average\.on_missing_energy"):
        Root.load(
            cli_overrides=["--head.e0s", '{"average": {"on_missing_energy": "zero"}}']
        )


def test_a_table_has_to_carry_values():
    """An empty table is the all-zero state the other kinds refuse."""
    with pytest.raises(ValidationError):
        E0sTable()


def test_the_isolated_atom_fallback_is_a_choice_and_defaults_to_refusing():
    """Legacy warns and contributes 0.0 (`mace/data/utils.py:333-337`).

    A zero is indistinguishable from a real reference energy of zero, and it
    shifts every structure containing that element, so the default is the
    error and the old behaviour has to be asked for.
    """
    assert E0sIsolatedAtoms().on_missing_energy == "error"
    assert E0sIsolatedAtoms(on_missing_energy="zero").on_missing_energy == "zero"


def test_the_singular_least_squares_fallback_is_not_a_choice_at_all():
    """Legacy logs an error and zeroes every element (`:381-387`).

    Unlike the isolated-atom case there is no legitimate reading of it, so
    there is no field to set. Pinned as a field set rather than as prose: a
    later `fallback` option would be a decision, and this fails when one
    appears.
    """
    assert set(E0sAverage.model_fields) == {"kind"}


@pytest.mark.parametrize("kind", sorted(FOUNDATION_E0_KINDS))
def test_a_foundation_kind_takes_a_head_and_refuses_to_guess(kind):
    """Legacy takes head 0 and only logs it (`mace/cli/run_train.py:510-514`).

    The field is optional here because a single-head foundation model needs no
    answer; what makes it an error to omit on a multi-head one is the
    cross-section validator, which needs the foundation section to know.
    """
    expected, _body = KINDS[kind]
    assert expected().head is None
    assert expected(head="mp").head == "mp"
    assert expected().missing == "error"


def test_the_two_foundation_kinds_are_named_once():
    """The validator and its message read the same set."""
    assert {"foundation", "estimated"} == FOUNDATION_E0_KINDS
    assert {E0sFromFoundation().kind, E0sEstimated().kind} == FOUNDATION_E0_KINDS


def test_resolving_twice_is_a_fixed_point():
    written = {"head": {"e0s": {"table": {"values": {6: -1000.0}}}}}
    once = Root.model_validate(written).to_resolved_dict()
    twice = Root.model_validate(once).to_resolved_dict()
    assert once == twice
    assert once["head"]["e0s"] == {"table": {"values": {"6": -1000.0}}}
