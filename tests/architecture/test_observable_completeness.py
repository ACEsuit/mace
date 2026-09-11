"""Every legacy model output is accounted for by the declarative spec.

Two failure modes this suite is built against. The first is a key nobody
classified, which is how the ase calculator ended up returning 21 of the 43
model keys with their padding rows still in them. The second, quieter one is a
test that measures nothing: an extraction that finds no keys reports perfect
coverage, so the counts are asserted as numbers rather than left to a set
comparison that an empty set would satisfy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from mace_core.observables import ObservableSpec, derivative_name, derivative_sign

from tests.architecture.observable_coverage import (
    DECLARED_INPUTS,
    DISPOSITIONS,
    LEGACY_MODEL_SOURCES,
    Derivative,
    Drop,
    Spec,
    channel_of,
    legacy_calculator_keys,
    legacy_eval_keys,
    legacy_model_keys,
    per_atom_of,
)
from tests.golden import surface_scan

SURFACE_DOC = (
    Path(__file__).resolve().parents[2] / "docs" / "reforge" / "output_surface.md"
)


def test_the_scan_resolves_every_write_it_finds():
    """The honesty check. A write whose key cannot be computed would shrink the
    surface silently, so it fails here instead."""
    scan = surface_scan.scan_model_surface(list(LEGACY_MODEL_SOURCES))
    assert surface_scan.unexplained(scan) == []


def test_the_model_forward_surface_is_forty_three_keys():
    keys = legacy_model_keys()
    assert len(keys) == 43, sorted(keys)


def test_every_legacy_key_has_exactly_one_disposition():
    keys = legacy_model_keys()
    missing = sorted(keys - set(DISPOSITIONS))
    assert not missing, (
        f"{missing} are returned by a frozen model forward and have no row in "
        f"observable_coverage.DISPOSITIONS. Add a Spec, a Derivative or a Drop "
        f"row; an unclassified output is one nothing downstream can pad, "
        f"unpad or train."
    )
    stale = sorted(set(DISPOSITIONS) - keys)
    assert not stale, (
        f"{stale} have a disposition row and are no longer returned by any "
        f"frozen model forward. Remove the rows."
    )


@pytest.mark.parametrize(
    "key", sorted(k for k, d in DISPOSITIONS.items() if isinstance(d, Spec))
)
def test_a_spec_row_builds_a_valid_observable_spec(key):
    """The classification the padding depends on has to exist for every row,
    and a row states its irreps either outright or with the model-dependent
    part named. There is no third state: no row is allowed to defer the
    question to a later ticket."""
    row = DISPOSITIONS[key]
    per_atom = per_atom_of(key)
    assert per_atom is not None, (
        f"{key!r} is declared as an observable but the golden harness gives it "
        f"a kind that is neither per-atom nor per-graph. It cannot be a Spec "
        f"row; make it a Derivative or a Drop."
    )
    assert bool(row.irreps) != bool(row.irreps_pattern), (
        f"{key!r} must state exactly one of `irreps` and `irreps_pattern`. "
        f"Neither means the row was never worked out, and both means it is "
        f"unclear which one a reader should believe."
    )
    if row.irreps_pattern:
        assert row.set_by.strip(), (
            f"{key!r} says its shape depends on the model without saying on "
            f"what. Name the parameter in `set_by`."
        )
        return
    channel = channel_of(key)
    assert channel is not None
    spec = ObservableSpec(
        name=key,
        irreps=row.irreps,
        per_atom=per_atom,
        # The legacy unit, as the golden harness records it. This ticket does
        # not canonicalise unit strings, so `Ang` is left as it is written
        # there rather than rewritten to `Å`.
        units=channel.unit,
        # Neutral: which normalization each family wants is decided by the
        # ticket that builds its head, not by this coverage table.
        normalization="none",
    )
    assert spec.per_atom == per_atom
    assert spec.dimension >= 1


def test_no_observable_row_defers_its_irreps_to_another_ticket():
    """The whole set was worked out, and this asserts it stays that way. A row
    that cannot say what its irreps are is a row nobody has read the model
    for, and it should fail here rather than sit as a TODO."""
    unresolved = sorted(
        key
        for key, row in DISPOSITIONS.items()
        if isinstance(row, Spec) and not row.irreps and not row.irreps_pattern
    )
    assert not unresolved, unresolved


@pytest.mark.parametrize(
    "key", sorted(k for k, d in DISPOSITIONS.items() if isinstance(d, Derivative))
)
def test_a_derivative_row_resolves_through_the_rule(key):
    row = DISPOSITIONS[key]
    assert row.wrt in DECLARED_INPUTS, (
        f"{key!r} is differentiated against {row.wrt!r}, which is not one of "
        f"the declared inputs {sorted(DECLARED_INPUTS)}."
    )
    parent = DISPOSITIONS.get(row.of)
    assert isinstance(parent, (Spec, Derivative)), (
        f"{key!r} is the derivative of {row.of!r}, which is not itself a "
        f"declared observable or a derivative of one. A derivative chain has "
        f"to ground out in something declared."
    )
    name = derivative_name(row.of, row.wrt)
    assert name.isidentifier()
    derived_sign = derivative_sign(row.of, row.wrt)
    assert row.sign == derived_sign, (
        f"{key!r} is reported with sign {row.sign:+d} by the frozen tree and "
        f"the rule derives {derived_sign:+d}. There is no way to annotate that "
        f"away, deliberately: either the row differentiates the wrong thing, "
        f"or the grammar has a real gap. `hessian` was the first candidate for "
        f"such a gap and turned out to be the former: it is a second "
        f"derivative of the energy, not a first derivative of the forces."
    )


def test_the_three_named_derivatives_keep_their_legacy_names():
    """forces, stress and magforces are the pairs that have a name of their own,
    and the two negated ones are the two the frozen tree negates."""
    assert derivative_name("energy", "pos") == "forces"
    assert derivative_sign("energy", "pos") == -1
    assert derivative_name("energy", "cell") == "stress"
    assert derivative_sign("energy", "cell") == +1
    assert derivative_name("energy", "magmom") == "magforces"
    assert derivative_sign("energy", "magmom") == -1
    for name in ("forces", "stress", "magforces"):
        assert isinstance(DISPOSITIONS[name], Derivative)


def test_the_renamed_derivatives_are_renamed_and_not_lost():
    """Two legacy spellings become their rule-derived names. A rename is a
    breaking change and is fine; losing one is not.

    `BEC` is deliberately absent. It looks like a third, and treating it as one
    would have merged it into `dmu_dr`; see its row for why the two are
    different quantities. `hessian` is absent for a different reason: it is a
    second derivative and carries a Drop row.
    """
    renamed = {
        "dmu_dr": "d_dipole_d_pos",
        "dalpha_dr": "d_polarizability_d_pos",
    }
    for key, expected in renamed.items():
        row = DISPOSITIONS[key]
        assert derivative_name(row.of, row.wrt) == expected
    assert isinstance(DISPOSITIONS["BEC"], Spec)


def test_no_two_derivative_rows_resolve_to_one_name():
    """Two legacy keys collapsing onto a single canonical name is either the
    abstraction working or a silent merge of two different quantities. It has
    to be looked at rather than discovered later, so it fails here."""
    seen: dict[str, str] = {}
    for key, row in DISPOSITIONS.items():
        if not isinstance(row, Derivative):
            continue
        name = derivative_name(row.of, row.wrt)
        assert name not in seen, (
            f"{key!r} and {seen[name]!r} both resolve to {name!r}. If they are "
            f"one quantity, say so and drop one row; if they are not, one of "
            f"them is not a derivative of what it claims."
        )
        seen[name] = key


def test_a_key_the_classification_cannot_place_is_never_a_spec_row():
    """`edge_forces` is per-edge and a hessian is a square over 3N degrees of
    freedom. Neither is per-atom or per-graph, so neither can be an observable
    under a classification that only has those two."""
    for key, row in DISPOSITIONS.items():
        if per_atom_of(key) is None:
            assert isinstance(row, (Derivative, Drop)), key


@pytest.mark.parametrize(
    "key", sorted(k for k, d in DISPOSITIONS.items() if isinstance(d, Drop))
)
def test_a_drop_row_says_what_owns_the_key_instead(key):
    reason = DISPOSITIONS[key].reason
    assert reason.strip()
    assert len(reason.split()) >= 8, (
        f"the reason for dropping {key!r} is too short to be a decision "
        f"anybody can review later: {reason!r}"
    )


def test_the_scf_trio_is_reached_at_all():
    """The three keys an extraction that stops at return literals never sees.
    They are assigned onto the output after it is built."""
    keys = legacy_model_keys()
    for key in ("scf_energy_history", "scf_steps", "equilibrated_magmom"):
        assert key in keys
        assert isinstance(DISPOSITIONS[key], Drop)


# ---------------------------------------------------------------------------
# The three-layer surface
# ---------------------------------------------------------------------------


def documented_counts() -> dict[str, tuple[int, int]]:
    """The per-layer `(keys, new here)` pairs the surface document records."""
    rows = {}
    for line in SURFACE_DOC.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\|\s*\((a|b|c)\)[^|]*\|(.*)$", line)
        if match is None:
            continue
        cells = [cell.strip() for cell in match.group(2).split("|")]
        numbers = [int(cell) for cell in cells if cell.isdigit()]
        assert len(numbers) == 2, line
        rows[match.group(1)] = (numbers[0], numbers[1])
    return rows


def test_the_surface_document_records_the_derived_counts():
    model = legacy_model_keys()
    calculator = legacy_calculator_keys()
    evaluation = legacy_eval_keys()
    derived = {
        "a": (len(model), len(model)),
        "b": (len(calculator), len(calculator - model)),
        "c": (len(evaluation), len(evaluation - model - calculator)),
    }
    assert documented_counts() == derived
    assert derived == {"a": (43, 43), "b": (31, 15), "c": (13, 3)}


def test_the_union_is_sixty_one_names():
    union = legacy_model_keys() | legacy_calculator_keys() | legacy_eval_keys()
    assert len(union) == 61
    assert "**61**" in SURFACE_DOC.read_text(encoding="utf-8")
