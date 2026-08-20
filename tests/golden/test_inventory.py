"""The feature-inventory gate, as a test so CI fails when a feature has no row.

It runs in the per-PR ci-core `unit` job, which already collects this directory:
the row has to be written by the pull request that adds the feature, because a
nightly only ever reports it late, and by then the row is written after the fact
if at all.

`check_inventory.py` is runnable on its own (`python3 tests/golden/check_inventory.py`);
this file is what makes it a gate rather than a tool someone remembers to run.

The negative tests matter as much as the positive one: a completeness checker that
cannot fail is worse than no checker, because it reports coverage that does not
exist. Each of the four failure conditions of the per-dest gate is exercised
against synthetic rows.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).with_name("check_inventory.py")
_spec = importlib.util.spec_from_file_location("check_inventory", _MODULE_PATH)
check_inventory = importlib.util.module_from_spec(_spec)
sys.modules["check_inventory"] = check_inventory
_spec.loader.exec_module(check_inventory)

Decl = check_inventory.Decl
Row = check_inventory.Row
SourceSet = check_inventory.SourceSet


def _row(ident, disposition="KEEP", pinned="`tests/unit/test_compile.py`"):
    return Row(
        ident=ident,
        feature="`--thing`",
        source="`mace/tools/arg_parser.py:1`",
        disposition=disposition,
        pinned_by=pinned,
        line=1,
    )


def _source(**decls):
    return SourceSet(
        "mace_run_train",
        "train.",
        "dests",
        {k: Decl(k, v, "mace/tools/arg_parser.py:907") for k, v in decls.items()},
    )


def test_inventory_covers_every_source():
    assert check_inventory.main() == 0, "see the output above for the offending rows"


def test_every_row_has_a_source_line():
    """Rows are unusable as a work list without a declaration site to fix."""
    rows, problems = check_inventory.read_rows()
    assert not problems
    assert len(rows) > 500
    assert all(row.source.strip() not in ("", "—") for row in rows)


def test_a_dest_with_no_row_fails():
    ok, report = check_inventory.check_set(
        _source(beta1_schedulefree="--beta1_schedulefree"), []
    )
    assert not ok
    assert "1 dests without a disposition" in report[0]
    # the message has to carry the site, or the fix needs a second lookup
    assert "beta1_schedulefree" in report[1]
    assert "--beta1_schedulefree" in report[1]
    assert "mace/tools/arg_parser.py:907" in report[1]


def test_a_row_with_no_disposition_fails():
    ok, report = check_inventory.check_set(
        _source(lr="--lr"), [_row("train.lr", disposition="")]
    )
    assert not ok
    assert "1 dests without a disposition" in report[0]


def test_a_review_disposition_fails():
    ok, _ = check_inventory.check_set(
        _source(lr="--lr"), [_row("train.lr", disposition="REVIEW — undecided")]
    )
    assert not ok


def test_a_stale_row_fails():
    """A renamed or deleted flag must not leave a row behind claiming coverage."""
    ok, report = check_inventory.check_set(
        _source(lr="--lr"), [_row("train.lr"), _row("train.deleted_flag")]
    )
    assert not ok
    assert any("no longer declares" in line for line in report)
    assert any("train.deleted_flag" in line for line in report)


def test_a_stale_undecided_row_is_reported_and_not_looked_up():
    """A row can be both stale and undecided, and only the stale half has a site.

    The undecided report prints the declaration the row belongs to, so it can
    only cover keys the source still declares; a REVIEW row for a deleted flag
    is a stale row and belongs to that report.
    """
    ok, report = check_inventory.check_set(
        _source(lr="--lr"),
        [_row("train.lr"), _row("train.deleted_flag", disposition="REVIEW")],
    )
    assert not ok
    assert not any("without a disposition" in line for line in report)
    assert any("train.deleted_flag" in line for line in report)


def test_a_matching_row_passes():
    ok, report = check_inventory.check_set(_source(lr="--lr"), [_row("train.lr")])
    assert ok
    assert report == []


@pytest.mark.parametrize(
    "row, expected",
    [
        (_row("x.a", disposition="DROP"), "no justification"),
        (_row("x.a", pinned=""), "no pinning test"),
        (_row("x.a", pinned="—"), "no pinning test"),
    ],
)
def test_row_hygiene_rejects(row, expected):
    problems = check_inventory.check_row_hygiene([row])
    assert any(expected in p for p in problems), problems


def test_row_hygiene_accepts_the_deliberate_escapes():
    """A DROP with a reason needs no pinning test, and an honest gap marker is a
    legitimate cell -- the point of the column is that it can say "nothing"."""
    rows = [
        _row("x.a", disposition="DROP — nobody uses it", pinned="—"),
        _row("x.b", pinned="⚠️ gap (add a case to `tests/unit/test_radial.py`)"),
    ]
    assert check_inventory.check_row_hygiene(rows) == []


@pytest.mark.parametrize(
    "pin, expected",
    [
        # The weakness this set closes: the old rule was "not empty", so the
        # literal string TODO was a pin and the gate still printed "all
        # sources covered".
        ("TODO", "free text"),
        ("later", "free text"),
        ("the tests", "free text"),
        # A ticket id used to be a legal pin. It named a test that did not exist
        # yet, so the cell recorded an intention and read as coverage.
        ("P0-5", "free text"),
        ("CORE-4 will write it", "free text"),
        # A dash is how a row says "nothing pins this, and nothing has to".
        # A dash followed by prose says something, and it used to be the one
        # spelling that escaped both checks: this one skipped anything opening
        # with a dash, and the hygiene rule only rejected the bare dash. So a
        # KEEP row could open the cell with a dash and pass with no evidence
        # at all -- and one did (`pkg.public_import_surface`).
        ("— see the docs", "free text"),
        ("— (deliberate break)", "free text"),
        # A pin naming a test that does not exist reads as coverage and is
        # therefore worse than an honest ⚠️ gap marker.
        ("`tests/unit/test_not_here.py`", "does not exist on disk"),
        (
            "`tests/workflows/test_run_train.py::test_never_written`",
            "no `test_never_written` exists",
        ),
        # Every path in a compound pin is claimed, not just the leading one.
        (
            "`tests/unit/test_compile.py` + `tests/unit/test_gone.py`",
            "does not exist on disk",
        ),
        # And the hole the "must resolve" rule left open: a path that exists
        # and pins nothing. `tests/` is the whole suite, so it is true of
        # every row and discriminates none of them -- TODO wearing a path.
        ("`tests/`", "the entire suite"),
        ("`tests`", "the entire suite"),
        ("`tests/unit`", "the whole fast CPU tier"),
        ("`tests/workflows`", "the whole e2e tier"),
        ("`tests/extensions`", "the whole optional-dependency tier"),
        ("`tests/unit/test_compile.py` + `tests/unit`", "the whole fast CPU tier"),
    ],
)
def test_a_pin_that_resolves_to_nothing_is_rejected(pin, expected):
    problems = check_inventory.check_pins([_row("x.a", pinned=pin)])
    assert any(expected in p for p in problems), problems


def test_the_specificity_floor_is_a_floor_and_not_a_ban_on_directories():
    """`tests/extensions/magnetic` is a legitimate pin and has to stay one.

    It is the whole of that family's coverage, and splitting the claim over
    its files would be noise. What separates it from `tests/unit` is not
    being a directory, it is being *about* the row -- expressed here as a
    depth, which admits every per-family directory in the tree and rejects
    exactly the tier-level ones.
    """
    for pin in (
        "`tests/extensions/magnetic`",
        "`tests/extensions/polar`",
        "`tests/integrations/lammps`",
    ):
        assert check_inventory.check_pins([_row("x.a", pinned=pin)]) == [], pin

    # every coarse name is a real directory, or the rule is guarding nothing
    for name in check_inventory.TOO_COARSE_TO_PIN:
        assert (check_inventory.REPO / name).is_dir(), name

    # and the depth rule and the named list agree about the tiers
    for name in check_inventory.TOO_COARSE_TO_PIN:
        if name != "tests":
            assert len(Path(name).parts) < check_inventory.MIN_PIN_DIRECTORY_DEPTH


def test_a_directory_pin_may_still_name_a_test_inside_it():
    """The floor is about vagueness, not about directories, so a `::node_id`
    makes even a coarse directory specific again."""
    assert (
        check_inventory.check_pins(
            [_row("x.a", pinned="`tests/unit::test_run_train_lbfgs`")]
        )
        != []
    ), "that test does not live under tests/unit, so this must still fail"
    assert (
        check_inventory.check_pins(
            [_row("x.a", pinned="`tests/workflows::test_run_train_lbfgs`")]
        )
        == []
    )


@pytest.mark.parametrize(
    "pin",
    [
        "`tests/unit/test_compile.py`; committee ⚠️ gap (§12.1)",
        "⚠️ gap (add a case to `tests/workflows/test_cli_contracts.py`)",
        "⚠️ gap (idem — spelled `--disallow_random_padding`, stored inverted)",
        "`tests/unit/test_compile.py`",
        "`tests/extensions/magnetic`",
        "`tests/extensions/magnetic::test_run_magnetic_scf`",
        "`tests/workflows/test_run_train.py::test_run_train_lbfgs`",
        "`tests/extensions/les` + `tests/unit/test_compile.py`",
        "`tests/golden/anchors.py::ANCHORS[tiny_mace]`",
        "the suite itself",
        "the lint job itself",
        "—",
        "",
    ],
)
def test_the_pin_vocabulary_actually_in_use_is_accepted(pin):
    """Every shape the inventory uses today, so the rule cannot be tightened
    into rejecting the file it gates."""
    assert check_inventory.check_pins([_row("x.a", pinned=pin)]) == []


def test_the_two_non_test_pins_are_named_and_justified():
    """Allowed by name, not by a wildcard, and each says why it is not a file.

    Both are extras groups: the only thing that can fail is a CI job
    installing them, and no test inside the suite can assert that the suite's
    own dependencies resolve. Everything else that used this phrasing -- the
    thirteen pytest markers -- now points into `tests/conftest.py`, each
    capability at its own `CAPABILITY_PROBES` entry rather than at the file.
    """
    assert set(check_inventory.NON_TEST_PINS) == {
        "the suite itself",
        "the lint job itself",
    }
    for pin, reason in check_inventory.NON_TEST_PINS.items():
        assert len(reason) > 40, pin
    rows, _ = check_inventory.read_rows()
    users = sorted(r.ident for r in rows if r.pinned_by in check_inventory.NON_TEST_PINS)
    assert users == ["extra.dev", "extra.test"], users


def test_every_marker_row_pins_its_own_probe():
    rows, _ = check_inventory.read_rows()
    assert check_inventory.check_marker_rows(rows) == []


def test_a_marker_row_pinning_the_bare_conftest_is_rejected():
    """The defect: twelve rows pinned `tests/conftest.py` and the only thing
    checked was that the file exists. It does, so `marker.anything` passed
    for a reason that had nothing to do with the marker."""
    problems = check_inventory.check_marker_rows(
        [_row("marker.cueq", pinned="`tests/conftest.py`")]
    )
    assert any("CAPABILITY_PROBES[cueq]" in p for p in problems), problems

    problems = check_inventory.check_marker_rows(
        [_row("marker.slow", pinned="`tests/conftest.py`")]
    )
    assert any("cost marker pinned by the bare conftest" in p for p in problems)


def test_a_marker_row_pinning_the_wrong_probe_is_rejected():
    """Tying each row to *its own* entry is the whole point: pinning some
    other capability's probe resolves on disk and still says nothing."""
    problems = check_inventory.check_marker_rows(
        [_row("marker.cueq", pinned="`tests/conftest.py::CAPABILITY_PROBES[oeq]`")]
    )
    assert any("CAPABILITY_PROBES[cueq]" in p for p in problems), problems


def test_a_cost_marker_may_not_claim_a_capability_probe():
    """The mirror rule, and the one the `marker.timeout` row was written to
    protect: absorbing a cost marker into the capability manifest is how
    `timeout` would start gating CI jobs on a plugin's presence."""
    problems = check_inventory.check_marker_rows(
        [_row("marker.timeout", pinned="`tests/conftest.py::CAPABILITY_PROBES[gpu]`")]
    )
    assert any("has no probe" in p for p in problems), problems


def test_a_capability_without_a_marker_row_is_rejected():
    """Both directions. A probe with no row is a capability the inventory
    does not know it has."""
    rows, _ = check_inventory.read_rows()
    kept = [r for r in rows if r.ident != "marker.network"]
    problems = check_inventory.check_marker_rows(kept)
    assert any("marker.network" in p for p in problems), problems


def test_the_probe_pin_resolves_against_the_real_dict():
    """A `TABLE[key]` pin is machine-checked against the dict literal, so a
    renamed capability fails here rather than in whichever CI job silently
    stops enforcing it."""
    conftest = check_inventory.CONFTEST
    assert check_inventory._dict_has_key(conftest, "CAPABILITY_PROBES", "cueq")
    assert not check_inventory._dict_has_key(conftest, "CAPABILITY_PROBES", "nope")
    assert check_inventory._dict_literal(conftest, "NOT_A_DICT_HERE") is None
    problems = check_inventory.check_pins(
        [_row("x.a", pinned="`tests/conftest.py::CAPABILITY_PROBES[nope]`")]
    )
    assert any("no `CAPABILITY_PROBES[nope]` exists" in p for p in problems), problems


def test_no_row_is_pinned_by_a_ticket_id():
    """The rule that replaced the ticket vocabulary, asserted against the real
    file: a pin names a test that runs, so the gate can resolve it."""
    rows, _ = check_inventory.read_rows()
    offenders = [
        (r.ident, r.pinned_by)
        for r in rows
        if re.search(r"\b[A-Z][A-Z0-9]{1,7}-\d+[a-z]?\b", r.pinned_by)
    ]
    assert offenders == []


def test_a_dict_entry_pin_resolves_through_an_annotated_assignment():
    """`ANCHORS: Dict[str, dict] = {...}` is still a dict literal. Reading only
    the bare `NAME = {...}` form reported a missing entry for a dict that has
    it, which is a false failure on most of this codebase's tables."""
    problems = check_inventory.check_pins(
        [_row("x.a", pinned="`tests/golden/anchors.py::ANCHORS[tiny_mace]`")]
    )
    assert problems == []


def test_duplicate_ids_are_rejected():
    problems = check_inventory.check_row_hygiene([_row("x.a"), _row("x.a")])
    assert any("duplicate id" in p for p in problems)


def test_argparse_dest_resolution():
    """The dest, not the option string, is the key: an explicit `dest=` wins, then
    the first long option, then the positional."""
    import ast

    tree = ast.parse(
        "p.add_argument('--swa_lr', '--stage_two_lr')\n"
        "p.add_argument('--disallow_random_padding_pt', dest='allow_random_padding_pt')\n"
        "p.add_argument('model_path')\n"
        "p.add('--config', is_config_file=True)\n"
        "seen.add('not_a_flag')\n"
    )
    dests = check_inventory._dests(tree, Path(check_inventory.REPO / "x.py"))
    assert set(dests) == {
        "swa_lr",
        "allow_random_padding_pt",
        "model_path",
        "config",
    }
    assert dests["swa_lr"].detail == "--swa_lr --stage_two_lr"


def test_the_three_output_key_surfaces_are_non_empty():
    """The user-observable surface is three sets, not one: what the model returns,
    what the calculator exposes, and what the eval CLI writes into the XYZ."""
    assert len(check_inventory.source_model_output_keys()) > 30
    assert len(check_inventory.source_calculator_result_keys()) > 20
    assert len(check_inventory.source_eval_output_keys()) > 10


# ---------------------------------------------------------------------------
# the gap audit: a heuristic, so what is tested is its shape, not its verdicts
# ---------------------------------------------------------------------------


def _gap_row(ident, feature="`--thing`", gap="⚠️ gap (nothing yet)"):
    return Row(
        ident=ident,
        feature=feature,
        source="`mace/tools/arg_parser.py:1`",
        disposition="KEEP",
        pinned_by=gap,
        line=1,
    )


def test_the_audit_finds_a_gap_the_suite_already_covers():
    """The failure it exists for: seventy-seven rows here stayed marked as gaps
    after the tests closing them had merged."""
    suspects = check_inventory.audit_gaps(
        [_gap_row("out.calc.energy_var", feature="`energy_var` — committee")]
    )

    assert len(suspects) == 1
    assert "test_calculator_committee" in suspects[0]


def test_the_audit_says_nothing_about_a_row_nothing_covers():
    row = _gap_row("train.no_such_flag_anywhere", feature="`--no_such_flag_anywhere`")

    assert check_inventory.audit_gaps([row]) == []


def test_a_pinned_row_is_not_audited():
    """Only gap rows are claims about the suite; a pin is checked by the gate."""
    pinned = _gap_row("out.calc.energy_var", gap="`tests/unit/test_compile.py`")
    assert check_inventory.audit_gaps([pinned]) == []


def test_the_documentation_rows_are_out_of_scope():
    """§18 tracks a promise rather than a code surface, so no test can close one
    of its gaps -- and matching page titles turns up "training" and "ase"."""
    assert check_inventory.audit_gaps([_gap_row("doc.training")]) == []


def test_the_harness_own_suite_is_not_evidence():
    """`test_harness.py` asserts that the harness *knows* a channel, which is the
    registry case the audit discounts, written as assertions rather than data."""
    functions = check_inventory._asserting_test_functions()

    assert functions, "the scan found no asserting tests at all"
    assert not [path for path, _, _ in functions if path in check_inventory.AUDIT_NON_EVIDENCE]


def test_a_subject_too_common_to_judge_says_so_instead_of_guessing():
    suspects = check_inventory.audit_gaps([_gap_row("reg.None")])

    assert len(suspects) == 1
    assert "too common a word" in suspects[0]


def test_the_subjects_come_from_the_id_and_the_option_strings():
    row = _gap_row("train.apply_cutoff", feature="`--apply_cutoff` `--legacy_name`")

    assert check_inventory.audit_subjects(row) == [
        "apply_cutoff",
        "legacy_name",
    ]


def test_the_gate_does_not_run_the_audit():
    """The structural guarantee. The gate is a set of comparisons that are either
    right or wrong; the audit is a search that can be neither, so wiring it into
    a required check would make a heuristic block pull requests -- and it would
    be switched off the first time it was wrong."""
    import inspect

    gate = inspect.getsource(check_inventory.main)
    audited = gate.index("audit_gaps")
    returns_early = gate.index('return 1 if suspects else 0')
    comparisons = gate.index("for source in collect_sources()")

    assert audited < returns_early < comparisons, (
        "the audit no longer short-circuits ahead of the gate, so a heuristic "
        "now runs as part of a required check"
    )
