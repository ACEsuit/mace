"""The feature-inventory gate, as a test so CI fails when a feature has no row.

`check_inventory.py` is runnable on its own (`python3 tests/golden/check_inventory.py`);
this file is what makes it a gate rather than a tool someone remembers to run.

The negative tests matter as much as the positive one: a completeness checker that
cannot fail is worse than no checker, because it reports coverage that does not
exist. Each of the four failure conditions of the per-dest gate is exercised
against synthetic rows.
"""

import importlib.util
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


def _row(ident, disposition="KEEP", pinned="P0-5", dest="CLI-1", ret="RET-6"):
    return Row(
        ident=ident,
        feature="`--thing`",
        source="`mace/tools/arg_parser.py:1`",
        disposition=disposition,
        pinned_by=pinned,
        destination=dest,
        retirement=ret,
        status="todo",
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
        (_row("x.a", dest=""), "no destination ticket"),
        (_row("x.a", ret=""), "no retirement ticket"),
        (_row("x.a", ret="later"), "no retirement ticket"),
    ],
)
def test_row_hygiene_rejects(row, expected):
    problems = check_inventory.check_row_hygiene([row])
    assert any(expected in p for p in problems), problems


def test_row_hygiene_accepts_the_deliberate_escapes():
    """A DROP with a reason needs no pinning test, and a row with no legacy code
    to delete says so instead of naming a RET ticket."""
    rows = [
        _row("x.a", disposition="DROP — nobody uses it", pinned="—", ret="RET-1"),
        _row("x.b", ret="n/a — test infrastructure, not legacy code"),
        _row("x.c", pinned="⚠️ gap (add to P0-5)"),
    ]
    assert check_inventory.check_row_hygiene(rows) == []


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
