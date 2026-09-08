"""The v1.0 deprecation table, and the guards that keep it honest.

The table is advance notice: it says, from a release that still has the
feature, which surfaces MACE v1.0 removes or replaces. Four things about it can
rot silently, and each has a test here.

* A row that has drifted from the inventory it came from. The inventory is in
  this tree, at tests/golden/feature_inventory.md, so the table can be checked
  against its source rather than trusted: same ids, same dispositions.
* A message that names a v1 command. The last 0.3.x release predates the v1
  CLI, so "use ``mace train``" points at a binary the reader cannot run.
* An emission site citing an identifier the table does not have. The sites are
  spread over the tree, so a typo in one of them silences that warning and
  nothing else changes.
* A row whose flag was renamed. The row still reads as coverage while the
  warning can no longer fire, because the dest it keys on is gone.
"""

import argparse
import ast
import logging
import re
import warnings
from pathlib import Path

import pytest

from mace.tools import deprecation
from mace.tools.arg_parser import build_default_arg_parser, build_preprocess_arg_parser

MACE_ROOT = Path(deprecation.__file__).resolve().parent.parent

#: The rows that do not come from the rewrite's feature inventory. The inventory
#: gives these three CLIs no entry-point row because nothing is registered to
#: keep or drop, but python -m still reaches them.
NOT_FROM_INVENTORY = {
    "ep.convert_e3nn_oeq",
    "ep.convert_oeq_e3nn",
    "ep.convert_e3nn_hybrid",
}

#: The inventory the table is generated from, in this tree. Anchored on the
#: package rather than the working directory, so the test does not depend on
#: where pytest was invoked from.
INVENTORY = MACE_ROOT.parent / "tests" / "golden" / "feature_inventory.md"


@pytest.fixture(autouse=True)
def _forget_warnings():
    deprecation.reset_warned()
    yield
    deprecation.reset_warned()


def test_the_table_is_well_formed():
    assert deprecation.DEPRECATIONS, "the table is empty"
    for dep_id, dep in deprecation.DEPRECATIONS.items():
        assert dep.id == dep_id
        assert dep.kind in (deprecation.DROP, deprecation.MERGE), dep_id
        assert dep.what.strip(), dep_id
        assert dep.why.strip(), dep_id
        # A reason is prose that completes "MACE v1.0 removes X: ...", so it
        # must not arrive already punctuated or still carrying markdown.
        assert not dep.why.endswith("."), dep_id
        assert "`" not in dep.why and "**" not in dep.why, dep_id


def test_a_drop_removes_and_a_merge_replaces():
    drops = deprecation.rows(deprecation.DROP)
    merges = deprecation.rows(deprecation.MERGE)
    assert drops and merges
    for dep in drops:
        assert "removes" in dep.message()
    for dep in merges:
        assert "replaces" in dep.message()


def test_no_message_names_a_v1_command():
    """The plan's rule: a 0.3.x warning may not point at the v1 CLI.

    ``mace.cli.convert_e3nn_oeq`` is a module path and stays legal; ``mace
    train`` is a command that will not exist until after the last 0.3.x
    release, and naming it would send the reader to a binary they cannot run.
    """
    v1_command = re.compile(r"\bmace (train|eval|export|data|model|validate|fit)\b")
    offenders = [
        dep.id
        for dep in deprecation.DEPRECATIONS.values()
        if v1_command.search(dep.message())
    ]
    assert not offenders, f"these messages name a v1 command: {offenders}"


def _inventory_dispositions():
    """The non-KEEP rows of the inventory, as {id: DROP|MERGE}."""
    text = INVENTORY.read_text(encoding="utf-8")
    found = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            continue
        row_id, _, _, disposition, _ = cells
        kind = disposition.split("\u2014")[0].strip()
        if kind in (deprecation.DROP, deprecation.MERGE):
            found[row_id.strip("`")] = kind
    return found


def test_the_table_agrees_with_the_inventory_it_came_from():
    """Same ids and same dispositions, both directions.

    This is the test that makes the table trustworthy rather than merely
    plausible. A flag added with a DROP row in the inventory and not added
    here is a feature that will disappear without warning, which is the exact
    failure the inventory exists to prevent; a disposition changed there and
    not here makes the warning say the wrong thing.
    """
    inventory = _inventory_dispositions()
    assert inventory, f"{INVENTORY} carries no DROP or MERGE rows"

    table = {
        dep_id: dep.kind
        for dep_id, dep in deprecation.DEPRECATIONS.items()
        if dep_id not in NOT_FROM_INVENTORY
    }
    missing = sorted(set(inventory) - set(table))
    surplus = sorted(set(table) - set(inventory))
    disagreeing = sorted(
        f"{k}: inventory says {inventory[k]}, table says {table[k]}"
        for k in set(inventory) & set(table)
        if inventory[k] != table[k]
    )
    assert not missing, f"in the inventory and not in the table: {missing}"
    assert not surplus, f"in the table and not in the inventory: {surplus}"
    assert not disagreeing, f"dispositions disagree: {disagreeing}"


def test_every_extra_row_is_a_declared_one():
    """Only the three unregistered converter CLIs may be outside the inventory.

    Without this, the exemption above could be widened to hide any drift.
    """
    entry_points = {
        dep_id
        for dep_id in deprecation.DEPRECATIONS
        if dep_id.startswith("ep.") and not dep_id.startswith("ep.mace_")
    }
    assert entry_points == NOT_FROM_INVENTORY


@pytest.mark.parametrize(
    "prefix,parser",
    [("train", build_default_arg_parser()), ("prep", build_preprocess_arg_parser())],
)
def test_every_flag_row_still_names_a_live_dest(prefix, parser):
    """A renamed flag must break this test rather than stop warning quietly."""
    dests = {action.dest for action in parser._actions}
    rows = {
        dep_id
        for dep_id in deprecation.DEPRECATIONS
        if dep_id.startswith(f"{prefix}.")
    }
    dead = {r for r in rows if r[len(prefix) + 1 :] not in dests}
    assert not dead, f"{prefix} rows whose dest no longer exists: {sorted(dead)}"


def test_every_emission_site_cites_a_row_that_exists():
    """Scan the package for warn(...) calls and resolve every identifier.

    This is the guard that a typo at a call site cannot pass: an unknown
    identifier raises at the site, but only if the site is ever reached, and
    most of these are on paths the unit suite does not run.
    """
    cited = set()
    for path in MACE_ROOT.rglob("*.py"):
        if "torch_geometric" in path.parts or path.name.startswith("deprecation"):
            continue
        text = path.read_text(encoding="utf-8")
        if "mace.tools.deprecation" not in text and "import deprecation" not in text:
            continue
        # warnings.warn is also spelled warn(...) in this tree, and its argument
        # is prose. An identifier never contains whitespace.
        cited.update(
            arg
            for arg in re.findall(r'\bwarn\(\s*"([^"]+)"', text)
            if not re.search(r"\s", arg)
        )
        cited.update(f"env.{n}" for n in re.findall(r'\bwarn_env\(\s*"([^"]+)"', text))
    assert cited, "found no emission sites at all, so this test proves nothing"
    unknown = sorted(c for c in cited if c not in deprecation.DEPRECATIONS)
    assert not unknown, f"emission sites cite rows that do not exist: {unknown}"


def test_every_scanned_namespace_has_rows_to_find():
    """A warn_args prefix that matches no row is a call that can never fire."""
    prefixes = set()
    for path in MACE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        prefixes.update(re.findall(r'warn_args\(\s*"([^"]+)"', text))
        prefixes.update(re.findall(r'warn_choice\(\s*"([^"]+)"', text))
    assert prefixes
    for prefix in sorted(prefixes):
        assert any(
            dep_id.startswith(f"{prefix}.") for dep_id in deprecation.DEPRECATIONS
        ), f"warn_args/warn_choice({prefix!r}) can never match a row"


def test_a_warning_fires_once_and_on_both_channels(caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert deprecation.warn("train.save_cpu") is True
        assert deprecation.warn("train.save_cpu") is False
    assert len(caught) == 1
    assert caught[0].category is FutureWarning
    assert "--save_cpu" in str(caught[0].message)
    assert sum("--save_cpu" in r.message for r in caplog.records) == 1


def test_an_unknown_identifier_is_an_error_at_the_call_site():
    with pytest.raises(KeyError, match="disposition table"):
        deprecation.warn("train.no_such_flag")


def test_only_the_options_actually_passed_are_reported():
    parser = build_default_arg_parser()
    passed = deprecation.explicit_options(parser, ["--name", "x", "--save_cpu"])
    assert set(passed) == {"name", "save_cpu"}
    # --default_dtype has a default and a deprecation row, and must not appear.
    assert "default_dtype" not in passed


def test_the_spelling_the_caller_wrote_is_the_one_quoted():
    """Deprecated aliases share a dest, so the dest cannot name the option."""
    parser = build_default_arg_parser()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deprecation.warn_args("train", parser, ["--stage_two_lr", "1e-3"])
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "--stage_two_lr" in message
    assert "--swa_lr" not in message


def test_an_equals_form_and_an_unambiguous_abbreviation_both_resolve():
    parser = build_default_arg_parser()
    assert "save_cpu" in deprecation.explicit_options(parser, ["--save_cpu=1"])
    assert "use_so3" in deprecation.explicit_options(parser, ["--use_so"])


def test_a_deprecated_model_choice_warns_and_a_kept_one_does_not():
    assert deprecation.warn_choice("choice", "BOTNet") == "choice.BOTNet"
    assert deprecation.warn_choice("choice", "NoSuchModel") is None
    assert deprecation.warn_choice("choice", None) is None


def test_an_environment_variable_warns_only_when_it_is_set(monkeypatch):
    monkeypatch.delenv("MACE_USE_CUEQ_CG", raising=False)
    assert deprecation.warn_env("MACE_USE_CUEQ_CG") == []
    monkeypatch.setenv("MACE_USE_CUEQ_CG", "1")
    assert deprecation.warn_env("MACE_USE_CUEQ_CG") == ["env.MACE_USE_CUEQ_CG"]


def test_the_report_covers_every_row():
    import io

    buffer = io.StringIO()
    deprecation.report(buffer)
    printed = buffer.getvalue()
    assert str(len(deprecation.DEPRECATIONS)) in printed
    for dep in deprecation.DEPRECATIONS.values():
        assert dep.what in printed, dep.id


def test_a_parser_with_no_deprecated_options_stays_quiet():
    parser = argparse.ArgumentParser()
    parser.add_argument("--something")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert deprecation.warn_args("train", parser, ["--something", "1"]) == []
    assert not caught


def test_the_silencing_recipe_in_the_readme_works():
    """The README tells users one filter silences the lot. Hold it to that.

    ``module="mace"`` is the filter people reach for first and it does not
    work, because the warning is attributed to the caller that passed the
    option, not to a module inside the package. The message prefix is what
    every entry shares, so that is what the README documents.
    """
    parser = build_default_arg_parser()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message="MACE v1.0", category=FutureWarning)
        deprecation.warn_args("train", parser, ["--save_cpu", "--use_so3"])
    assert not caught

    deprecation.reset_warned()
    with warnings.catch_warnings(record=True) as still_caught:
        warnings.simplefilter("always")
        deprecation.warn_args("train", parser, ["--save_cpu"])
    assert len(still_caught) == 1


def test_every_message_carries_the_prefix_the_readme_promises():
    for dep in deprecation.DEPRECATIONS.values():
        assert dep.message().startswith("MACE v1.0 "), dep.id


def test_the_never_warned_surfaces_really_have_no_call_site():
    """The policy in deprecation.NEVER_WARNED, held to what the tree does.

    These rows exist for the record and for the report; MACE constructs them
    itself on every run, so a warning would fire for every user. The option
    that selects them is what warns instead. A call site appearing here is
    either a mistake or a decision that belongs in that docstring.
    """
    cited = set()
    for path in MACE_ROOT.rglob("*.py"):
        if "torch_geometric" in path.parts or path.name.startswith("deprecation"):
            continue
        text = path.read_text(encoding="utf-8")
        if "mace.tools.deprecation" not in text and "import deprecation" not in text:
            continue
        cited.update(
            arg
            for arg in re.findall(r'\bwarn\(\s*"([^"]+)"', text)
            if not re.search(r"\s", arg)
        )
    offenders = sorted(c for c in cited if c.split(".", 1)[0] in deprecation.NEVER_WARNED)
    assert not offenders, f"these are documented as never warned: {offenders}"


def test_the_never_warned_surfaces_are_all_real_namespaces():
    """A stale entry would make the test above vacuously true."""
    namespaces = {dep_id.split(".", 1)[0] for dep_id in deprecation.DEPRECATIONS}
    assert deprecation.NEVER_WARNED <= namespaces, (
        f"stale entries: {sorted(deprecation.NEVER_WARNED - namespaces)}"
    )


def test_no_message_uses_a_dash_as_a_connector():
    """House style: a dash never stands between two halves of a thought.

    The inventory these rows come from writes that way freely, so the
    generation step rewrites it and this holds the result.
    """
    offenders = [
        dep.id
        for dep in deprecation.DEPRECATIONS.values()
        if "—" in dep.message() or "–" in dep.message()
    ]
    assert not offenders, f"these messages contain a dash connector: {offenders}"


def test_warning_does_not_install_a_handler_on_the_root_logger():
    """The module-level logging.warning() would, and that breaks every run.

    ``logging.warning`` calls ``basicConfig`` when the root logger has no
    handlers, installing a ``StreamHandler`` at ``NOTSET``. These warnings fire
    while the CLI is still parsing, before ``setup_logger``, so that handler
    would then receive every record the run logs afterwards and duplicate the
    whole training log onto stderr. It cost a real failure in
    ``tests/workflows/test_foundation_kwargs.py``, where a run's stderr
    suddenly carried the parsed argument namespace.
    """
    root = logging.getLogger()
    saved = list(root.handlers)
    for handler in saved:
        root.removeHandler(handler)
    try:
        deprecation.warn("train.save_cpu")
        assert not root.handlers, (
            "warning installed a root handler; use a named logger, "
            f"not logging.warning: {root.handlers}"
        )
    finally:
        for handler in saved:
            root.addHandler(handler)


def test_the_message_still_reaches_the_log():
    """The named logger must not have made the log half of it silent."""
    logger = logging.getLogger("mace.tools.deprecation")
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture()
    logger.addHandler(handler)
    try:
        deprecation.warn("train.use_so3")
    finally:
        logger.removeHandler(handler)
    assert any("--use_so3" in message for message in records), records


def test_options_supplied_through_a_config_file_warn(tmp_path):
    """A YAML config sets values without putting anything in argv.

    The training and preprocessing parsers are configargparse parsers with
    ``is_config_file=True`` on ``--config``, and the README documents the YAML
    workflow, so reading argv alone would leave every config-file user with no
    advance notice at all. This is the route most training runs actually use to
    set ``--model``, ``--loss`` and the rest.
    """
    config = tmp_path / "run.yaml"
    config.write_text(
        "name: cfg\ntrain_file: fit.xyz\nsave_cpu: true\nuse_so3: true\n",
        encoding="utf-8",
    )
    parser = build_default_arg_parser()
    argv = ["--config", str(config)]
    parser.parse_args(argv)

    seen = deprecation.explicit_options(parser, argv)
    assert {"save_cpu", "use_so3"} <= set(seen), seen

    fired = deprecation.warn_args("train", parser, argv)
    assert "train.save_cpu" in fired
    assert "train.use_so3" in fired


def test_a_config_file_default_is_not_reported_as_chosen(tmp_path):
    """Only what the config actually sets, not everything it leaves alone."""
    config = tmp_path / "run.yaml"
    config.write_text("name: cfg\ntrain_file: fit.xyz\n", encoding="utf-8")
    parser = build_default_arg_parser()
    argv = ["--config", str(config)]
    parser.parse_args(argv)

    seen = deprecation.explicit_options(parser, argv)
    assert "save_cpu" not in seen
    assert "default_dtype" not in seen
    assert deprecation.warn_args("train", parser, argv) == ["train.config"]


def test_a_plain_argparse_parser_is_still_handled():
    """explicit_options must not require configargparse to be in play."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_cpu", action="store_true")
    assert not hasattr(parser, "get_source_to_settings_dict")
    assert deprecation.explicit_options(parser, ["--save_cpu"]) == {
        "save_cpu": "--save_cpu"
    }


def test_inspecting_a_parser_that_was_never_parsed_does_not_raise():
    """configargparse raises AttributeError until parse_args has run.

    ``warn_args`` is called on parsers in several CLIs, and one of them
    inspecting a parser it did not parse with must get "nothing from a config
    file" rather than a crash inside argument handling.
    """
    parser = build_default_arg_parser()
    with pytest.raises(AttributeError):
        parser.get_source_to_settings_dict()
    assert deprecation.explicit_options(parser, ["--save_cpu"]) == {
        "save_cpu": "--save_cpu"
    }


def test_no_warning_sits_inside_a_try_that_swallows_exceptions():
    """Under ``-W error`` these warnings are exceptions.

    A call inside a ``try`` whose handler catches ``Exception`` turns the
    warning into that handler's failure path. It happened once: the warning for
    the compiled side artifact sat inside the ``try`` guarding
    ``jit.compile``, so ``-W error`` reported a compile failure and the
    artifact was never written. ``-W error`` is a documented way to surface
    these, so it must not be able to corrupt a run.
    """
    helpers = {"warn", "warn_args", "warn_choice", "warn_env"}

    def called_helpers(node):
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in helpers:
                yield name, inner.lineno

    def swallows_broadly(handlers):
        for handler in handlers:
            if handler.type is None:
                return True
            names = [
                n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)
            ]
            if "Exception" in names or "BaseException" in names:
                return True
        return False

    offenders = []
    for path in MACE_ROOT.rglob("*.py"):
        if "torch_geometric" in path.parts or path.name.startswith("deprecation"):
            continue
        text = path.read_text(encoding="utf-8")
        if "mace.tools.deprecation" not in text and "import deprecation" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Try) or not swallows_broadly(node.handlers):
                continue
            for statement in node.body:
                for name, lineno in called_helpers(statement):
                    rel = path.relative_to(MACE_ROOT.parent)
                    offenders.append(f"{rel}:{lineno} calls {name}")
    assert not offenders, (
        "these deprecation warnings would become the handler's failure "
        f"under -W error: {offenders}"
    )
