"""That the black-box suite reaches the CLI through one door.

The 68 black-box functions are not rewritten for the rewrite; they are re-run
against it, with the engine chosen by an environment variable. That only works
if every one of them launches the command line the same way, through
`tests/helpers.run_mace_train`, which is the single place that knows about the
launcher.

A file that builds its own `subprocess.run([sys.executable, run_train, ...])`
does not get the engine. Under `MACE_ENGINE=v1` it runs **legacy** and passes,
which is a green tick for a stack that was never exercised: the exact silent
pass the re-run exists to avoid.

No file does, and the list that would record an exception is empty. A new one
fails this test, which is the point: the gap it would open is a green tick for
a stack that was never run.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers import NOT_MIGRATED

REPO_ROOT = Path(__file__).resolve().parents[2]
BLACK_BOX = ("tests/workflows", "tests/integrations")

#: Files that launch a MACE command line without going through the shared
#: runner, and therefore ignore `MACE_ENGINE`. Such a file runs legacy whatever
#: engine is asked for, so its v1 result means nothing.
#:
#: Empty, and meant to stay that way. Six files were on it and all six are
#: routed: the ones that need `Popen` take the argv prefix from the same place
#: the runner does, which is what keeps the launcher known in one module rather
#: than in each of them.
UNROUTED: set[str] = set()


def launches_a_command_line(source: str) -> bool:
    """Whether a file builds its own command line for a subprocess.

    The mark is `sys.executable` in a file that also starts a process: a routed
    file takes the whole argv prefix from `tests.helpers.cli_command`, engine
    and all, so it has no reason to name the interpreter.

    Read from the syntax tree rather than by searching the text, so a mention
    in a docstring or a comment is not a match. This file is such a mention
    itself.
    """
    tree = ast.parse(source)
    starts_process = False
    names_a_script = False
    for node in ast.walk(tree):
        # Any way of starting one, not only `subprocess.run`: the distributed
        # test uses `Popen`, and a detector that only knew one spelling would
        # report it as routed.
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "subprocess"
            and node.attr in {"run", "Popen", "call", "check_call", "check_output"}
        ):
            starts_process = True
        # `sys.executable` beside a script is the argv a routed file no longer
        # writes: `cli_command` returns the whole prefix, engine included.
        # Looking for the interpreter rather than for the script name is what
        # separates a file that builds its own command from one that asks for
        # the right one.
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "executable"
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        ):
            names_a_script = True
    return starts_process and names_a_script


def black_box_files() -> list[Path]:
    found = []
    for directory in BLACK_BOX:
        root = REPO_ROOT / directory
        if root.exists():
            found.extend(sorted(root.rglob("test_*.py")))
    return found


def test_the_scan_reaches_the_black_box_suite():
    """The precondition. "No file bypasses" is also true of no files at all."""
    files = black_box_files()
    assert len(files) >= 20, f"only {len(files)} black-box files were found"


def test_no_new_file_launches_the_command_line_on_its_own():
    """The bypass list does not grow.

    A file that builds its own subprocess ignores the engine, so its v1 run is
    a legacy run wearing a v1 label.
    """
    offenders = set()
    for path in black_box_files():
        relative = str(path.relative_to(REPO_ROOT))
        if launches_a_command_line(path.read_text()):
            offenders.add(relative)

    unexpected = sorted(offenders - UNROUTED)
    assert not unexpected, (
        f"{unexpected} launch a MACE command line without going through "
        f"tests/helpers.run_mace_train, so they ignore MACE_ENGINE and their "
        f"v1 result would be a legacy result. Route them, or add them to "
        f"UNROUTED with a reason."
    )


def test_the_bypass_list_has_no_stale_entries():
    """A routed file left on the list makes the list a lie."""
    offenders = {
        str(path.relative_to(REPO_ROOT))
        for path in black_box_files()
        if launches_a_command_line(path.read_text())
    }
    stale = sorted(UNROUTED - offenders)
    assert not stale, (
        f"{stale} no longer launch their own command line, so they can come "
        f"off the list"
    )


def test_the_shared_runner_knows_about_both_engines():
    """The single door, and that it is the only one that needs to know."""
    from tests.helpers import LAUNCHER_ENTRY_POINTS, NOT_MIGRATED, run_train

    assert run_train in LAUNCHER_ENTRY_POINTS
    assert "v1 engine" in NOT_MIGRATED

    source = (REPO_ROOT / "tests/helpers.py").read_text()
    assert source.count("MACE_ENGINE") >= 2, (
        "the engine is read in tests/helpers.py and nowhere else"
    )
    for path in black_box_files():
        # The launcher's own test is about the engine, so naming it there is
        # the subject rather than a second place choosing it.
        if path.name == "test_launcher_engine_parity.py":
            continue
        assert "MACE_ENGINE" not in path.read_text(), (
            f"{path.name} reads the engine itself; it is chosen in one place"
        )


def test_the_refusal_is_a_skip_whatever_the_caller_asked_for(tmp_path, monkeypatch):
    """A caller that captures output must skip too, not die on the refusal.

    The first version of the routing only handled callers that let output
    through. The ones that capture it took the other branch, met the launcher's
    refusal as a failed process, and failed a run they should have sat out. Six
    plotting cases went red that way, and they went red only in a full run,
    which is the kind of gap a two-file sample does not show.
    """
    import pytest as pytest_module

    from tests import helpers

    monkeypatch.setenv("MACE_ENGINE", "v1")
    if not helpers.launcher_available():
        pytest_module.skip("the launcher is not installed in this environment")

    for capture in (False, True):
        with pytest_module.raises(BaseException) as raised:
            helpers.run_mace_train(
                {"name": "unmigrated", "train_file": str(tmp_path / "none.xyz")},
                capture_output=capture,
                text=True,
                cwd=tmp_path,
            )
        # `pytest.skip` raises its own exception rather than returning, so
        # catching it is how a caller of the runner observes the skip.
        assert "Skipped" in type(raised.value).__name__ or NOT_MIGRATED in str(
            raised.value
        ), (
            f"with capture_output={capture} the runner raised "
            f"{type(raised.value).__name__} instead of skipping"
        )
