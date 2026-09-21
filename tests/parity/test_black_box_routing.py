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

Five files do that today. They are named here rather than left to be
discovered, so the gap is bounded and a sixth fails this test.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BLACK_BOX = ("tests/workflows", "tests/integrations")

#: Files that launch a MACE command line without going through the shared
#: runner, and therefore ignore `MACE_ENGINE`. Each is a black-box test that
#: will run legacy whatever engine is asked for, so its v1 result means
#: nothing until it is routed.
#:
#: They are listed rather than fixed here because routing them is an edit to
#: the tests, and the wiring ticket's line is that the tests are re-run rather
#: than rewritten. Whoever routes them deletes the entry.
UNROUTED = {
    "tests/workflows/test_distributed.py",
    "tests/workflows/test_embedding_train.py",
    "tests/workflows/test_mdp_finetune.py",
    "tests/workflows/test_multifiles.py",
    "tests/workflows/test_run_train_dipole_polar.py",
    "tests/workflows/test_small_training_set.py",
}


def launches_a_command_line(source: str) -> bool:
    """Whether a file starts a subprocess that runs a MACE CLI script.

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
        if isinstance(node, ast.Name) and node.id in {"run_train", "preprocess_data"}:
            names_a_script = True
        # And by its path, which is how the multi-file test spells it.
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.endswith(".py")
            and node.value.startswith(("run_train", "preprocess_data"))
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
