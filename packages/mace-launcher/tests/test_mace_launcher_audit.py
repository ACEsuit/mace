"""The runtime guard against a v1 module importing the frozen legacy package.

Two things shape how these are written. The hook is process-wide and cannot be
removed once installed, so each scenario runs in a subprocess rather than
arming the guard for the rest of the session. And CPython raises the `import`
audit event from inside `_find_and_load`, which a cached module never reaches,
so the packages have to be real files on disk: pre-seeding `sys.modules` would
produce a test that passes because no event ever fired.
"""

# pylint: disable=redefined-outer-name
# Requesting a fixture shadows its name by construction, which is how pytest
# works and not a defect. The repository excludes tests/** from pylint for this
# reason; packages/**/tests/ is not covered by that exclusion yet, so the file
# carries the pragma itself rather than the shared config being edited.

import subprocess
import sys
import textwrap

import pytest

PREAMBLE = """
import sys
sys.path.insert(0, {root!r})
from mace_launcher import audit
audit.install()
"""


@pytest.fixture(scope="module")
def fake_stacks(tmp_path_factory):
    """A minimal importable `mace`, `mace_core` and `mace_torch` on disk."""
    root = tmp_path_factory.mktemp("stacks")
    for name in ("mace", "mace_core", "mace_torch"):
        package = root / name
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "modules.py").write_text("value = 1\n")
    return root


def run(fake_stacks, body: str, module_name: str, filename: str):
    """Execute `body` in a subprocess, under the identity of `module_name`."""
    script = PREAMBLE.format(root=str(fake_stacks)) + textwrap.dedent(f"""
        code = compile({textwrap.dedent(body)!r}, {filename!r}, "exec")
        exec(code, {{"__name__": {module_name!r}, "__file__": {filename!r}}})
        print("allowed")
        """)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )


CROSS = "import importlib; importlib.import_module('mace.modules')"


def test_a_dynamic_cross_import_from_the_v1_stack_fails(fake_stacks):
    """importlib is the form static analysis cannot see, so it is the one tested."""
    result = run(fake_stacks, CROSS, "mace_core.sneaky", "/fake/mace_core/sneaky.py")
    assert result.returncode != 0, result.stdout
    assert "LegacyReachIn" in result.stderr
    assert "mace_core.sneaky" in result.stderr


def test_a_plain_import_statement_from_the_v1_stack_fails_too(fake_stacks):
    result = run(fake_stacks, "import mace.modules", "mace_torch.port", "/fake/x.py")
    assert result.returncode != 0, result.stdout
    assert "LegacyReachIn" in result.stderr


def test_the_launcher_itself_may_import_the_legacy_package(fake_stacks):
    """It is one of the two allowlisted crossers; dispatch is its whole job."""
    result = run(
        fake_stacks, CROSS, "mace_launcher.dispatch", "/fake/mace_launcher/d.py"
    )
    assert result.returncode == 0, result.stderr
    assert "allowed" in result.stdout


def test_the_parity_harness_may_import_both(fake_stacks):
    result = run(fake_stacks, CROSS, "test_energy", "/repo/tests/parity/test_energy.py")
    assert result.returncode == 0, result.stderr
    assert "allowed" in result.stdout


def test_an_ordinary_import_of_something_else_is_not_touched(fake_stacks):
    result = run(
        fake_stacks,
        "import importlib; importlib.import_module('json')",
        "mace_core.fine",
        "/fake/mace_core/fine.py",
    )
    assert result.returncode == 0, result.stderr
    assert "allowed" in result.stdout


def test_a_module_merely_named_like_the_legacy_package_is_not_a_crossing(fake_stacks):
    """`mace_torch` starts with `mace`; only `mace` and `mace.*` are the oracle."""
    result = run(
        fake_stacks,
        "import importlib; importlib.import_module('mace_torch.modules')",
        "mace_core.fine",
        "/fake/mace_core/fine.py",
    )
    assert result.returncode == 0, result.stderr
    assert "allowed" in result.stdout


def test_code_outside_both_stacks_is_not_policed(fake_stacks):
    """A user script importing legacy is ordinary use, not a reach-in."""
    result = run(fake_stacks, CROSS, "__main__", "/home/someone/script.py")
    assert result.returncode == 0, result.stderr
    assert "allowed" in result.stdout
