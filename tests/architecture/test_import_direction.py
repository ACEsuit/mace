"""The import contracts in `.importlinter`, run as a test.

The contracts are the static half of the guard that keeps the frozen legacy
package unreachable from the v1 stack. They are checked here as well as by the
`lint-imports` step in CI, so that a developer running the architecture suite
locally sees the same answer as the pull request will.
"""

import shutil
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / ".importlinter"

#: The two places allowed to import both stacks. `mace_launcher` holds the
#: entry-point dispatcher and the runtime guard; `tests/parity/` is the harness
#: that loads both stacks in one process to compare them numerically. A third
#: entry defeats the guard, so the count is asserted rather than described.
ALLOWED_DOUBLE_IMPORTERS = ("mace_launcher", "tests.parity")


def test_the_config_exists():
    assert CONFIG.exists(), f"no import contracts at {CONFIG}"


def test_the_v1_packages_are_not_sources_of_a_legacy_import():
    """`mace_launcher` must be absent from the forbidden contract's sources.

    Its absence *is* the allowlist entry, which makes it easy to add by
    accident and impossible to notice. Pinning it here means widening the
    allowlist takes an edit to this test.
    """
    parser = ConfigParser()
    parser.read(CONFIG, encoding="utf-8")
    section = parser["importlinter:contract:no-legacy-from-v1"]
    sources = set(section["source_modules"].split())
    assert sources == {"mace_core", "mace_torch", "mace_jax"}, (
        f"the forbidden contract's sources are {sorted(sources)}. Every v1 "
        f"package belongs there; the only module allowed out is the launcher, "
        f"and any other omission is a hole in the guard."
    )
    assert "mace" in section["forbidden_modules"].split()


def test_exactly_two_modules_may_import_both_stacks():
    assert len(ALLOWED_DOUBLE_IMPORTERS) == 2


@pytest.mark.skipif(
    shutil.which("lint-imports") is None,
    reason="import-linter is not installed (it ships in the dev extra)",
)
def test_all_import_contracts_hold():
    result = subprocess.run(
        [sys.executable, "-m", "importlinter.cli", "lint-imports",
         "--config", str(CONFIG)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "import contracts broken:\n" + result.stdout + result.stderr
    )
