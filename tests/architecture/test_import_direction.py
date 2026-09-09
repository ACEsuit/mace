"""The import contracts in `.importlinter`, run as a test.

The contracts are the static half of the guard that keeps the frozen legacy
package unreachable from the v1 stack. They are checked here as well as by the
`lint-imports` step in CI, so that a developer running the architecture suite
locally sees the same answer as the pull request will.
"""

import importlib.util
import shutil
import subprocess
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


#: Every package the config declares as a root. import-linter resolves each on
#: the filesystem, so one that is not installed is reported as a broken
#: contract rather than as a missing install, which reads as a real violation.
ROOT_PACKAGES = ("mace", "mace_core", "mace_torch", "mace_jax", "mace_launcher")


def _missing_roots() -> list[str]:
    return [name for name in ROOT_PACKAGES if importlib.util.find_spec(name) is None]


@pytest.mark.skipif(
    shutil.which("lint-imports") is None,
    reason="import-linter is not installed (it ships in the dev extra)",
)
def test_all_import_contracts_hold():
    """Run the contracts through the console script, not `python -m`.

    `importlinter.cli` has no `__main__` guard, so `python -m importlinter.cli`
    imports the module, does nothing and exits 0. A test asserting only on the
    return code then passes without checking a single contract, which is worse
    than not having it.
    """
    missing = _missing_roots()
    if missing:
        pytest.skip(
            f"the import contracts need every root package installed; missing "
            f"{missing}. The ci-core `architecture` job installs both trees; "
            f"a job that only installs one cannot run this."
        )

    result = subprocess.run(
        [shutil.which("lint-imports"), "--config", str(CONFIG)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    # Assert the work happened, not just that nothing failed: the silent no-op
    # above also returned 0.
    assert "Contracts:" in output, (
        "lint-imports produced no contract report, so nothing was checked:\n"
        + output
    )
    assert result.returncode == 0, "import contracts broken:\n" + output


# ---------------------------------------------------------------------------
# The fitness suite's entry to this contract
# ---------------------------------------------------------------------------


def test_import_direction_one_way():
    """The named fitness function, delegating to the guard above.

    The fitness suite lists five invariants about the target shape and this is
    one of them, so it is nameable and greppable from that list. It owns no
    logic: the guard is `.importlinter` plus the `lint-imports` step, and a
    second implementation of the same check would be a second thing to keep
    correct. What it does own is the invariant's *statement* -- packages/ never
    imports mace/, with exactly two allowlisted double importers -- and the
    fact that the check is reachable at all, which is what fails when the
    config is deleted or the console script disappears from the environment.
    """
    assert CONFIG.exists(), (
        "there are no import contracts, so the direction is unguarded whatever "
        "the rest of this file asserts"
    )
    assert len(ALLOWED_DOUBLE_IMPORTERS) == 2, ALLOWED_DOUBLE_IMPORTERS

    if shutil.which("lint-imports") is None or _missing_roots():
        pytest.skip(
            "the contracts need import-linter and every root package "
            "installed; test_all_import_contracts_hold above is the one that "
            "runs them, and the ci-core `architecture` job is where it does"
        )
    test_all_import_contracts_hold()
