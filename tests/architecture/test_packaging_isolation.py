"""The v1 scaffold must not leak into the legacy wheel.

The legacy build is `setuptools.build_meta` with an unscoped `packages = find:`
(`setup.cfg`), so it discovers whatever `setuptools.find_packages()` returns
from the repository root. Nothing under `packages/` is discovered today, and
the reason is worth pinning: `find_packages` walks only path components that
are Python identifiers, and every v1 package directory is hyphenated. Rename
one to `packages/mace_core/` and it silently ships inside the `mace-torch`
wheel, with no error anywhere.
"""

from pathlib import Path

import setuptools

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_DIR = REPO_ROOT / "packages"

# Import names of the v1 packages, which are what a leak would look like.
V1_IMPORT_NAMES = {"mace_core", "mace_torch", "mace_jax", "mace_launcher"}


def test_scaffold_directories_are_not_python_identifiers():
    """Hyphenated names are what keeps find_packages out; assert it, do not assume it."""
    offenders = [
        entry.name
        for entry in sorted(PACKAGES_DIR.iterdir())
        if entry.is_dir() and entry.name.isidentifier()
    ]
    assert not offenders, (
        f"package directories {offenders} are Python identifiers, so "
        f"setuptools.find_packages() will walk into them and ship their "
        f"contents inside the legacy mace-torch wheel. Use a hyphen."
    )


def test_legacy_wheel_does_not_discover_the_v1_packages():
    """The legacy distribution ships `mace` and its subpackages, never a v1 one."""
    discovered = set(setuptools.find_packages(where=str(REPO_ROOT)))

    assert "mace" in discovered, (
        "the legacy package itself is no longer discovered from the repository "
        "root, so this test can no longer prove anything about what the wheel ships"
    )

    top_level = {name.split(".")[0] for name in discovered}
    leaked = top_level & (V1_IMPORT_NAMES | {"packages"})
    assert not leaked, (
        f"{sorted(leaked)} would be packaged into the legacy mace-torch wheel; "
        f"the v1 stack is installed from packages/*/pyproject.toml, never from "
        f"the legacy build"
    )
