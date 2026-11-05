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
V1_IMPORT_NAMES = {
    "mace_core",
    "mace_torch",
    "mace_jax",
    "mace_launcher",
    "mace_backend_example",
}


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


def test_repository_wheel_keeps_the_test_and_dual_toolchain_extras(tmp_path):
    import shutil
    import subprocess
    import zipfile
    from email.parser import Parser

    import pytest
    from packaging.requirements import Requirement

    if shutil.which("uv") is None:
        pytest.skip("building the wheel needs uv")
    completed = subprocess.run(
        ["uv", "build", "--wheel", "--quiet", str(REPO_ROOT), "-o", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    wheel, = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        name, = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        metadata = Parser().parsestr(archive.read(name).decode())
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    extras = {}
    for extra in ("test", "dev"):
        assert extra in metadata.get_all("Provides-Extra", [])
        extras[extra] = {item.name: item for item in requirements
                         if item.marker is not None and item.marker.evaluate({"extra": extra})
                         and not item.marker.evaluate({"extra": ""})}
    tests = {"hypothesis", "pytest", "pytest-benchmark", "pytest-cov", "pytest-split", "pytest-timeout", "pytest-xdist"}
    assert set(extras["test"]) == tests
    assert tests | {"black", "isort", "pylint", "mypy", "pre-commit", "import-linter", "ruff", "ty", "prek"} <= set(extras["dev"])
    assert "9.2" not in extras["test"]["pytest"].specifier
