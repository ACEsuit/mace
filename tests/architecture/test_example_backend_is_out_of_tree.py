"""The example kernel backend needs no edit to MACE.

It is the executable proof that a backend in its own wheel plugs in through
its entry point and nothing else. The import contract in `.importlinter`
forbids the core packages from importing it; this forbids them from naming it
at all, in code, configuration or tests, since an entry point, a
special case keyed on its name or a test that installs it would each be a
coupling an import scan does not see.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = REPO_ROOT / "packages"
EXAMPLE = PACKAGES / "mace-backend-example"
CORE = ("mace-core", "mace-torch", "mace-jax", "mace-launcher")
NAMES = ("mace_backend_example", "mace-backend-example")


def test_no_core_package_names_the_example():
    offenders = []
    scanned = 0
    for package in CORE:
        for path in sorted((PACKAGES / package).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".toml", ".cfg", ".txt", ".md", ".json"}:
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(name in text for name in NAMES):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert scanned > 100, f"only {scanned} files scanned under the core packages"
    assert not offenders, (
        f"{offenders} name the example backend. It is reached through its entry "
        f"point only; anything else is an edit to MACE a third-party backend "
        f"would not get."
    )


def test_it_is_a_distribution_of_its_own_registered_by_entry_point():
    """Read as text, since the standard library parses TOML only from 3.11."""
    lines = [line.strip() for line in (EXAMPLE / "pyproject.toml").read_text().splitlines()]
    assert 'name = "mace-backend-example"' in lines
    section = lines.index('[project.entry-points."mace.kernel_backends.torch"]')
    entries = []
    for line in lines[section + 1 :]:
        if line.startswith("["):
            break
        if line and not line.startswith("#"):
            entries.append(line)
    assert entries == ['example = "mace_backend_example:ExampleBackend"']
    dependencies = lines[lines.index("dependencies = [") + 1 : lines.index("]")]
    assert not any("mace-torch" in line for line in dependencies), (
        "the backend depends on the contract, not on the implementation it "
        "plugs into"
    )
