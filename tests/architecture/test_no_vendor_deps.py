"""`mace_core` derives the equivariant mathematics rather than importing it.

The import contracts already forbid torch, jax and e3nn. This adds
`cuequivariance`, and it adds a check the contracts cannot make: that no module
under `mace_core` imports either of the two libraries the reduced basis used to
come from, **at any depth and by any spelling**, including inside a function and
including through `importlib`.

It matters because of what the dependency did rather than because of the
dependency itself. On the frozen tree the reduced basis exists only when
`cuequivariance` is installed, so the same hyperparameters train a 29-parameter
network on one host and an 86-parameter one on another, with no warning. An
import that creeps back in is how that returns.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_SOURCE = REPO_ROOT / "packages" / "mace-core" / "src" / "mace_core"

FORBIDDEN = ("e3nn", "cuequivariance", "cuequivariance_torch", "torch", "jax")


def core_modules() -> list[Path]:
    return sorted(CORE_SOURCE.rglob("*.py"))


def test_the_package_has_modules_to_check():
    """An empty scan reports perfect compliance, so the scan says how much it
    looked at."""
    assert len(core_modules()) >= 5


@pytest.mark.parametrize(
    "module", core_modules(), ids=lambda p: str(p.relative_to(CORE_SOURCE))
)
def test_no_module_imports_a_framework_or_a_kernel_library(module):
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            assert root not in FORBIDDEN, (
                f"{module.relative_to(REPO_ROOT)} line {node.lineno} imports "
                f"{name!r}. mace_core derives this mathematics instead; see "
                f"mace_core.clebsch_gordan for why the dependency was the "
                f"defect and not the solution."
            )


@pytest.mark.parametrize(
    "module", core_modules(), ids=lambda p: str(p.relative_to(CORE_SOURCE))
)
def test_no_module_reaches_a_framework_through_importlib(module):
    """The hole a static import check leaves open, and the one the frozen tree
    actually used: `cg.py` decides on `CUET_AVAILABLE`, set by a guarded
    import."""
    source = module.read_text(encoding="utf-8")
    for name in FORBIDDEN:
        assert f'import_module("{name}' not in source
        assert f"import_module('{name}" not in source
        assert f'find_spec("{name}' not in source
        assert f"find_spec('{name}" not in source


@pytest.mark.parametrize("library", ["e3nn", "cuequivariance", "torch", "jax"])
def test_importing_mace_core_pulls_in_no_framework(library):
    """Run in a fresh interpreter, and that is not fussiness.

    Asserting this in-process passes or fails on what else the session happened
    to import, and this suite runs beside an oracle test that imports
    `cuequivariance` on purpose. The first version of this test failed for
    exactly that reason, which is the argument for the subprocess.
    """
    probe = (
        "import sys, mace_core, mace_core.clebsch_gordan\n"
        f"assert {library!r} not in sys.modules, "
        f"'importing mace_core pulled in {library}'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_the_basis_does_not_change_with_what_is_installed():
    """The defect, stated as a test. `cuequivariance` is importable in this
    environment or it is not; either way the count is the same number."""
    from mace_core.clebsch_gordan.reduced_basis import path_count

    installed = importlib.util.find_spec("cuequivariance") is not None
    counts = [path_count("0e+1o+2e+3o", nu, "0e+1o") for nu in (1, 2, 3)]
    assert sum(counts) == 29, (
        f"the reduced basis gave {sum(counts)} paths with cuequivariance "
        f"{'installed' if installed else 'absent'}. It must give 29 either way."
    )
