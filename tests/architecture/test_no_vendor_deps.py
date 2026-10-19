"""`mace_core` derives the equivariant mathematics rather than importing it.

The import contracts already forbid torch, jax and e3nn. This adds
`cuequivariance`, and three checks the contracts cannot make:

- no ``import`` or ``from ... import`` statement anywhere in a module, at any
  depth including inside a function, names one of those libraries;
- every call to ``__import__``, ``importlib.import_module`` or
  ``importlib.util.find_spec``, however it is spelled or spaced, names its
  module as a string literal, and that literal is not one of them. A module
  name held in a variable is refused outright, because a static scan cannot
  see what it will hold;
- importing ``mace_core`` in a fresh interpreter loads none of them.

What this does not catch is a module reached in some other way at run time,
such as an entry point's ``load()`` or an ``exec`` of built source. The fresh
interpreter check covers whatever runs at import time, and nothing later.

It matters because of what the dependency did rather than because of the
dependency itself. On the frozen tree the reduced basis exists only when
`cuequivariance` is installed, so the same hyperparameters train a 29-parameter
network on one host and an 86-parameter one on another, with no warning. An
import that creeps back in is how that returns.
"""

from __future__ import annotations

import ast
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


#: The calls that import, or look up, a module named by a string.
DYNAMIC_IMPORTERS = ("__import__", "import_module", "find_spec")


def _called_name(call: ast.Call) -> str:
    """``f`` for ``f(...)``, ``attr`` for ``x.y.attr(...)``, else empty."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


@pytest.mark.parametrize(
    "module", core_modules(), ids=lambda p: str(p.relative_to(CORE_SOURCE))
)
def test_no_module_reaches_a_framework_through_importlib(module):
    """The hole a static import check leaves open, and the one the frozen tree
    actually used: `cg.py` decides on `CUET_AVAILABLE`, set by a guarded
    import."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    where = module.relative_to(REPO_ROOT)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = _called_name(node)
        if called not in DYNAMIC_IMPORTERS:
            continue
        named = node.args[0] if node.args else None
        for keyword in node.keywords:
            if keyword.arg == "name":
                named = keyword.value
        assert isinstance(named, ast.Constant) and isinstance(named.value, str), (
            f"{where} line {node.lineno} calls {called} with a module name that "
            f"is not a string literal, so this scan cannot tell what it imports. "
            f"Name the module literally."
        )
        assert named.value.split(".")[0] not in FORBIDDEN, (
            f"{where} line {node.lineno} reaches {named.value!r} through "
            f"{called}. mace_core derives this mathematics instead."
        )


def test_importing_mace_core_pulls_in_no_framework():
    """Run in a fresh interpreter, and that is not fussiness.

    Asserting this in-process passes or fails on what else the session happened
    to import, and this suite runs beside an oracle test that imports
    `cuequivariance` on purpose. The first version of this test failed for
    exactly that reason, which is the argument for the subprocess. One probe
    checks every library and names all that leaked.
    """
    probe = (
        "import sys, mace_core, mace_core.clebsch_gordan\n"
        f"leaked = [name for name in {FORBIDDEN!r} if name in sys.modules]\n"
        "assert not leaked, f'importing mace_core pulled in {leaked}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
