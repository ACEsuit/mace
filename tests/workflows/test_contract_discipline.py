"""The contract suites stay black-box, and this is what enforces it.

The whole value of the end-to-end contracts is that the live parity run can
re-execute them against a different engine without editing them. That
property survives exactly as long as nobody reaches into ``mace`` for an
assertion, and it is the kind of property that decays one convenient import
at a time -- each of which looks harmless and none of which fails anything.

So it is a test rather than a paragraph. The check is deliberately crude
(imports are read out of the source with ``ast``, not resolved), because a
crude check that runs is worth more than an exact one that does not: an
import this refuses can always be moved into a regeneration path, which is
where the two exceptions below live.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CONTRACT_FILES = (
    "tests/workflows/test_cli_contracts.py",
    "tests/workflows/test_finetuning_contracts.py",
    "tests/workflows/conftest.py",
    "tests/integrations/lammps/test_export_golden.py",
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The only pieces of the package a contract test may name, each with the
#: reason it is not reachable any other way.
ALLOWED = {
    # The ase calculator is itself one of the contracts. There is no console
    # script for it, and the graph-padding arguments exist nowhere else, so
    # "drive it through a subprocess" is not available.
    "mace.calculators": "the ase calculator is one of the contracts under test",
}

#: Modules that are the implementation. Naming one of these in a contract
#: test is what the rule forbids.
FORBIDDEN_PREFIXES = ("mace.modules", "mace.data", "mace.tools", "mace.cli")


def _imported_modules(path: Path):
    """Every module named by an ``import`` in ``path``, at any depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.module, node.lineno))
    return found


@pytest.mark.parametrize("relative", CONTRACT_FILES)
def test_a_contract_test_does_not_import_the_implementation(relative):
    path = REPO_ROOT / relative
    assert path.exists(), relative

    offending = []
    for module, lineno in _imported_modules(path):
        if not (module == "mace" or module.startswith("mace.")):
            continue
        if any(module.startswith(allowed) for allowed in ALLOWED):
            continue
        offending.append(f"{relative}:{lineno} imports {module}")

    assert not offending, (
        "a contract test reached into the package it is supposed to be "
        "testing from the outside:\n  "
        + "\n  ".join(offending)
        + "\n\nThese tests are re-run verbatim against a different engine, so "
        "an assertion that depends on this repository's internals is an "
        "assertion that cannot travel. Drive the console entry point, read "
        "the artefact it wrote, or -- if the input genuinely cannot be built "
        "without the package -- commit the input and replay it, as "
        "tests/integrations/lammps/export_golden.py does. Allowed: "
        + ", ".join(f"{name} ({why})" for name, why in ALLOWED.items())
    )


@pytest.mark.parametrize("relative", CONTRACT_FILES)
def test_a_contract_test_names_no_implementation_module_in_a_string(relative):
    """The second-order version: an import hidden behind ``importlib``.

    Cheap to check and worth checking, because the obvious way around the
    test above is to spell the module name as data.
    """
    source = (REPO_ROOT / relative).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(relative))
    offending = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if any(node.value.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            offending.append(f"{relative}:{node.lineno}: {node.value!r}")
    assert not offending, (
        "an implementation module is named as a string in a contract test, "
        "which is how an import gets past the sibling check:\n  "
        + "\n  ".join(offending)
    )
