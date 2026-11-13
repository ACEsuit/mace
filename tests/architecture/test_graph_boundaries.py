"""Where the graph dict is checked, and what a model forward may read of it.

Three rules of the flat graph contract that only a whole-tree scan can hold:

* ``torch_geometric`` is imported nowhere under ``packages/``, vendored or not;
* the batch checks, ``check_layout`` and ``check_batch_invariants``, run at
  boundaries. No module that defines a ``forward`` names one, and the modules
  that do name them are listed here, so a new caller is a reviewed edit;
* the host values of a batch live in ``GraphInfo``, read outside compiled
  regions only, so no module that defines a ``forward`` names ``GraphInfo``,
  ``PaddingInfo`` or ``structure_sizes``. A compiled frame would guard on
  their values and recompile per batch.

The scan is over syntax trees, so docstrings do not count.
"""

import ast
from pathlib import Path

from tests.architecture.test_no_legacy_leak import names_used

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = REPO_ROOT / "packages"

BOUNDARY_ONLY = frozenset({"check_layout", "check_batch_invariants"})
HOST_VALUES = frozenset({"GraphInfo", "PaddingInfo", "structure_sizes"})

#: The modules allowed to name a batch check: the graph contract itself, and
#: the collation and padding that produce a batch.
CHECKING_MODULES = frozenset(
    {
        "packages/mace-core/src/mace_core/graph/__init__.py",
        "packages/mace-core/src/mace_core/graph/collate.py",
        "packages/mace-core/src/mace_core/graph/validation.py",
        "packages/mace-torch/src/mace_torch/graph/collate.py",
        "packages/mace-torch/src/mace_torch/graph/padding.py",
    }
)


def _sources() -> list[Path]:
    return sorted(p for p in PACKAGES.glob("*/src/**/*.py"))


def _defines_forward(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == "forward"
        for node in ast.walk(tree)
    )


def _imported_modules(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_the_scan_has_modules_and_forwards_to_check():
    """An empty scan reports compliance, so it says what it looked at."""
    trees = [ast.parse(p.read_text(encoding="utf-8")) for p in _sources()]
    assert len(trees) > 20
    assert sum(_defines_forward(tree) for tree in trees) >= 3


def test_torch_geometric_is_imported_nowhere_in_packages():
    offenders = []
    for path in sorted(PACKAGES.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = _imported_modules(tree)
        if any("torch_geometric" in module.split(".") for module in imported) or (
            "torch_geometric" in names_used(source)
        ):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"torch_geometric is named in {offenders}"


def test_no_forward_reaches_a_batch_check():
    offenders = []
    for path in _sources():
        source = path.read_text(encoding="utf-8")
        if _defines_forward(ast.parse(source)) and names_used(source) & BOUNDARY_ONLY:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"{offenders} define a forward and name a batch check. Checks run at "
        f"boundaries, never inside a model call."
    )


def test_only_the_listed_boundaries_name_a_batch_check():
    naming = {
        str(path.relative_to(REPO_ROOT))
        for path in _sources()
        if names_used(path.read_text(encoding="utf-8"))
        & {"check_layout", "check_batch_invariants"}
    }
    assert naming == CHECKING_MODULES, (
        f"a batch check is named by {sorted(naming - CHECKING_MODULES)} as "
        f"well. If that module is a boundary, add it to CHECKING_MODULES."
    )


def test_no_forward_reads_a_host_value():
    offenders = []
    for path in _sources():
        source = path.read_text(encoding="utf-8")
        hits = names_used(source) & HOST_VALUES
        if _defines_forward(ast.parse(source)) and hits:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {sorted(hits)}")
    assert not offenders, (
        f"{offenders}: a forward reads a host value. Branch on GraphInfo outside "
        f"the compiled region."
    )
