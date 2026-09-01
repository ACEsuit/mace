"""No file under `packages/` may name a legacy symbol.

The import guard stops the v1 stack importing the frozen legacy package. This
catches the step before that: naming a legacy class or registry in new code,
with or without an import. v1 reproduces legacy *behaviour*; a legacy symbol
appearing in a v1 file is the tell of a structural port, and a port is how the
oracle stops being independent of the thing it judges.

Matching is done over the parsed syntax tree, never over the file's text. The
denylist contains `MACE`, which is also the project's name and appears in
almost every docstring and comment in the tree, so a textual search would flag
the whole repository and be switched off within a week.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = REPO_ROOT / "packages"
LEGACY_MODELS = REPO_ROOT / "mace" / "modules" / "models.py"
LEGACY_EXTENSIONS = REPO_ROOT / "mace" / "modules" / "extensions.py"
LEGACY_REGISTRIES = REPO_ROOT / "mace" / "modules" / "__init__.py"

# Legacy symbols v1 does not carry over. Adding or removing an entry is a
# reviewed edit to this list, deliberately, rather than an open-ended read of
# the legacy public surface: the point is to name what must not be ported, not
# to mirror whatever legacy happens to export.
LEGACY_MODEL_CLASSES = frozenset(
    {
        "MACE",
        "ScaleShiftMACE",
        "AtomicDipolesMACE",
        "AtomicDielectricMACE",
        "EnergyDipolesMACE",
        "MACELES",
        "PolarMACE",
        "MagneticMACE",
        "MagneticScaleShiftMACE",
        "MagneticSCFMACE",
        "TimeReversalSymmetrizedMACE",
    }
)
LEGACY_REGISTRY_NAMES = frozenset(
    {"interaction_classes", "readout_classes", "scaling_classes", "gate_dict"}
)
LEGACY_DATA_TYPES = frozenset({"AtomicData", "AtomicDataDict"})

DENYLIST = LEGACY_MODEL_CLASSES | LEGACY_REGISTRY_NAMES | LEGACY_DATA_TYPES

# Names v1 reuses for its own ported code. They are legacy names too, and the
# whole point is that v1 is allowed to reimplement the concept under the same
# name, so they must never reach the denylist.
DELIBERATELY_REUSED = frozenset(
    {
        "BesselBasis",
        "PolynomialCutoff",
        "ZBLBasis",
        "RadialEmbeddingBlock",
        "LinearNodeEmbeddingBlock",
        "AtomicNumberTable",
        "Configuration",
        "KeySpecification",
        "DefaultKeys",
    }
)

# Blocks that happen to live in extensions.py. They are not model classes, and
# nothing about them is forbidden to v1.
NOT_MODELS = frozenset({"SHModule", "ChebyshevBasisGeneral"})


def names_used(source: str) -> set[str]:
    """Every identifier the code actually refers to, docstrings excluded.

    Attribute access and import targets count, so `mace.modules.ScaleShiftMACE`
    and `from x import ScaleShiftMACE as Y` are both caught. Non-docstring
    string constants count too, because `getattr(module, "ScaleShiftMACE")` is
    a reference that carries no identifier at all.

    Definitions count as well. A file that declares `class ScaleShiftMACE` and
    never mentions the name again refers to nothing, so collecting only
    references would miss the wholesale copy this check exists to catch.
    """
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)

    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            used.add(node.name)
        elif isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.alias):
            used.add(node.name.rsplit(".", 1)[-1])
            if node.asname:
                used.add(node.asname)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                used.update(part for part in node.value.replace(".", " ").split())
    return used


def package_sources() -> list[Path]:
    return sorted(PACKAGES.rglob("*.py"))


def legacy_top_level_classes() -> set[str]:
    found: set[str] = set()
    for path in (LEGACY_MODELS, LEGACY_EXTENSIONS):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found.update(n.name for n in tree.body if isinstance(n, ast.ClassDef))
    return found - NOT_MODELS


def test_no_packages_file_names_a_legacy_symbol():
    offenders: list[str] = []
    for path in package_sources():
        hits = names_used(path.read_text(encoding="utf-8")) & DENYLIST
        if hits:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {sorted(hits)}")
    assert not offenders, (
        "v1 files naming legacy symbols:\n  "
        + "\n  ".join(offenders)
        + "\n\nReproduce the behaviour, not the structure. If a name here is one "
        "v1 legitimately reuses for its own implementation, add it to "
        "DELIBERATELY_REUSED and remove it from the denylist, in a reviewed edit."
    )


def test_the_denylist_covers_every_legacy_model_class():
    """A model class added to legacy must not be able to slip past this file."""
    actual = legacy_top_level_classes()
    missing = actual - LEGACY_MODEL_CLASSES
    assert not missing, (
        f"legacy model classes with no denylist entry: {sorted(missing)}. "
        f"A class that is not on the list can be ported into packages/ with "
        f"nothing failing."
    )
    stale = LEGACY_MODEL_CLASSES - actual
    assert not stale, (
        f"denylist entries that no longer exist in legacy: {sorted(stale)}. "
        f"A stale entry forbids a name v1 is now free to use."
    )
    assert len(LEGACY_MODEL_CLASSES) == 11


def test_the_denylist_covers_every_legacy_registry():
    tree = ast.parse(LEGACY_REGISTRIES.read_text(encoding="utf-8"))
    declared = set()
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
        name = getattr(target, "id", None)
        if name and name.endswith(("_classes", "_dict")):
            declared.add(name)
    assert declared == LEGACY_REGISTRY_NAMES, (
        f"registries in legacy: {sorted(declared)}; on the denylist: "
        f"{sorted(LEGACY_REGISTRY_NAMES)}"
    )


def test_names_v1_reuses_are_not_denied():
    """v1 reimplements these concepts under the same name, by design."""
    assert not (DELIBERATELY_REUSED & DENYLIST)


@pytest.mark.parametrize(
    "snippet",
    [
        "from mace.modules import ScaleShiftMACE",
        "import mace\nmodel = mace.modules.ScaleShiftMACE()",
        "from x import ScaleShiftMACE as Backbone",
        "cls = getattr(module, 'ScaleShiftMACE')",
        "class ScaleShiftMACE(Module):\n    pass",
        "class MACELES(ScaleShiftMACE):\n    pass",
        "def interaction_classes():\n    return {}",
        "from mace.modules import interaction_classes",
        "batch = AtomicData.from_config(config)",
    ],
)
def test_the_check_fires_on_a_structural_port(snippet):
    """Each spelling a port would actually use, including the string form."""
    assert names_used(snippet) & DENYLIST


@pytest.mark.parametrize(
    "snippet",
    [
        '"""The PyTorch stack for MACE v1."""',
        "# MACE reproduces the behaviour of ScaleShiftMACE\nx = 1",
        "from mace_core.radial import BesselBasis, PolynomialCutoff",
        "table = AtomicNumberTable([1, 8])",
    ],
)
def test_the_check_does_not_fire_on_prose_or_reused_names(snippet):
    """A docstring, a comment and the names v1 keeps must all pass."""
    assert not (names_used(snippet) & DENYLIST)
