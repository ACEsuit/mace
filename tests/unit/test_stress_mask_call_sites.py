"""The stress mask is shared, and this is what keeps it shared.

``cell_volume_and_mask`` decides per graph whether a virial has a volume to be
normalized by. That decision is only worth anything if every model reaches it,
and a model reaches it by passing ``pbc`` alongside ``cell``. Miss one call
site and that model reports the padding box again, which is invisible until
somebody compares a molecule's stress against zero.

It is exactly the kind of property that decays one call site at a time. The
call site missed the first time round was ``PolarMACE``'s atomic stresses,
which reaches the helper through a function-local ``import ... as _gav``, so
grepping for the function's own name did not see it while the total stress
right above it was already masked.

So it is a test rather than a comment. The check is crude on purpose: names
are read out of the source with ``ast`` and never resolved, which is enough to
notice a new call site and cheap enough to always run.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Helpers that turn a virial into a stress. Any call to one of these that
#: says which cell to divide by must also say which graphs have a volume.
MASK_CONSUMERS = frozenset(
    {
        "get_outputs",
        "get_atomic_virials_stresses",
        "compute_forces_virials",
        "compute_forces_virials_magforces",
    }
)


def _local_names(tree: ast.Module) -> dict[str, str]:
    """Local name -> helper name, for every import anywhere in the file.

    Walks the whole tree rather than the module body, so a function-local
    import (and an ``as`` alias) is picked up too.
    """
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for imported in node.names:
                if imported.name in MASK_CONSUMERS:
                    names[imported.asname or imported.name] = imported.name
    return names


def _call_sites():
    for path in sorted((REPO_ROOT / "mace").rglob("*.py")):
        if "torch_geometric" in str(path):  # vendored
            continue
        tree = ast.parse(path.read_text())
        names = _local_names(tree)
        if not names:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id not in names:
                continue
            keywords = {kw.arg for kw in node.keywords}
            yield (
                path.relative_to(REPO_ROOT),
                node.lineno,
                names[node.func.id],
                keywords,
            )


def test_every_stress_call_site_passes_pbc():
    sites = list(_call_sites())
    # A rename or a moved import would empty this and pass silently.
    assert len(sites) >= 10, f"expected the known call sites, found {len(sites)}"

    unmasked = [
        f"{path}:{line} calls {helper} with cell= but no pbc="
        for path, line, helper, keywords in sites
        if "cell" in keywords and "pbc" not in keywords
    ]
    assert not unmasked, "\n".join(unmasked)
