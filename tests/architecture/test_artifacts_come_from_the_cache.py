"""Every artifact reaches v1 through the one resolver and its cache.

Two halves. Nothing under `packages/` fetches over the network except
`mace_core.artifacts`, so there is one download, atomic and guarded, and no
second writer into the cache. And nothing under `packages/` is a model file,
so no install, editable or wheel, short-circuits to a bundled copy: the frozen
tree's source checkouts did, while its wheels never shipped the files, which
made one call take two paths depending on how MACE was installed.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = REPO_ROOT / "packages"
RESOLVER = PACKAGES / "mace-core" / "src" / "mace_core" / "artifacts.py"

#: What fetches over the network from python's standard library and the usual
#: third-party clients.
NETWORK_MODULES = ("urllib.request", "http.client", "requests", "httpx", "urllib3")

#: What a model file is called, in either stack.
MODEL_SUFFIXES = (".model", ".pt", ".pth", ".ckpt")


def _sources() -> list[Path]:
    return sorted(PACKAGES.glob("*/src/**/*.py"))


def test_only_the_resolver_reaches_the_network():
    offenders = []
    for path in _sources():
        if path == RESOLVER:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name.startswith(NETWORK_MODULES) for name in names):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert _sources(), "no sources under packages/: the scan found nothing to read"
    assert not offenders, (
        f"{offenders} reach the network themselves. Artifacts are fetched by "
        f"`mace_core.artifacts.resolve_artifact`, the one download that is "
        f"atomic, refuses HTML and writes into the cache."
    )


def test_no_model_file_is_part_of_any_package():
    bundled = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(PACKAGES.rglob("*"))
        if path.is_file() and path.suffix in MODEL_SUFFIXES
    ]
    assert not bundled, (
        f"{bundled} are model files inside the v1 packages. Every artifact is "
        f"resolved through the cache, so none is bundled."
    )
