"""The regeneration driver must stay a driver.

These are the tests that keep ``regenerate.py`` free of per-family knowledge.
A family of goldens is added by dropping a module into ``targets/``, and the
value of that is entirely in what it prevents: two families added on two
branches merging without touching a line in common. The moment the driver
grows an enumeration again, or a family documents itself in the shared
README, that property is gone and nothing else would notice.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.golden import targets
from tests.golden.paths import GOLDEN_ROOT, REPO_ROOT

TARGETS_DIR = GOLDEN_ROOT / "targets"
DOCS_DIR = GOLDEN_ROOT / "docs"


def test_every_target_module_declares_the_four_names_the_driver_reads():
    # discover() raises on a module that does not; calling it is the check.
    discovered = targets.discover()
    assert discovered, "no targets were discovered at all"
    for name, target in discovered.items():
        assert target.help.strip(), f"{name} has an empty help line"
        assert callable(target.run)


def test_all_runs_the_families_in_dependency_order():
    order = [name for name, t in targets.discover().items() if t.in_all]
    # Fixtures are inputs to the anchors, and the anchors are inputs to the
    # references. A regeneration in any other order reads a stale artifact
    # and writes a reference that no longer describes what produced it.
    assert order.index("fixtures") < order.index("anchors")
    assert order.index("anchors") < order.index("references")


def _code_of(path: Path) -> str:
    """The module's code with its docstrings and comments removed.

    Prose is allowed to say "references"; the point of the rule is that no
    branch of the driver does.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            docstring = ast.get_docstring(node, clean=False)
            if docstring is not None:
                node.body = node.body[1:]
    return ast.unparse(tree)


def test_the_driver_names_no_family_of_its_own():
    source = _code_of(GOLDEN_ROOT / "regenerate.py")
    families = [
        path.stem for path in TARGETS_DIR.glob("*.py") if not path.stem.startswith("_")
    ]
    named = [
        family
        for family in families
        if re.search(rf"\b{re.escape(family)}\b", source)
    ]
    assert not named, (
        f"regenerate.py names {named}. It discovers its targets so that two "
        "families added on two branches do not edit the same line; naming one "
        "puts that enumeration back."
    )


def test_a_family_documents_itself_beside_its_target_module():
    # A target names its page with DOC when two targets are one family, so
    # the two foundation tiers share one story rather than splitting it.
    families = {target.doc for target in targets.discover().values()}
    # The three built-in families are described by the parent README, which
    # is about the anchors themselves. Everything added after them documents
    # itself in docs/, or the shared file becomes the conflict again.
    builtin = {"fixtures", "anchors", "references"}
    documented = {path.stem for path in DOCS_DIR.glob("*.md")} - {"README"}
    undocumented = families - builtin - documented
    assert not undocumented, (
        f"{sorted(undocumented)} have a target module and no docs/ page. "
        "See tests/golden/docs/README.md for what the page has to answer."
    )


def test_help_works_without_importing_the_framework():
    # --help has to work on a box that cannot load torch, or the one command
    # that says what can be regenerated is the one command you cannot run
    # until the environment is already right.
    blocker = (
        "import runpy, sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch is blocked for this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "sys.argv = ['regenerate.py', '--help']\n"
        # runpy, not exec: the driver resolves the repository root from
        # __file__, which a bare exec of its source does not define.
        f"runpy.run_path({str(GOLDEN_ROOT / 'regenerate.py')!r}, "
        "run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", blocker],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--i-know-what-i-am-doing" in result.stdout


@pytest.mark.parametrize("flag", [[], ["--target", "fixtures"]])
def test_it_still_refuses_without_the_acknowledgement(flag):
    result = subprocess.run(
        [sys.executable, str(GOLDEN_ROOT / "regenerate.py"), *flag],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode != 0
    assert "--i-know-what-i-am-doing" in result.stderr


def test_no_two_families_claim_the_same_anchor():
    anchors = targets.all_anchors()
    assert anchors, "the anchor registry is empty"
    for name, spec in anchors.items():
        checkpoint = REPO_ROOT / spec[0]
        assert checkpoint.exists(), f"{name} points at a missing checkpoint"
        assert Path(spec[1]).suffix == ".json"
