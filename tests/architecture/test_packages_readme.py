"""The navigation guide in `packages/README.md` describes the tree as it is.

Every source file under `packages/*/src` has a row in the guide's tables, and
every row names a file that exists. A guide that lists files by hand goes stale
the first time someone adds one, so the check is what makes it worth writing.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = REPO_ROOT / "packages"
README = PACKAGES / "README.md"

#: A table row whose first cell is a backticked path.
ROW = re.compile(r"^\| `([^`]+)` \|")


def listed() -> set[str]:
    guide = README.read_text().split("## Navigating the tree", 1)
    assert len(guide) == 2, "the README has no `## Navigating the tree` section"
    return {
        match.group(1)
        for line in guide[1].splitlines()
        if (match := ROW.match(line))
    }


def on_disk() -> set[str]:
    files = set()
    for source in PACKAGES.glob("*/src"):
        for path in source.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files.add(path.relative_to(source).as_posix())
    return files


def test_every_source_file_has_a_row():
    missing = sorted(on_disk() - listed())
    assert not missing, (
        f"source files with no row in packages/README.md: {missing}. Add one "
        f"line saying what is in each."
    )


def test_every_row_names_a_file_that_exists():
    stale = sorted(listed() - on_disk())
    assert not stale, (
        f"packages/README.md describes files that do not exist: {stale}. "
        f"Remove or rename their rows."
    )
