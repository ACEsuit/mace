#!/usr/bin/env python3
"""Assert that no file is governed by two toolchains, and none by neither.

Two toolchains coexist for the length of the migration:

    mace/**                    black + isort + pylint + mypy
    packages/** and the seams  ruff (lint + format) + ty

The cost of that is thrash: a file claimed by both formatters is rewritten one
way by a pull request and back by the next, and a file claimed by neither
quietly rots. This computes each toolchain's effective file set from the hook
patterns themselves, so the answer comes from the configuration that actually
runs rather than from a table in a document.

Two overlaps are hard failures, because they are the pairs that fight:

    black  and ruff-format   two formatters, opposite rewrites
    mypy   and ty            two type checkers, contradictory verdicts

Ruff *lint* on `mace/**` is deliberately not a conflict. It is reduced there to
the single FA102 rule, which guards a real TorchScript constraint, and linting
never rewrites a file.

Scope: the governed trees are `mace/**`, `packages/**` and the declared seams.
Everything else is ungoverned on purpose and reported as such, not as a
failure. `tests/` in particular is not formatted, by a standing decision, and
handing it over would reformat 93 files of the frozen suite.

Run from anywhere:  python3 tests/architecture/check_toolchain_ownership.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / ".pre-commit-config.yaml"
CI = REPO / ".github" / "workflows" / "ci-core.yaml"

#: Hook id -> the toolchain it belongs to. Hooks outside this map (whitespace
#: hygiene) belong to neither and are not part of the ownership question.
TOOLCHAIN_OF = {
    "black": "legacy",
    "isort": "legacy",
    "pylint": "legacy",
    "ruff-check": "new",
    "ruff-format": "new",
    "ty": "new",
}

#: The pairs that actively fight over a file.
CONFLICTS = (("black", "ruff-format"), ("mypy", "ty"))

#: mypy has no hook. Its scope is the argument of the legacy-lint CI step, so
#: it is read from there rather than invented here.
MYPY_STEP = re.compile(r"check_mypy_baseline\.py|python -m mypy (?P<target>\S+)")

GOVERNED = ("mace/", "packages/")
SEAMS = ("tests/parity/", "tests/conftest.py")


def tracked_python_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return out.stdout.split()


def hook_patterns() -> dict[str, str]:
    """hook id -> its `files:` regex, resolved through the YAML anchors."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    patterns: dict[str, str] = {}
    for repo in config["repos"]:
        for hook in repo["hooks"]:
            if hook["id"] in TOOLCHAIN_OF:
                patterns[hook["id"]] = hook.get("files", "")
    return patterns


def mypy_scope() -> str:
    """mypy owns `mace/**`, declared by the CI step rather than by a hook."""
    text = CI.read_text(encoding="utf-8") if CI.exists() else ""
    if "check_mypy_baseline.py" in text:
        return "^mace/"
    return ""


def main() -> int:
    patterns = hook_patterns()
    missing = sorted(set(TOOLCHAIN_OF) - set(patterns))
    if missing:
        print(f"hooks named in the ownership map but absent from the config: {missing}")
        return 1

    unscoped = sorted(name for name, pattern in patterns.items() if not pattern)
    if unscoped:
        print(
            f"hooks with no `files:` scope, so they claim the whole tree: {unscoped}"
        )
        return 1

    patterns["mypy"] = mypy_scope()
    if not patterns["mypy"]:
        print("mypy has no declared scope: the legacy-lint job does not run it")
        return 1

    compiled = {name: re.compile(pattern) for name, pattern in patterns.items()}
    files = tracked_python_files()

    overlaps: list[str] = []
    for left, right in CONFLICTS:
        both = [
            path for path in files
            if compiled[left].match(path) and compiled[right].match(path)
        ]
        if both:
            overlaps.append(
                f"  {left} AND {right} both claim {len(both)} file(s), "
                f"e.g. {both[:3]}"
            )

    unowned = [
        path for path in files
        if path.startswith(GOVERNED) or path.startswith(SEAMS)
        if not any(regex.match(path) for regex in compiled.values())
    ]

    if overlaps:
        print("toolchain ownership overlaps:")
        print("\n".join(overlaps))
    if unowned:
        print(f"\nfiles in a governed tree that no toolchain claims: {len(unowned)}")
        for path in unowned[:10]:
            print(f"  {path}")
    if overlaps or unowned:
        return 1

    legacy = sum(1 for path in files if compiled["black"].match(path))
    new = sum(1 for path in files if compiled["ruff-format"].match(path))
    ungoverned = len(files) - legacy - new
    print(
        f"1:1 ownership ok  (legacy {legacy} files, new {new} files, "
        f"{ungoverned} deliberately ungoverned)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
