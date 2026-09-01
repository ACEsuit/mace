#!/usr/bin/env python3
"""Run mypy over the frozen tree and fail only on newly introduced errors.

mypy is configured in `.mypy.ini` and shipped in the `dev` extra, but nothing
has ever invoked it, so the legacy tree carries a large backlog. That tree is
byte-frozen as the numerical oracle and this migration may not edit it, so the
check cannot be made to pass by fixing the code. It is baselined instead: the
counts below are what the tree produced when the check was added, and the job
fails when a count goes up or a new one appears.

The baseline is keyed on `path::error-code`, never on line numbers or message
text. A frozen tree still moves when a file above it changes, and a message can
be reworded by a mypy release; either would turn a stable backlog into a
spurious failure.

A count that drops is not an error. It is reported, with the command to record
the improvement, so the baseline ratchets downwards and cannot silently absorb
a new error behind a fixed one.

Run from anywhere:  python3 tests/architecture/check_mypy_baseline.py
                    python3 tests/architecture/check_mypy_baseline.py --update
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name("mypy_baseline.json")
TARGET = "mace"

#: `path:line: error: message  [code]`
ERROR = re.compile(r"^(?P<path>[^:]+):\d+: error: .*\[(?P<code>[a-z-]+)\]\s*$")


def run_mypy() -> Counter:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", TARGET],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    if "error:" not in result.stdout and result.returncode not in (0, 1):
        raise SystemExit(
            f"mypy did not run:\n{result.stdout}\n{result.stderr}"
        )
    counts: Counter = Counter()
    for line in result.stdout.splitlines():
        match = ERROR.match(line)
        if match:
            counts[f"{match['path']}::{match['code']}"] += 1
    return counts


def load_baseline() -> Counter:
    if not BASELINE.exists():
        return Counter()
    return Counter(json.loads(BASELINE.read_text(encoding="utf-8")))


def write_baseline(counts: Counter) -> None:
    BASELINE.write_text(
        json.dumps(dict(sorted(counts.items())), indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    current = run_mypy()
    if "--update" in sys.argv:
        write_baseline(current)
        print(f"baseline written: {sum(current.values())} errors "
              f"over {len(current)} path/code pairs")
        return 0

    baseline = load_baseline()
    regressions = {
        key: (baseline.get(key, 0), count)
        for key, count in current.items()
        if count > baseline.get(key, 0)
    }
    improvements = {
        key: (was, current.get(key, 0))
        for key, was in baseline.items()
        if current.get(key, 0) < was
    }

    if improvements:
        print(f"{len(improvements)} path/code pair(s) improved. Record them with:")
        print("  python3 tests/architecture/check_mypy_baseline.py --update")
        for key, (was, now) in sorted(improvements.items())[:10]:
            print(f"  {key}: {was} -> {now}")

    if regressions:
        print(f"\n{len(regressions)} NEW mypy error(s) in the frozen tree:")
        for key, (was, now) in sorted(regressions.items()):
            path, code = key.split("::")
            print(f"  {path}  [{code}]  {was} -> {now}")
        print("\nThe legacy tree is frozen; a new error here means something "
              "edited it, or a dependency changed under it.")
        return 1

    print(f"mypy baseline ok: {sum(current.values())} known errors, none new")
    return 0


if __name__ == "__main__":
    sys.exit(main())
