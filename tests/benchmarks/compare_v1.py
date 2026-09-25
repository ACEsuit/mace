#!/usr/bin/env python3
"""Pair each v1 benchmark with the frozen tree's on the same case, as a table.

Reads a pytest-benchmark json that holds both ``test_inference_cpu.py`` and
``test_inference_v1.py`` from one session, and writes a markdown table for the
nightly run summary: one row per v1 case, its legacy counterpart's median, and
the ratio. A case where v1 is slower is named under the table, and one slower
than the frozen tree's cuEquivariance path is named first, since that is the
number an accelerated rewrite is expected to beat.

It never fails the job. A v1 case with no legacy counterpart in the file is
listed as unpaired rather than dropped, so a night on which the pairing stops
working says so.

Usage:  python tests/benchmarks/compare_v1.py benchmark.json >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _key(info: dict, backend: str) -> tuple:
    return (
        info.get("model"),
        info.get("regime"),
        info.get("dtype"),
        info.get("device"),
        backend,
    )


def compare(runs: list[dict]) -> str:
    legacy = {}
    for run in runs:
        info = run.get("extra_info", {})
        if info.get("stack") != "v1" and "median_seconds" in info:
            legacy[_key(info, info.get("backend"))] = info["median_seconds"]
    header = "| case | v1 backend | v1 median | legacy backend | legacy median |"
    lines = [f"{header} v1 / legacy |", "|---|---|---|---|---|---|"]
    slower, slower_than_cueq, unpaired = [], [], []
    for run in runs:
        info = run.get("extra_info", {})
        if info.get("stack") != "v1" or "median_seconds" not in info:
            continue
        case = f"{info['model']} {info['regime']} {info['dtype']} {info['device']}"
        mine = info["median_seconds"]
        theirs = legacy.get(_key(info, info.get("legacy_backend")))
        if theirs is None:
            unpaired.append(f"{case} on {info['backend']}")
            continue
        ratio = mine / theirs
        lines.append(
            f"| {case} | {info['backend']} | {mine * 1e3:.2f} ms | "
            f"{info['legacy_backend']} | {theirs * 1e3:.2f} ms | {ratio:.2f} |"
        )
        if ratio > 1.0:
            slower.append(f"{case}: {info['backend']} is {ratio:.2f}x the legacy time")
        legacy_cueq = legacy.get(_key(info, "cueq"))
        if legacy_cueq is not None and mine > legacy_cueq:
            slower_than_cueq.append(
                f"{case}: {info['backend']} {mine * 1e3:.2f} ms against legacy "
                f"cueq {legacy_cueq * 1e3:.2f} ms"
            )
    out = ["## v1 against the legacy baselines", "", *lines, ""]
    for title, items in (
        ("Slower than the legacy cuEquivariance path", slower_than_cueq),
        ("Slower than its legacy counterpart", slower),
        ("No legacy counterpart in this run", unpaired),
    ):
        if items:
            out += [f"**{title}:**", *[f"- {item}" for item in items], ""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "benchmark.json")
    raw = path.read_text() if path.exists() else ""
    runs = json.loads(raw or "{}").get("benchmarks", [])
    print(compare(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
