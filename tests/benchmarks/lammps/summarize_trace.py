"""Summarize a single rank's Nsight Systems SQLite export from nsys-window.patch."""

import argparse
import json
import sqlite3
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from statistics import mean


def occupied_ns(intervals):
    """Count overlapping device intervals once."""
    total = end = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def cu_eq_launches(db):
    """Map CUDA correlation IDs to the enclosing cuEq call on the launching thread."""
    by_thread = defaultdict(list)
    for thread, start, end, label in db.execute(
        """SELECT e.globalTid, e.start, e.end, coalesce(e.text, s.value) label
           FROM NVTX_EVENTS e LEFT JOIN StringIds s ON s.id = e.textId
           WHERE label LIKE 'MACE cuEq %' AND e.end > e.start ORDER BY e.start"""
    ):
        by_thread[thread].append((start, end, label))
    starts = {t: [r[0] for r in ranges] for t, ranges in by_thread.items()}
    launches = {}
    for thread, start, correlation in db.execute(
        "SELECT globalTid, start, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        i = bisect_right(starts.get(thread, ()), start) - 1
        if i >= 0:
            _, end, label = by_thread[thread][i]
            if start < end:
                launches[correlation] = label
    return launches


def summarize(db):
    windows = db.execute(
        """SELECT e.start, e.end FROM NVTX_EVENTS e
           LEFT JOIN StringIds s ON s.id = e.textId
           WHERE coalesce(e.text, s.value) = 'MACE pair step' AND e.end > e.start
           ORDER BY e.start"""
    ).fetchall()
    if len(windows) != 8:
        raise ValueError(f"expected 8 complete MD steps, found {len(windows)}")
    # Discard the first and last steps next to the CUPTI capture boundaries.
    windows = windows[1:-1]
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
    kernels = db.execute(
        """SELECT k.start, k.end, k.deviceId, s.value, k.correlationId
           FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON k.demangledName = s.id"""
    ).fetchall()
    transfers = []
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            transfers.extend(db.execute(f"SELECT start, end FROM {table}"))
    launches = cu_eq_launches(db)
    operators = defaultdict(int)
    durations = defaultdict(int)
    calls = defaultdict(int)
    kernel_busy, device_busy, pair_wall = [], [], []
    devices = set()
    for first, last in windows:
        spans = []
        for start, end, device, name, correlation in kernels:
            if end <= first or start >= last:
                continue
            a, b = max(start, first), min(end, last)
            spans.append((a, b))
            durations[name] += b - a
            if correlation in launches:
                operators[launches[correlation]] += b - a
            calls[name] += 1
            devices.add(device)
        if not spans:
            raise ValueError("no device kernels in the MD step")
        pair_wall.append(last - first)
        kernel_busy.append(occupied_ns(spans))
        spans.extend(
            (max(a, first), min(b, last))
            for a, b in transfers
            if b > first and a < last
        )
        device_busy.append(occupied_ns(spans))
    if len(devices) != 1:
        raise ValueError(f"expected one compute device per rank, found {devices}")
    device = devices.pop()
    gpu, bus = db.execute(
        "SELECT name, busLocation FROM TARGET_INFO_GPU WHERE id = ?", (device,)
    ).fetchone()
    samples = len(windows)
    return {
        "gpu": gpu,
        "bus": bus,
        "sampled_steps": samples,
        "pair_wall_ms": mean(pair_wall) / 1e6,
        "gpu_kernel_busy_ms": mean(kernel_busy) / 1e6,
        "gpu_busy_including_copies_ms": mean(device_busy) / 1e6,
        "cueq_ms_by_call": {
            k: v / (samples * 1e6) for k, v in sorted(operators.items())
        },
        "kernels": [
            {
                "name": name,
                "ms_per_step": duration / (samples * 1e6),
                "calls_per_step": calls[name] / samples,
            }
            for name, duration in sorted(durations.items(), key=lambda x: -x[1])
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    with sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True) as db:
        print(json.dumps(summarize(db), indent=2))
