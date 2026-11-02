"""Training curves shared by the CLI and in-run reporting."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def plot_training(
    paths,
    output: Path,
    *,
    keys=("loss",),
    min_epoch=0,
    linear=False,
    start_stage_two=None,
    heads=None,
    error_bars=False,
) -> Path:
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    series = defaultdict(lambda: defaultdict(list))
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            epoch = row.get("epoch")
            if epoch is None or epoch < min_epoch:
                continue
            head = row.get("head", "all heads")
            # Optimizer/training records belong to the whole run.
            if heads and "head" in row and head not in heads:
                continue
            for key in keys:
                value = row.get(key)
                if value is not None:
                    series[(head, row.get("mode", "valid"), key)][epoch].append(
                        float(value)
                    )
    if not series:
        raise ValueError(f"no plotted values for {list(keys)} in the selected log(s)")
    figure = Figure(figsize=(8, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for (head, mode, key), epochs in sorted(series.items()):
        steps = sorted(epochs)
        means = np.array([np.mean(epochs[step]) for step in steps])
        (line,) = axis.plot(steps, means, label=f"{head} {mode} {key}")
        if error_bars:
            # Match the sample deviation of the legacy grouped logs. A
            # single measurement has no uncertainty band.
            deviation = np.array(
                [
                    np.std(epochs[step], ddof=1) if len(epochs[step]) > 1 else 0.0
                    for step in steps
                ]
            )
            axis.fill_between(
                steps,
                means - deviation,
                means + deviation,
                alpha=0.2,
                color=line.get_color(),
            )
    if not linear:
        axis.set_yscale("symlog", linthresh=1e-12)
    if start_stage_two is not None:
        axis.axvline(start_stage_two, linestyle="--", color="grey")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Metric (recorded units)")
    axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    return output
