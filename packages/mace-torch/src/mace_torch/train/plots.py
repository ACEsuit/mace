"""Training curves shared by the CLI and in-run reporting."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


def plot_training(
    paths,
    output: Path,
    *,
    keys=("loss",),
    min_epoch=-1,
    linear=False,
    start_stage_two=None,
    heads=None,
    error_bars=False,
    separate_heads: bool = False,
    stage: str | None = None,
) -> Path:
    """Draw epochs strictly greater than ``min_epoch``, as in the legacy CLI.

    In-run reporting keeps epoch zero with the default of ``-1``. The CLI
    supplies its own default of zero; ``epoch=None`` is never a trained epoch.
    ``separate_heads`` adds head-specific files beside the combined output;
    ``stage`` identifies the training snapshot in those filenames.
    """
    series = defaultdict(lambda: defaultdict(list))
    known_heads = set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "head" in row:
                known_heads.add(row["head"])
            epoch = row.get("epoch")
            if epoch is None or epoch <= min_epoch:
                continue
            head = row.get("head")
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
    if separate_heads:
        for head in sorted(known_heads.intersection(heads) if heads else known_heads):
            selected = {
                key: value for key, value in series.items() if key[0] in (None, head)
            }
            if selected:
                _draw(
                    selected,
                    _head_path(output, head, stage),
                    linear,
                    start_stage_two,
                    error_bars,
                )
    _draw(series, output, linear, start_stage_two, error_bars)
    return output


def _head_path(output: Path, head: str, stage: str | None) -> Path:
    suffix = quote(head, safe="")
    if stage is not None:
        # Reserve legacy's two suffixes; custom stages get a separate namespace.
        # Escaping underscores prevents a head/stage pair aliasing another pair.
        label = (
            stage
            if stage in {"stage_one", "stage_two"}
            else "stage-" + quote(stage, safe="").replace("_", "%5F")
        )
        suffix += "_" + label
    return output.with_name(f"{output.stem}_{suffix}{output.suffix}")


def _draw(
    series, output: Path, linear: bool, start_stage_two, error_bars: bool
) -> None:
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for (head, mode, key), epochs in sorted(
        series.items(),
        key=lambda entry: (
            "all heads" if entry[0][0] is None else entry[0][0],
            *entry[0][1:],
        ),
    ):
        steps = sorted(epochs)
        means = np.array([np.mean(epochs[step]) for step in steps])
        label = "all heads" if head is None else head
        (line,) = axis.plot(steps, means, label=f"{label} {mode} {key}")
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
