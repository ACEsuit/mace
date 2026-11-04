"""Plot selection against the frozen renderer, comparing data before styling."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from mace_torch.train.plots import plot_training

from tests.golden.harness import CLOSED_FORM_FP64


@pytest.mark.parametrize("min_epoch", [-1, 0, 2])
@pytest.mark.parametrize("head", ["water", "salt"])
def test_minimum_epoch_matches_legacy_for_each_head(tmp_path, monkeypatch, isolated, min_epoch, head):
    rows = []
    for seed in (1, 2):
        for epoch in range(4):
            rows.append(dict(name="run", seed=seed, epoch=epoch, mode="opt", loss=seed + epoch))
            for name, factor in (("water", 1), ("salt", 2)):
                rows.append(dict(name="run", seed=seed, epoch=epoch, mode="eval", head=name,
                                 loss=seed + epoch, rmse_e=0.01 * factor * (seed + epoch)))
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    figures = []
    original = Figure.savefig

    def save(figure, *args, **kwargs):
        figures.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    with matplotlib.rc_context():
        from mace.cli.plot_train import plot

        plot(pd.DataFrame(rows), min_epoch, str(tmp_path / "legacy"), "png",
             True, None, False, "rmse_e", head)
        plot_training([path], tmp_path / "native.png", keys=("rmse_e",),
                      min_epoch=min_epoch, heads=[head], linear=True)
    reference = figures[0].axes[1].lines[0]
    native = figures[1].axes[0].lines[0]
    np.testing.assert_array_equal(native.get_xdata(), reference.get_xdata())
    # Legacy presents errors in meV; V1 labels them in their recorded units.
    np.testing.assert_allclose(native.get_ydata(), reference.get_ydata() / 1000,
                               atol=CLOSED_FORM_FP64.atol, rtol=CLOSED_FORM_FP64.rtol)


@pytest.mark.parametrize("epoch,stage", [(1, "stage_one"), (2, "stage_two"), (3, "stage_two")])
def test_head_and_stage_filenames_follow_the_stage_contract_with_legacy_curves(
    tmp_path, monkeypatch, isolated, epoch, stage
):
    import torch

    rows = []
    for step in range(epoch + 1):
        rows.append(dict(epoch=step, mode="opt", loss=1.0 + step))
        for head, offset in (("water", 2), ("salt", 4)):
            rows.append(dict(epoch=step, mode="eval", head=head, loss=offset + step,
                             rmse_e_per_atom=0.02, rmse_f=0.04))
    legacy_dir, native_dir = tmp_path / "legacy", tmp_path / "native"
    legacy_dir.mkdir()
    path = legacy_dir / "history.txt"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    figures = {}
    original = Figure.savefig

    def save(figure, output, *args, **kwargs):
        figures[Path(output)] = figure
        return original(figure, output, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    with matplotlib.rc_context():
        import mace.tools  # initialize the frozen training imports before its plotter
        from mace.cli.visualise_train import TrainingPlotter

        # No inference dataset: this comparison owns filenames and loss curves.
        plotter = TrainingPlotter(str(path), ["water", "salt"], "PerAtomRMSE",
                                  {}, {}, {}, "cpu", 0, swa_start=2)
        plotter.plot(epoch, torch.nn.Module(), 0)
        plot_training([path], native_dir / "history.png", separate_heads=True, stage=stage)
    native_names = {f"history_{head}_{stage}.png" for head in ("water", "salt")}
    assert {p.name for p in native_dir.glob("*.png")} == {"history.png", *native_names}
    legacy_names = {p.name for p in legacy_dir.glob("*.png")}
    # Legacy's strict comparison misnames the first stage-two epoch. V1 uses
    # the active schedule entry at that epoch, as its training loop requires.
    legacy_stage = "stage_one" if epoch == 2 else stage
    assert legacy_names == {f"history_{head}_{legacy_stage}.png" for head in ("water", "salt")}
    assert (native_names == legacy_names) == (epoch != 2)
    for head in ("water", "salt"):
        name = f"history_{head}_{stage}.png"
        reference = figures[legacy_dir / f"history_{head}_{legacy_stage}.png"]
        curves = {line.get_label(): line for line in figures[native_dir / name].axes[0].lines}
        for label, ylabel in [("all heads opt loss", "Training Loss"),
                              (f"{head} eval loss", "Validation Loss")]:
            expected = next(axis for axis in reference.axes if axis.get_ylabel() == ylabel).lines[0]
            np.testing.assert_array_equal(curves[label].get_xdata(), expected.get_xdata())
            np.testing.assert_allclose(curves[label].get_ydata(), expected.get_ydata(),
                                       atol=CLOSED_FORM_FP64.atol, rtol=CLOSED_FORM_FP64.rtol)
