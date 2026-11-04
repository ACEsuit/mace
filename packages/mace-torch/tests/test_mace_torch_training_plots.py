"""Plot flags are checked against the data drawn, including error bands."""

import json

import numpy as np
import pytest
from mace_core.cli import main
from mace_core.config.runtime import RuntimeConfig
from matplotlib.figure import Figure

from tests.golden.harness import CLOSED_FORM_FP64


@pytest.fixture
def log(tmp_path):
    records = [{"epoch": None, "mode": "valid", "head": "water", "loss": 100.0}]
    for epoch in range(4):
        records.extend(
            [
                {"epoch": epoch, "mode": "opt", "loss": 1.0 + epoch},
                {"epoch": epoch, "mode": "opt", "loss": 3.0 + epoch},
                {"epoch": epoch, "mode": "train", "loss": 2.0 + epoch},
                {
                    "epoch": epoch,
                    "mode": "valid",
                    "head": "water",
                    "loss": 4.0 + epoch,
                    "rmse_energy": 0.02,
                },
                {
                    "epoch": epoch,
                    "mode": "valid",
                    "head": "salt",
                    "loss": 8.0 + epoch,
                    "rmse_energy": 0.04,
                },
            ]
        )
    path = tmp_path / "curves.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records))
    return path


@pytest.fixture
def figures(monkeypatch):
    saved = []
    savefig = Figure.savefig

    def save(figure, *args, **kwargs):
        saved.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    return saved


def test_plot_filters_scale_metrics_and_stage_alias(log, figures):
    options = [
        "plot",
        "training",
        "--path",
        str(log),
        "--keys",
        "rmse_energy",
        "--min_epoch",
        "2",
        "--heads",
        "water",
        "--linear",
    ]
    content = []
    for alias in ("--start_swa", "--start_stage_two"):
        assert main([*options, alias, "3"]) == 0
        axis = figures[-1].axes[0]
        assert axis.get_yscale() == "linear"
        curve, stage = axis.lines
        assert curve.get_label() == "water valid rmse_energy"
        assert list(curve.get_xdata()) == [3]
        assert list(curve.get_ydata()) == [0.02]
        assert list(stage.get_xdata()) == [3, 3]
        content.append(log.with_suffix(".png").read_bytes())
    assert content[0] == content[1]


def test_head_selection_keeps_global_training_and_sample_deviation(log, figures):
    assert (
        main(
            ["plot", "training", "--path", str(log), "--heads", "water", "--error_bars"]
        )
        == 0
    )
    axis = figures[-1].axes[0]
    assert axis.get_yscale() == "symlog"
    curves = {line.get_label(): line for line in axis.lines}
    assert set(curves) == {
        "all heads opt loss",
        "all heads train loss",
        "water valid loss",
    }
    curve = curves["all heads opt loss"]
    assert list(curve.get_xdata()) == [1, 2, 3]
    assert list(curve.get_ydata()) == [3.0, 4.0, 5.0]
    # Two step losses [2, 4] at epoch one: sample std is sqrt(2).
    band = axis.collections[0].get_paths()[0].vertices
    limits = band[band[:, 0] == 1, 1]
    np.testing.assert_allclose(
        [limits.min(), limits.max()],
        [3 - np.sqrt(2), 3 + np.sqrt(2)],
        atol=CLOSED_FORM_FP64.atol,
        rtol=CLOSED_FORM_FP64.rtol,
    )


@pytest.mark.parametrize(
    "enabled,frequency,epochs", [(False, 1, []), (True, 0, [2]), (True, 2, [1, 2])]
)
def test_plot_frequency_includes_final_epoch(
    tmp_path, figures, enabled, frequency, epochs
):
    from mace_torch.train.logs import write_epoch

    runtime = RuntimeConfig(work_dir=tmp_path, plot=enabled, plot_frequency=frequency)
    metrics = {"water": {"loss": 2.0}}
    write_epoch(runtime, None, None, metrics, last=False)
    assert not figures
    drawn = []
    for epoch in range(3):
        before = len(figures)
        write_epoch(runtime, epoch, 1.0, metrics, last=epoch == 2)
        if len(figures) > before:
            drawn.append(epoch)
    assert drawn == epochs
    output = runtime.directory("results") / f"{runtime.name}.png"
    assert output.exists() == enabled
    if enabled:
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert all(
            list(line.get_xdata()) == [0, 1, 2] for line in figures[-1].axes[0].lines
        )
