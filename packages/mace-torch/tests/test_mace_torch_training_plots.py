"""Plot flags are checked against the data drawn, including error bands."""

import json
from pathlib import Path

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


@pytest.fixture
def named_figures(monkeypatch):
    saved = {}
    original = Figure.savefig

    def save(figure, path, *args, **kwargs):
        saved[Path(path).name] = figure
        return original(figure, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    return saved


def test_cli_names_each_head_and_keeps_an_explicit_output_exact(log, named_figures):
    options = ["plot", "training", "--path", str(log)]
    assert main(options) == 0
    assert set(named_figures) == {"curves.png", "curves_water.png", "curves_salt.png"}
    for head in ("water", "salt"):
        labels = {
            line.get_label()
            for line in named_figures[f"curves_{head}.png"].axes[0].lines
        }
        assert labels == {
            "all heads opt loss",
            "all heads train loss",
            f"{head} valid loss",
        }
    original = log.with_name("curves_water.png").read_bytes()
    named_figures.clear()
    explicit = log.with_name("chosen.png")
    assert main([*options, "--heads", "water", "--output", str(explicit)]) == 0
    assert set(named_figures) == {"chosen.png"}
    assert explicit.read_bytes() == original


def test_stage_files_keep_earlier_snapshots_and_global_curves(tmp_path, named_figures):
    from mace_torch.train.logs import write_epoch

    runtime = RuntimeConfig(work_dir=tmp_path)
    metrics = {"water": {"loss": 2.0}, "salt": {"loss": 4.0}}
    write_epoch(runtime, None, None, metrics, last=False)
    for epoch in range(2):
        write_epoch(runtime, epoch, 1.0, metrics, stage="stage_one", last=epoch == 1)
    directory = runtime.directory("results")
    earlier = {
        path.name: path.read_bytes() for path in directory.glob("*stage_one.png")
    }
    assert set(earlier) == {"mace_water_stage_one.png", "mace_salt_stage_one.png"}
    write_epoch(runtime, 2, 3.0, metrics, stage="stage_two", last=True)
    assert all(
        (directory / name).read_bytes() == content for name, content in earlier.items()
    )
    for head in metrics:
        curves = named_figures[f"mace_{head}_stage_two.png"].axes[0].lines
        assert {line.get_label() for line in curves} == {
            "all heads train loss",
            f"{head} valid loss",
        }
        assert all(list(line.get_xdata()) == [0, 1, 2] for line in curves)


def test_requested_heads_name_only_their_own_files(log, named_figures):
    assert (
        main(
            [
                "plot",
                "training",
                "--path",
                str(log),
                "--heads",
                "water",
                "--output_format",
                "pdf",
            ]
        )
        == 0
    )
    assert set(named_figures) == {"curves.pdf", "curves_water.pdf"}
    assert log.with_name("curves_water.pdf").read_bytes().startswith(b"%PDF-")


def test_head_and_custom_stage_names_cannot_alias_paths(tmp_path):
    from mace_torch.train.plots import plot_training

    path = tmp_path / "names.jsonl"
    heads = ["water/ice", "water%2Fice", "a_stage-one", "a", "all heads"]
    rows = [{"epoch": 0, "mode": "train", "loss": 1.0}]
    rows += [{"epoch": 0, "mode": "valid", "head": head, "loss": 2.0} for head in heads]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    for stage in ("stage_one", "one", "one_stage-two"):
        plot_training([path], tmp_path / "run.png", separate_heads=True, stage=stage)
    names = {item.name for item in tmp_path.glob("*.png")}
    assert len(names) == 1 + len(heads) * 3
    assert {
        "run_water%2Fice_stage_one.png",
        "run_water%252Fice_stage_one.png",
        "run_a_stage-one_stage-one.png",
        "run_a_stage-one%5Fstage-two.png",
    } <= names
    assert not (tmp_path / "run_water").exists()


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--heads", "water", "--error_bars"],
        [
            "--keys",
            "rmse_energy",
            "--min_epoch",
            "2",
            "--linear",
            "--start_stage_two",
            "3",
        ],
    ],
)
def test_arbitrary_input_names_preserve_curves_and_options(log, options):
    output = log.parent / "chosen.png"
    baseline = None
    for name in (
        "fit_run-7_train.txt",
        "not-a-results-log.txt",
        "native.jsonl",
        "a log without a suffix",
    ):
        source = log.with_name(name)
        source.write_bytes(log.read_bytes())
        assert (
            main(
                [
                    "plot",
                    "training",
                    "--path",
                    str(source),
                    "--output",
                    str(output),
                    *options,
                ]
            )
            == 0
        )
        content = output.read_bytes()
        if baseline is None:
            baseline = content
        assert content == baseline


@pytest.mark.parametrize(
    "invalid,expected",
    [
        ("{", "Expecting property name"),
        ("[]", "JSON object"),
        ("null", "JSON object"),
        ('{"loss": 1.0}', "missing epoch"),
        *[
            (json.dumps({"epoch": epoch, "loss": 1.0}), "epoch must be")
            for epoch in ("one", True, float("nan"), float("inf"))
        ],
        (json.dumps({"epoch": 0, "head": ["water"], "loss": 1.0}), "head must be"),
        (json.dumps({"epoch": 0, "mode": None, "loss": 1.0}), "mode must be"),
        *[
            (json.dumps({"epoch": 0, "loss": loss}), "metric 'loss'")
            for loss in ({"value": 1.0}, "bad", True)
        ],
    ],
)
def test_invalid_log_content_names_file_and_line_before_drawing(
    tmp_path, capsys, figures, invalid, expected
):
    directory = tmp_path / "logs"
    directory.mkdir()
    valid = json.dumps({"epoch": 1, "head": "water", "loss": 1.0})
    (directory / "a.jsonl").write_text(valid + "\n")
    source = directory / "z arbitrary.txt"
    source.write_text("\n" + valid + "\n" + invalid + "\n")
    output = directory.with_suffix(".png")
    output.write_bytes(b"previous plot")
    assert main(["plot", "training", "--path", str(directory)]) == 2
    error = capsys.readouterr().err
    assert f"{source}:3:" in error and expected in error
    assert "Traceback" not in error
    assert not figures
    assert output.read_bytes() == b"previous plot"
    assert list(tmp_path.glob("*.png")) == [output]


def test_optional_metrics_and_initial_diagnostics_remain_readable(tmp_path, figures):
    rows = [
        {"epoch": None, "loss": None, "head": "water"},
        {"epoch": 0, "loss": 1000},
        {"epoch": 1.0, "loss": 2, "head": None, "unselected": {"detail": 1}},
        {"epoch": 2, "loss": float("nan")},
        {"epoch": 3, "loss": float("inf")},
        {"epoch": 4, "loss": None},
        {"epoch": 5, "other_metric": 1},
    ]
    source = tmp_path / "diagnostics.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n")
    assert main(["plot", "training", "--path", str(source), "--heads", "water"]) == 0
    (line,) = figures[-1].axes[0].lines
    assert line.get_label() == "all heads valid loss"
    np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
    np.testing.assert_array_equal(line.get_ydata(), [2, float("nan"), float("inf")])
