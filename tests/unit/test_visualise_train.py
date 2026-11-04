"""Tests for training plots and their interaction with application state."""

import json
import subprocess
import sys

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

# mace.tools first on purpose: mace.tools.train imports TrainingPlotter, so
# importing mace.cli.visualise_train before it hits a partially initialised
# module. The cycle predates this test.
import mace.tools  # noqa: F401  # pylint: disable=unused-import
from mace.cli.visualise_train import TrainingPlotter, belongs_to_head


def test_calculator_import_preserves_font_size(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import matplotlib; "
            "matplotlib.rcParams['font.size'] = 19; "
            "from mace.calculators import MACECalculator; "
            "assert matplotlib.rcParams['font.size'] == 19, "
            "matplotlib.rcParams['font.size']",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("fail_save", [False, True])
def test_plot_font_size_is_local(tmp_path, monkeypatch, fail_save):
    results_path = tmp_path / "results.log"
    records = []
    for epoch in (1, 2):
        records.append({"mode": "opt", "epoch": epoch, "loss": 1.0 / epoch})
        records.append(
            {
                "mode": "eval",
                "epoch": epoch,
                "head": "Default",
                "loss": 1.0 / epoch,
                "rmse_e": 0.1 / epoch,
                "rmse_f": 0.2 / epoch,
            }
        )
    results_path.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    monkeypatch.setattr("mace.cli.visualise_train.model_inference", lambda *args: {})
    plotter = TrainingPlotter(
        results_dir=str(results_path),
        heads=["Default"],
        table_type="TotalRMSE",
        train_valid_data={},
        test_data={},
        output_args={},
        device="cpu",
        plot_frequency=1,
    )
    savefig = Figure.savefig
    saved = []

    def check_and_save(fig, *args, **kwargs):
        assert plt.rcParams["font.size"] == 8
        assert fig.axes[0].yaxis.label.get_fontsize() == 8
        saved.append(args[0])
        if fail_save:
            raise RuntimeError("save failed")
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_and_save)
    try:
        with plt.rc_context({"font.size": 19, "axes.labelsize": "medium"}):
            if fail_save:
                with pytest.raises(RuntimeError, match="save failed"):
                    plotter.plot(2, None, rank=0)
            else:
                plotter.plot(2, None, rank=0)
                assert (tmp_path / "results_Default_stage_one.png").is_file()
            assert plt.rcParams["font.size"] == 19
        assert len(saved) == 1
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "name,head,expected",
    [
        # train/valid keys put the head last, test keys put it first
        ("train_H2O", "H2O", True),
        ("valid_H2O", "H2O", True),
        ("H2O_test", "H2O", True),
        ("H2O_liquid_test", "H2O", True),
        # a substring match would hand every H2O set to the H2 head as well
        ("train_H2O", "H2", False),
        ("valid_H2O", "H2", False),
        ("H2O_test", "H2", False),
        # and would claim the H2 sets for H2O in neither direction
        ("train_H2", "H2O", False),
        ("H2_test", "H2O", False),
        # unrelated heads never match
        ("train_solid", "H2O", False),
    ],
)
def test_belongs_to_head(name, head, expected):
    assert belongs_to_head(name, head) is expected
