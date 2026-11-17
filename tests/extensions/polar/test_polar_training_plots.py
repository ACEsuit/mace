"""Training plots for PolarMACE runs.

These live here rather than in tests/workflows so that they run in the polar CI
job, the only PR job with graph_longrange installed. The training helpers are
shared with tests/workflows/test_training_plots.py.
"""

import pytest

from tests.workflows.test_training_plots import check_multihead_plots, train

pytest.importorskip("matplotlib")


@pytest.mark.parametrize("stage_two", [False, True])
def test_polar_energy_force_training_plots_without_dipole_labels(tmp_path, stage_two):
    extra = dict(model="PolarMACE", loss="weighted", max_num_epochs=2, plot_frequency=1)
    if stage_two:
        extra.update(swa=None, start_swa=1)
    drawn, output = train(tmp_path, "polar", **extra)
    assert "Plotting failed" not in output
    assert "Creating Plotter failed" not in output
    assert any("stage_one" in name for name in drawn), drawn
    assert any("stage_two" in name for name in drawn) == stage_two, drawn


@pytest.mark.parametrize("stage_two", [False, True])
def test_polar_multihead_plots_with_shared_test_basenames(tmp_path, stage_two):
    check_multihead_plots(tmp_path, "PolarMACE", stage_two)
