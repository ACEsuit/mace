"""Tests for plot-time data-loader key handling."""

import pytest

# mace.tools first on purpose: mace.tools.train imports TrainingPlotter, so
# importing mace.cli.visualise_train before it hits a partially initialised
# module. The cycle predates this test.
import mace.tools  # noqa: F401  # pylint: disable=unused-import
from mace.cli.visualise_train import belongs_to_head


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


@pytest.mark.parametrize("labelled", [False, True])
def test_dipole_scatter_filters_components_and_configurations(labelled):
    from types import SimpleNamespace

    import numpy as np
    import torch

    from mace.cli.visualise_train import InferenceMetric

    batch = SimpleNamespace(
        ptr=torch.tensor([0, 2, 5, 6]),
        weight=torch.tensor([1.0, 1.0, 0.0]),
        energy=None,
        forces=None,
        stress=None,
        virials=None,
        dipole=torch.tensor([[2.0, 4.0, 6.0], [9.0, 12.0, 15.0], [1.0, 2.0, 3.0]]),
        dipole_weight=(
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
            if labelled
            else torch.zeros((3, 3))
        ),
    )
    metric = InferenceMetric()
    metric.update(batch, {"dipole": batch.dipole + 3.0})
    results = metric.compute()
    if not labelled:
        assert "dipole" not in results
        return
    dipole = results["dipole"]
    np.testing.assert_allclose(dipole["reference"], [2, 6, 12])
    np.testing.assert_allclose(dipole["predicted"], [5, 9, 15])
    np.testing.assert_allclose(dipole["reference_per_atom"], [1, 3, 4])
    np.testing.assert_allclose(dipole["predicted_per_atom"], [2.5, 4.5, 5])


@pytest.mark.parametrize(
    "swa_start,epoch,stage",
    [
        (None, 1, "stage_one"),
        (2, 1, "stage_one"),
        (1, 1, "stage_two"),
        (0, 0, "stage_two"),
    ],
)
def test_plot_stage_filename_at_boundary(
    tmp_path, monkeypatch, swa_start, epoch, stage
):
    import json

    from mace.cli import visualise_train

    log = tmp_path / "run_train.txt"
    rows = []
    for i in range(2):
        rows.append(dict(mode="opt", epoch=i, loss=1.0))
        rows.append(
            dict(
                mode="eval",
                epoch=i,
                head="Default",
                loss=1.0,
                rmse_e_per_atom=0.1,
                rmse_f=0.2,
            )
        )
    log.write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(visualise_train, "model_inference", lambda *args: {})
    plotter = visualise_train.TrainingPlotter(
        str(log), ["Default"], "PerAtomRMSE", {}, {}, {}, "cpu", 0, swa_start=swa_start
    )
    plotter.plot(epoch, None, 0)
    assert [p.name for p in tmp_path.glob("*.png")] == [
        f"run_train_Default_{stage}.png"
    ]
