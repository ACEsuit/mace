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


@pytest.mark.parametrize("head", ["H2", "H2O", "H2_liquid"])
def test_scatter_points_stay_with_their_head(head):
    import matplotlib.pyplot as plt
    import numpy as np

    from mace.cli.visualise_train import plot_inference_from_results

    heads = ["H2", "H2O", "H2_liquid"]
    train_valid = {}
    tests = {}
    for i, name in enumerate(heads):
        result = {
            "head": name,
            "energy": {
                "reference_per_atom": np.array([i + 1.0]),
                "predicted_per_atom": np.array([i + 1.5]),
            },
        }
        train_valid[f"train_{name}"] = result
        train_valid[f"valid_{name}"] = result
        tests[f"{name}_test"] = result
    fig, ax = plt.subplots()
    try:
        plot_inference_from_results(
            [ax], train_valid, tests, head, [("energy", "Energy")]
        )
        assert len(ax.collections) == 3
        for points in ax.collections:
            np.testing.assert_allclose(
                points.get_offsets(),
                [[heads.index(head) + 1.0, heads.index(head) + 1.5]],
            )
    finally:
        plt.close(fig)


def test_inference_preserves_head_identity_for_ambiguous_loader_names():
    from types import SimpleNamespace

    import torch

    from mace.cli.visualise_train import model_inference

    class Batch(SimpleNamespace):
        def to(self, device):
            return self

        def to_dict(self):
            return vars(self)

    class Model(torch.nn.Module):
        heads = ["H2", "H2_liquid"]

        def forward(self, batch, **kwargs):
            return {"energy": batch["energy"] + 0.1}

    batch = Batch(
        ptr=torch.tensor([0, 2]),
        head=torch.tensor([0]),
        energy=torch.tensor([1.0]),
        weight=torch.ones(1),
        energy_weight=torch.ones(1),
        forces=None,
        stress=None,
        virials=None,
        dipole=None,
    )
    result = model_inference({"H2_liquid_test": [batch]}, Model(), {}, "cpu")
    assert result["H2_liquid_test"]["head"] == "H2"


@pytest.mark.parametrize("head", ["H2", "H2O"])
def test_epoch_curves_stay_with_their_head(head):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from mace.cli.visualise_train import plot_epoch_dependence

    rows = [dict(mode="opt", epoch=i, loss=1.0) for i in range(2)]
    for name, error in [("H2", 0.1), ("H2O", 0.9)]:
        rows.extend(
            dict(mode="eval", epoch=i, head=name, loss=error, rmse_e_per_atom=error)
            for i in range(2)
        )
    fig, axes = plt.subplots(1, 2)
    try:
        plot_epoch_dependence(
            axes, pd.DataFrame(rows), head, 1, [("rmse_e_per_atom", "Energy")]
        )
        expected = 100.0 if head == "H2" else 900.0
        np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [expected, expected])
    finally:
        plt.close(fig)


def test_single_quantity_scatter_accepts_a_single_axis():
    import matplotlib.pyplot as plt
    import numpy as np

    from mace.cli.visualise_train import plot_inference_from_results

    fig, ax = plt.subplots()
    result = {
        "dipole": {
            "reference_per_atom": np.array([1.0, 2.0, 3.0]),
            "predicted_per_atom": np.array([1.1, 2.1, 3.1]),
        }
    }
    try:
        plot_inference_from_results(
            ax, {"valid_Default": result}, {}, "Default", [("dipole", "Dipole")]
        )
        assert len(ax.collections) == 1
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "table,absolute,relative",
    [
        ("DipoleRMSE", "rmse_mu_per_atom", "rel_rmse_mu"),
        ("DipoleMAE", "mae_mu", "rel_mae_mu"),
    ],
)
def test_dipole_curve_keys_and_percentage_units(table, absolute, relative):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from mace.cli.visualise_train import error_type, plot_epoch_dependence

    rows = [dict(mode="opt", epoch=i, loss=1.0) for i in range(2)]
    rows += [
        dict(
            mode="eval",
            epoch=i,
            head="Default",
            loss=0.1,
            **{absolute: 0.2, relative: 25.0},
        )
        for i in range(2)
    ]
    fig, axes = plt.subplots(1, 2)
    try:
        plot_epoch_dependence(
            axes, pd.DataFrame(rows), "Default", 1, error_type[table][0]
        )
        np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [200, 200])
        np.testing.assert_allclose(fig.axes[-1].lines[0].get_ydata(), [25, 25])
    finally:
        plt.close(fig)


def _distributed_inference_head_worker(rank, rendezvous, directory, mixed):
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import torch

    from mace.cli.visualise_train import model_inference

    class Batch(SimpleNamespace):
        def to(self, device):
            return self

        def to_dict(self):
            return vars(self)

    class Model(torch.nn.Module):
        heads = ["H2", "H2_liquid"]

        def forward(self, batch, **kwargs):
            return {"energy": batch["energy"] + 0.1}

    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2
    )
    try:
        batch = Batch(
            ptr=torch.tensor([0, 2]),
            head=torch.tensor([rank if mixed else 1]),
            energy=torch.tensor([1.0 + rank]),
            weight=torch.ones(1),
            energy_weight=torch.ones(1),
            forces=None,
            stress=None,
            virials=None,
            dipole=None,
        )
        result = model_inference(
            {"H2_liquid_test": [batch]}, Model(), {}, "cpu", distributed=True
        )["H2_liquid_test"]
        np.testing.assert_allclose(result["energy"]["reference"], [1.0, 2.0])
        if mixed:
            assert "head" not in result
        else:
            assert result["head"] == "H2_liquid"
        (Path(directory) / str(rank)).write_text("passed")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("mixed", [False, True])
def test_distributed_inference_uses_global_head_identity(tmp_path, mixed):
    import torch.multiprocessing as mp

    mp.spawn(
        _distributed_inference_head_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path), mixed),
        nprocs=2,
        join=True,
    )
    assert all((tmp_path / str(rank)).read_text() == "passed" for rank in range(2))
