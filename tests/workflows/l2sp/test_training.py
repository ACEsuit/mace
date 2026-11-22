import json
from pathlib import Path

import pytest
import torch

from tests.golden.anchors import TRAIN_SET, anchor_path, load_anchor
from tests.helpers import run_mace_train


def training_args(work, **overrides):
    args = {
        "name": "l2sp",
        "seed": 11,
        "train_file": str(TRAIN_SET),
        "valid_fraction": 0.25,
        "E0s": "foundation",
        "loss": "huber",
        "energy_weight": 10,
        "forces_weight": 100,
        "stress_weight": 1000,
        "batch_size": 4,
        "valid_batch_size": 4,
        "max_num_epochs": 2,
        "eval_interval": 1,
        "device": "cpu",
        "default_dtype": "float64",
        "foundation_model": str(anchor_path("tiny_scaleshift")),
        "foundation_model_elements": True,
        "multiheads_finetuning": False,
        "l2sp": True,
        "l2sp_strength": 10.0,
        "freeze": "products,interactions.1,readouts",
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "lr": 0.001,
        "error_table": "PerAtomRMSEstressvirials",
        "save_cpu": None,
        "work_dir": str(work),
        "save_all_checkpoints": None,
    }
    args.update(overrides)
    return args


def latest_checkpoint(work):
    paths = (Path(work) / "checkpoints").glob("l2sp_run-11_epoch-*.pt")
    path = max(paths, key=lambda p: int(p.stem.split("epoch-")[1].split("_")[0]))
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "optimizer,averaging",
    [
        ("adam", False),
        ("adam", True),
        ("adamw", False),
        ("adamw", True),
        pytest.param("schedulefree", False, marks=pytest.mark.schedulefree),
        ("lbfgs", False),
        ("lbfgs", True),
    ],
)
def test_cli_training_checkpoint_resume_and_frozen_weights(
    tmp_path, optimizer, averaging
):
    options = {"ema": None, "start_swa": 1} if averaging else {}
    if optimizer == "lbfgs":
        options["lbfgs"] = None
    args = training_args(
        tmp_path,
        optimizer="adam" if optimizer == "lbfgs" else optimizer,
        **options,
    )
    result = run_mace_train(args, capture_output=True, text=True)
    assert result.returncode == 0
    initial = latest_checkpoint(tmp_path)
    source = load_anchor("tiny_scaleshift").state_dict()
    changed = []
    selected = {
        name
        for group in initial["optimizer"]["param_groups"]
        for name in group["param_names"]
    }
    for name, tensor in initial["model"].items():
        if name in selected:
            if not torch.equal(tensor, source[name]):
                changed.append(name)
        elif name in source and tensor.shape == source[name].shape:
            torch.testing.assert_close(tensor, source[name], atol=1e-14, rtol=1e-14)
    assert any(name.startswith("node_embedding.") for name in changed)
    assert any(name.startswith("interactions.0.") for name in changed)
    for group, refs in zip(
        initial["optimizer"]["param_groups"], initial["optimizer"]["l2sp"]["references"]
    ):
        for name, reference in zip(group["param_names"], refs):
            torch.testing.assert_close(reference, source[name], atol=0, rtol=0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "l2sp_run-11_train.txt")
        .read_text()
        .splitlines()
    ]
    assert any(row.get("l2sp_squared_distance", 0) > 0 for row in rows)
    assert (tmp_path / "l2sp.model").exists()
    export_name = "l2sp_stagetwo.model" if averaging else "l2sp.model"
    exported = torch.load(tmp_path / export_name, weights_only=False)
    assert {
        name for name, param in exported.named_parameters() if param.requires_grad
    } == selected

    run_mace_train(
        {**args, "max_num_epochs": 3, "restart_latest": None},
        capture_output=True,
        text=True,
    )
    resumed = latest_checkpoint(tmp_path)
    for old_refs, new_refs in zip(
        initial["optimizer"]["l2sp"]["references"],
        resumed["optimizer"]["l2sp"]["references"],
    ):
        for old, new in zip(old_refs, new_refs):
            torch.testing.assert_close(old, new, atol=0, rtol=0)

    # Changing the selected set must fail rather than partially load and restart.
    bad = run_mace_train(
        {**args, "max_num_epochs": 4, "restart_latest": None, "freeze": 0},
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0
    assert "L2SPCheckpointError" in bad.stderr


@pytest.mark.timeout(90)
def test_restart_without_checkpoint_fails(tmp_path):
    result = run_mace_train(
        training_args(tmp_path, restart_latest=None),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "found no L2-SP checkpoint" in result.stderr
