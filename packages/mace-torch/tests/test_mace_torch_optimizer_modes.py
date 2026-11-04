"""Optimizer modes, actual CLI configuration and exact training continuation."""

import json

import numpy as np
import pytest
from mace_core.config.training import TrainingConfig
from mace_torch.cli.run_train import parse, run
from mace_torch.deploy.loader import load_deployed
from mace_torch.train import loop, optimizers
from mace_torch.train.checkpoint import latest_run_checkpoint
from pydantic import ValidationError
from safetensors.numpy import load_file
from test_mace_torch_full_batch import configuration
from torch import nn


def test_missing_schedulefree_dependency_has_an_install_hint(monkeypatch):
    monkeypatch.setattr(optimizers, "AdamWScheduleFree", None)
    model = nn.Module()
    model.node_embedding = nn.Linear(2, 2)
    with pytest.raises(ImportError, match=r"mace-torch-v1\[schedulefree\]"):
        optimizers.build_optimizer(
            model, TrainingConfig(optimizer={"kind": "schedulefree"})
        )


RECIPES = [
    "adam",
    "adamw",
    *[
        pytest.param(name, marks=pytest.mark.schedulefree)
        for name in (
            "schedulefree",
            "schedulefree_ema",
            "schedulefree_sparse",
            "schedulefree_stage",
            "schedulefree_to_adam",
            "schedulefree_to_lbfgs",
            "adam_to_schedulefree",
        )
    ],
]


@pytest.mark.parametrize("recipe", RECIPES)
def test_optimizer_cli_training_and_resume_preserve_every_state(
    tmp_path, recipe, monkeypatch
):
    built_optimizers = []
    build_optimizer = loop.build_optimizer
    evaluate_heads = loop.evaluate_heads

    def build(*args, **kwargs):
        optimizer = build_optimizer(*args, **kwargs)
        built_optimizers.append(optimizer)
        return optimizer

    def evaluate(*args, **kwargs):
        # Real validation must see evaluation coordinates, including the
        # diagnostic before epoch zero and validation under EMA's context.
        for group in built_optimizers[-1].param_groups:
            assert not group.get("train_mode", False)
        return evaluate_heads(*args, **kwargs)

    monkeypatch.setattr(loop, "build_optimizer", build)
    monkeypatch.setattr(loop, "evaluate_heads", evaluate)
    base = configuration(tmp_path).model_dump(mode="json")
    start_kind = "adam" if recipe == "adam_to_schedulefree" else recipe.split("_")[0]
    base["runtime"].update(name="fit")
    base["training"].update(
        ema={"enabled": recipe == "schedulefree_ema", "decay": 0.9},
        eval_interval=2 if recipe == "schedulefree_sparse" else 1,
    )
    if recipe.endswith("_stage") or "_to_" in recipe:
        next_kind = recipe.split("_to_")[-1] if "_to_" in recipe else "inherit"
        base["training"]["stages"] = [
            {"name": "main", "optimizer": {"kind": "inherit"}},
            {
                "name": "next",
                "start_epoch": 2,
                "lr": 0.003,
                "optimizer": {"kind": next_kind},
            },
        ]
        if next_kind == "lbfgs":
            base["training"]["stages"][1].update(
                optimizer={"kind": "lbfgs", "max_iter": 2},
                scheduler={"kind": {"kind": "constant"}},
            )
    flags = [
        "--optimizer",
        start_kind,
        "--lr",
        "0.007",
        "--scheduler",
        "ExponentialLR",
        "--lr_scheduler_gamma",
        "0.8",
    ]
    if start_kind == "schedulefree":
        flags += [
            "--beta1_schedulefree",
            "0.73",
            "--beta2_schedulefree",
            "0.93",
            "--warmup_steps_schedulefree",
            "3",
        ]
    else:
        flags += ["--beta", "0.87"]

    def fit(directory, epochs, resume=False):
        directory.mkdir(exist_ok=True)
        document = json.loads(json.dumps(base))
        document["runtime"]["work_dir"] = str(directory)
        document["training"]["max_num_epochs"] = epochs
        source = directory / "config.json"
        source.write_text(json.dumps(document))
        return run(
            parse(
                [
                    "--config",
                    str(source),
                    *flags,
                    *(["--restart_latest"] if resume else []),
                ]
            )
        )

    full, split = tmp_path / "full", tmp_path / "split"
    reference = fit(full, 4)
    fit(split, 2)
    checkpoint = latest_run_checkpoint(split, "fit")
    initial = load_file(str(checkpoint).removesuffix(".json") + ".safetensors")
    trained = fit(split, 4, resume=True)
    assert all(
        not group.get("train_mode", False)
        for optimizer in built_optimizers
        for group in optimizer.param_groups
    )
    assert [record.epoch for record in trained.history] == [2, 3]
    assert all(
        np.isfinite(record.train_loss)
        for record in (*reference.history, *trained.history)
    )
    documents, tensors = [], []
    for directory in (full, split):
        latest = latest_run_checkpoint(directory, "fit")
        documents.append(json.loads(latest.read_text()))
        tensors.append(load_file(str(latest).removesuffix(".json") + ".safetensors"))
    assert tensors[0].keys() == tensors[1].keys()
    for key in tensors[0]:
        np.testing.assert_array_equal(tensors[0][key], tensors[1][key], err_msg=key)
    for key in ("state", "optimizer", "scheduler", "ema", "model_tensors"):
        assert documents[0][key] == documents[1][key], key
    assert any(
        not np.array_equal(initial[key], tensors[1][key])
        for key in documents[1]["model_tensors"]
    )
    ending = documents[1]["optimizer"]["kind"]
    if ending == "AdamWScheduleFree":
        assert documents[1]["scheduler"] is None
    deployed = load_deployed(split / "fit.json")
    expected = load_file(full / "fit.safetensors")
    actual = load_file(split / "fit.safetensors")
    assert expected.keys() == actual.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    assert deployed.config.training.optimizer.kind == start_kind


@pytest.mark.parametrize("source", ["flags", "legacy_file", "structured_file"])
def test_individual_optimizer_and_scheduler_flags_keep_the_other_settings(
    tmp_path, source
):
    flags = ["--beta", "0.73", "--lr_factor", "0.6", "--scheduler_patience", "7"]
    if source != "flags":
        path = tmp_path / "config.json"
        document = {"beta": 0.73, "lr_factor": 0.6, "scheduler_patience": 7}
        if source == "structured_file":
            document = {"training": {"optimizer": {"kind": "adamw", "amsgrad": False}}}
        path.write_text(json.dumps(document))
        flags = ["--config", str(path), *(flags if source == "structured_file" else [])]
    config = parse(flags).training
    assert config.optimizer.beta == 0.73
    assert config.scheduler.kind.factor == 0.6
    assert config.scheduler.kind.patience == 7
    if source == "structured_file":
        assert config.optimizer.kind == "adamw"
        assert config.optimizer.amsgrad is False


def test_schedulefree_and_scheduler_overrides_use_the_file_choice(tmp_path):
    path = tmp_path / "config.json"
    document = {
        "training": {
            "optimizer": {"kind": "schedulefree", "beta2": 0.91},
            "scheduler": {
                "kind": {"kind": "exponential", "gamma": 0.8},
                "group_factors": {"embedding": 0.5},
            },
        }
    }
    path.write_text(json.dumps(document))
    config = parse(
        [
            "--config",
            str(path),
            "--beta1_schedulefree",
            "0.73",
            "--warmup_steps_schedulefree",
            "3",
            "--lr_scheduler_gamma",
            "0.7",
        ]
    ).training
    assert config.optimizer.model_dump() == {
        "kind": "schedulefree",
        "beta1": 0.73,
        "beta2": 0.91,
        "warmup_steps": 3,
    }
    assert config.scheduler.kind.gamma == 0.7
    changed = parse(
        [
            "--config",
            str(path),
            "--optimizer",
            "adamw",
            "--scheduler",
            "ReduceLROnPlateau",
            "--lr_factor",
            "0.6",
        ]
    ).training
    assert changed.optimizer.kind == "adamw"
    assert changed.scheduler.kind.factor == 0.6
    assert changed.scheduler.group_factors == {"embedding": 0.5}
    document["training"]["optimizer"]["beta"] = 0.8
    path.write_text(json.dumps(document))
    with pytest.raises(ValidationError, match="beta"):
        parse(["--config", str(path), "--beta1_schedulefree", "0.7"])
