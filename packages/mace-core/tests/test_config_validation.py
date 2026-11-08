"""User mistakes fail before data loading; saved configurations remain readable."""

import json

import pytest
from mace_core.config.base import ConfigError
from mace_core.config.resolved import ResolvedConfig
from pydantic import ValidationError


@pytest.mark.parametrize(
    "document",
    [
        {"training": {"batch_size": 0}},
        {"training": {"valid_batch_size": 0}},
        {"training": {"eval_interval": 0}},
        {"training": {"lr": -1}},
        {"training": {"lr": float("nan")}},
        {"training": {"weight_decay": -1}},
        {"training": {"ema": {"decay": 1.5}}},
        {"training": {"optimizer": {"kind": "adam", "beta": 1}}},
        {"training": {"scheduler": {"kind": "plateau", "factor": 1.2}}},
        {"data": {"valid_fraction": 1.5}},
        {"data": {"num_workers": -1}},
        {"data": "train.xyz"},
        {"model": {"observables": 42}},
        {"loss": {"kind": "huber", "delta": 0}},
        {"loss": {"weights": {"forces": -1}}},
        {"loss": {"weights": {"forces": float("inf")}}},
    ],
)
def test_invalid_settings_raise_validation_errors(document):
    with pytest.raises(ValidationError):
        ResolvedConfig.from_dict(document)


@pytest.mark.parametrize(
    "field", ["weights", "stage_two_weights", "use_structure_weights"]
)
def test_unknown_loss_names_are_refused(field):
    with pytest.raises(ValidationError, match="forcces"):
        ResolvedConfig.from_dict({"loss": {field: {"forcces": 1}}})


def test_unknown_stage_loss_names_are_refused():
    with pytest.raises(ValidationError, match="forcces"):
        ResolvedConfig.from_dict(
            {"training": {"stages": [{"loss_weights": {"forcces": 2}}]}}
        )


@pytest.mark.parametrize(
    "stages",
    [
        [{"start_epoch": 0}, {"start_epoch": 2}],
        [{"name": "warmup"}, {"name": "fit"}],
    ],
)
def test_ambiguous_stages_are_refused(stages):
    with pytest.raises(ValidationError, match=r"training.stages"):
        ResolvedConfig.from_dict({"training": {"stages": stages}})


@pytest.mark.parametrize(
    "text,suffix",
    [
        ("training:\n  lr: 0.002\ntraining:\n  batch_size: 4\n", ".yaml"),
        ("training:\n  lr: 0.002\n  lr: 0.01\n", ".yaml"),
        ('{"training":{"lr":0.002},"training":{"batch_size":4}}', ".json"),
        ('{"training":{"lr":0.002,"lr":0.01}}', ".json"),
    ],
)
def test_duplicate_keys_are_refused(tmp_path, text, suffix):
    path = tmp_path / ("run" + suffix)
    path.write_text(text)
    with pytest.raises(ConfigError, match=r"duplicate.*(training|lr)"):
        ResolvedConfig.load(path)


def test_yaml_anchor_overrides_are_still_supported(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "training:\n  <<: &defaults {lr: 0.01, batch_size: 4}\n  lr: 0.002\n"
    )
    config = ResolvedConfig.load(path)
    assert config.training.lr == 0.002 and config.training.batch_size == 4


@pytest.mark.parametrize(
    "document",
    [
        {"loss": {"kind": {"kind": "huber", "delta": 0.02}}},
        {"training": {"scheduler": {"kind": {"kind": "constant"}}}},
    ],
)
def test_nested_discriminators_explain_the_flat_format(document):
    with pytest.raises(
        ValidationError, match="kind must be a string beside its settings"
    ):
        ResolvedConfig.from_dict(document)


@pytest.mark.parametrize(
    "loss",
    [
        {"kind": "weighted", "weights": {"energy": 2.0}},
        {"kind": "huber", "delta": 0.02},
        {"kind": "universal", "delta": 0.1},
        {"kind": "registered", "name": "example", "settings": {"size": 2}},
    ],
)
def test_flat_loss_roundtrip(loss):
    config = ResolvedConfig.from_dict({"loss": loss})
    assert config.to_resolved_dict()["loss"]["kind"] == loss["kind"]
    assert ResolvedConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    "scheduler",
    [
        {"kind": "constant"},
        {"kind": "plateau", "patience": 10},
        {"kind": "exponential", "gamma": 0.9},
    ],
)
def test_flat_scheduler_roundtrip(scheduler):
    config = ResolvedConfig.from_dict({"training": {"scheduler": scheduler}})
    assert (
        config.to_resolved_dict()["training"]["scheduler"]["kind"] == scheduler["kind"]
    )
    assert ResolvedConfig.from_dict(json.loads(config.model_dump_json())) == config
