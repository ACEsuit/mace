"""Explicit statistics replace legacy scalar flags without entering the model config."""

import json
from dataclasses import asdict, replace

import pytest
import torch
from conftest import assert_close
from mace_core.config.legacy import LegacyFlagError
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.keys import KeySpecification
from mace_core.data.xyz import write_configurations
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.cli.run_train import parse, run
from mace_torch.deploy.loader import load_deployed
from mace_torch.finetune.stages import build
from mace_torch.train import run_data_stage, run_model_stage
from mace_torch.train.checkpoint import write_model
from test_mace_torch_training_pipeline import configuration


@pytest.mark.parametrize(
    "flag,value",
    [
        ("mean", "1.5"),
        ("std", "2.5"),
        ("avg_num_neighbors", "7.5"),
        ("compute_avg_num_neighbors", "False"),
        ("atomic_numbers", "[1,8]"),
    ],
)
def test_replaced_statistics_flags_explain_the_native_inputs(flag, value):
    with pytest.raises(LegacyFlagError, match=f"--{flag} merged") as error:
        parse([f"--{flag}", value])
    assert "--statistics_file" in str(error.value)
    assert "finetune.element_table" in str(error.value)


@pytest.mark.parametrize("legacy_json", [False, True])
def test_explicit_statistics_reach_the_model_and_reload_without_data(
    tmp_path, legacy_json
):
    config = configuration(tmp_path)
    measured = run_data_stage(config, DEFAULT_CATALOGUE)
    statistics = asdict(measured.statistics)
    statistics.update(mean=1.5, std=2.5, avg_num_neighbors=7.5)
    if legacy_json:
        for name in ("atomic_numbers", "atomic_energies"):
            statistics[name] = str(statistics[name])
    stats_path = tmp_path / "statistics.json"
    stats_path.write_text(json.dumps(statistics))
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    selected = parse(
        ["--config", str(config_path), "--statistics_file", str(stats_path)]
    )
    data = run_data_stage(selected, DEFAULT_CATALOGUE)
    assert data.e0s == measured.e0s
    assert tuple(data.z_table.zs) == (1, 8)
    built = run_model_stage(selected, data, DEFAULT_CATALOGUE)
    head = built.model.backbone.outputs.energy_head
    assert head.scale.tolist() == [2.5]
    assert head.shift.tolist() == [1.5]
    neighbor_buffers = [
        value
        for name, value in built.model.named_buffers()
        if name.endswith(".neighbours")
    ]
    assert neighbor_buffers and all(value.item() == 7.5 for value in neighbor_buffers)
    batch = next(iter(data.train_eval_loaders["default"])).graph
    expected = built.model(batch, compute=("forces",))
    artifact = write_model(tmp_path / "model", built.model, built.metadata)
    # The measured constants travel with the weights, so a deployed model
    # needs neither its training source nor the statistics file to rebuild.
    stats_path.unlink()
    (tmp_path / "train.xyz").unlink()
    loaded = load_deployed(artifact)
    actual = loaded.engine(batch, compute=("forces",))
    for name in ("energy", "forces"):
        torch.testing.assert_close(actual.get(name), expected.get(name), rtol=0, atol=0)
    assert loaded.config.model.normalization == {}


def test_estimated_e0s_use_the_named_foundation_head_and_train(tmp_path):
    document = configuration(tmp_path).model_dump(mode="json")
    head = document["data"]["heads"]["default"]
    head["keep_isolated_atoms"] = True
    document["data"]["heads"] = {
        "a": head,
        "b": {**head, "e0s": {"kind": "table", "values": {1: 10.0, 8: 20.0}}},
    }
    foundation = build(ResolvedConfig.model_validate(document))
    artifact = write_model(
        tmp_path / "foundation", foundation.model, foundation.metadata
    )
    loader = foundation.data.train_eval_loaders["b"]
    predictions = []
    for batch in loader:
        predictions.extend(
            foundation.model(batch.graph, compute=()).get("energy").detach().tolist()
        )
    correction = {1: 0.25, 8: -0.5}
    labelled = [
        replace(
            item,
            head="target",
            config_type="Default",
            properties={
                **item.properties,
                "energy": prediction
                + sum(correction[int(z)] for z in item.atomic_numbers),
            },
        )
        for item, prediction in zip(
            loader.dataset.configurations, predictions, strict=True
        )
    ]
    target = tmp_path / "target.xyz"
    write_configurations(target, labelled, KeySpecification.from_defaults())
    document["finetune"]["foundation_model"] = str(artifact)
    document["model"] = {"observables": ["energy", "forces"]}
    document["data"]["heads"] = {
        "target": {
            "train_file": str(target),
            "valid_file": str(target),
            "readout_from": "b",
            "e0s": {"kind": "estimated", "head": "b"},
        }
    }
    document["training"]["max_num_epochs"] = 1
    config = ResolvedConfig.model_validate(document)
    built = build(config)
    expected = {1: 10.25, 8: 19.5}
    assert_close(
        [built.data.e0s.values["target"][z] for z in expected], list(expected.values())
    )
    trained = run(config)
    assert len(trained.history) == 1
    assert torch.isfinite(torch.tensor(trained.history[0].train_loss))
    loaded = load_deployed(tmp_path / f"{config.runtime.name}.json")
    assert_close([loaded.e0s["target"][z] for z in expected], list(expected.values()))
