"""Typed head weights, validation batches and explicit graph input contracts."""

import json
from dataclasses import replace

import numpy as np
import pytest
import yaml
from ase.io import read, write
from conftest import assert_close
from mace_core.config.legacy import LegacyFlagError
from mace_core.config.resolved import ResolvedConfig
from mace_core.data import Configuration, KeySpecification, open_dataset
from mace_core.data_spec.shard_format import write_shards
from mace_core.elements import AtomicNumberTable, DefaultKeys
from mace_core.observables import DEFAULT_CATALOGUE, resolve_requested
from mace_torch.cli.run_train import parse, run
from mace_torch.data import GraphDataset, graph_from_configuration, make_loader
from mace_torch.graph.collate import collate
from mace_torch.train import build_loss, run_data_stage, run_model_stage
from mace_torch.train.checkpoint import latest_run_checkpoint
from mace_torch.train.loop import evaluate
from mace_torch.train.metrics import metric_specs
from safetensors.numpy import load_file
from test_mace_torch_training_pipeline import configuration


def test_typed_heads_weights_match_stored_weights_and_resume(tmp_path, dtype):
    document = configuration(tmp_path).model_dump(mode="json")
    document["precision"]["model"] = str(dtype).removeprefix("torch.")
    document["runtime"]["name"] = "weights"
    source = tmp_path / "train.xyz"
    frames = read(source, index=":")
    for index, atoms in enumerate(frames[2:]):
        atoms.info.update(
            config_type="bulk" if index % 2 else "molecule", config_weight=0.5
        )
    write(source, frames)
    base_head = document["data"]["heads"]["default"]
    weights = {"a": {"bulk": 2.0, "molecule": 0.0}, "b": {"bulk": 0.5, "molecule": 3.0}}
    document["data"]["heads"] = {
        name: {
            **base_head,
            "config_type_weights": values,
            "weight": 0.25 if name == "b" else 1.0,
        }
        for name, values in weights.items()
    }
    document["data"]["heads"]["b"]["e0s"] = {
        "kind": "table",
        "values": {1: -13.1, 8: -2039.5},
    }
    stored = {}
    for name, multipliers in weights.items():
        adjusted = [item.copy() for item in frames]
        for atoms in adjusted:
            atoms.info["config_weight"] = atoms.info.get(
                "config_weight", 1.0
            ) * multipliers.get(atoms.info["config_type"], 1.0)
        stored[name] = tmp_path / f"stored-{name}.xyz"
        write(stored[name], adjusted)

    def fit(directory, epochs, baked=False, resume=False):
        directory.mkdir(exist_ok=True)
        current = json.loads(json.dumps(document))
        current["runtime"]["work_dir"] = str(directory)
        current["training"]["max_num_epochs"] = epochs
        if baked:
            for name, head in current["data"]["heads"].items():
                head.update(train_file=str(stored[name]), config_type_weights={})
        path = directory / "run.yaml"
        path.write_text(yaml.safe_dump(current, sort_keys=False))
        config = parse(
            ["--config", str(path), *(["--restart_latest"] if resume else [])]
        )
        result = run(config)
        assert all(np.isfinite(row.train_loss) for row in result.history)
        return result

    full, baked, split = (tmp_path / name for name in ("full", "baked", "split"))
    fit(full, 2)
    fit(baked, 2, baked=True)
    fit(split, 1)
    resumed = fit(split, 2, resume=True)
    assert [record.epoch for record in resumed.history] == [1]
    documents, tensors = [], []
    for directory in (full, baked, split):
        path = latest_run_checkpoint(directory, "weights")
        assert path is not None
        documents.append(json.loads(path.read_text()))
        tensors.append(load_file(str(path.with_suffix(".safetensors"))))
    for actual in tensors[1:]:
        assert actual.keys() == tensors[0].keys()
        for name, expected in tensors[0].items():
            np.testing.assert_array_equal(actual[name], expected, err_msg=name)
    for actual in documents[1:]:
        for name in ("state", "optimizer", "scheduler", "ema", "model_tensors"):
            assert actual[name] == documents[0][name], name


@pytest.mark.parametrize("batch_size", [1, 5, 32])
def test_validation_batch_size_keeps_every_structure_and_the_same_errors(
    tmp_path, dtype, batch_size
):
    document = configuration(tmp_path).model_dump(mode="json")
    document["precision"]["model"] = str(dtype).removeprefix("torch.")
    valid = tmp_path / "valid.xyz"
    write(valid, read(tmp_path / "train.xyz", index=":")[2:])
    document["data"]["heads"]["default"]["valid_file"] = str(valid)
    document["training"]["valid_batch_size"] = batch_size
    config = ResolvedConfig.model_validate(document)
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    loader = data.valid_loaders["default"]
    counts = [len(batch.graph["ptr"]) - 1 for batch in loader]
    assert counts == [min(batch_size, 12 - start) for start in range(0, 12, batch_size)]
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    requested = resolve_requested(config.model.observables, DEFAULT_CATALOGUE)
    loss, specs = build_loss(requested, config.loss), metric_specs(requested)
    actual = evaluate(built.model, loader, loss, specs)
    assert isinstance(loader.dataset, GraphDataset)
    whole = make_loader(
        loader.dataset, batch_size=12, shuffle=False, float_dtype=config.precision.model
    )
    expected = evaluate(built.model, whole, loss, specs)
    assert actual.keys() == expected.keys()
    # Reported loss retains legacy's batch-dependent reduction; its parity
    # is checked against MACELoss in test_data_controls_parity.py.
    for name in actual.keys() - {"loss"}:
        assert_close(actual[name], expected[name], name)


def test_test_directory_applies_each_heads_type_weights(tmp_path):
    document = configuration(tmp_path).model_dump(mode="json")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    frames = read(tmp_path / "train.xyz", index=":")[2:5]
    for atoms in frames:
        atoms.info.update(config_type="bulk", config_weight=0.5)
    write(test_dir / "probe.xyz", frames)
    head = document["data"]["heads"]["default"]
    document["data"].update(
        test_dir=str(test_dir),
        heads={
            name: {**head, "config_type_weights": {"bulk": factor}}
            for name, factor in (("a", 2.0), ("b", 0.0))
        },
    )
    data = run_data_stage(ResolvedConfig.model_validate(document), DEFAULT_CATALOGUE)
    for name, expected in (("a", 1.0), ("b", 0.0)):
        dataset = data.test_loaders[f"{name}_probe"].dataset
        assert isinstance(dataset, GraphDataset)
        items = dataset.configurations
        assert [item.weight for item in items] == [expected] * 3
        assert all(item.head == name for item in items)


def test_retired_head_keys_and_undeclared_property_promotion_stay_absent(dtype):
    with pytest.raises(LegacyFlagError, match="head_key"):
        parse(["--head_key", "custom_head"])
    assert "head" not in DefaultKeys.keydict()
    assert "head_key" not in DefaultKeys.keydict()
    array = np.array([1.0, 2.0])
    item = Configuration(
        atomic_numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        properties={"payload": array, "note": "untyped metadata"},
        head="source",
    )
    graph = graph_from_configuration(item, cutoff=2.0, z_table=AtomicNumberTable([1]))
    batch = collate([graph], dtype=dtype)
    assert batch["note"] == ("untyped metadata",)
    assert isinstance(batch["payload"], tuple) and batch["payload"][0] is array
    assert item.properties["payload"] is array and array.shape == (2,)
    declared = graph_from_configuration(
        item,
        cutoff=2.0,
        z_table=AtomicNumberTable([1]),
        features={"payload": "atom"},
    )
    np.testing.assert_array_equal(declared["payload"], array)
    batch = collate([declared], dtype=dtype, features={"payload": "atom"})
    assert batch["payload"].dtype == dtype and batch["payload"].shape == (2,)


@pytest.mark.parametrize("prepared_weights", [False, True])
def test_prepared_weights_and_run_weights_are_applied_once(tmp_path, prepared_weights):
    document = configuration(tmp_path).model_dump(mode="json")
    source = tmp_path / "train.xyz"
    frames = read(source, index=":")
    for atoms in frames[2:]:
        atoms.info.update(config_type="bulk", config_weight=0.5)
    write(source, frames)
    head = document["data"]["heads"]["default"]
    head.update(
        config_type_weights={"bulk": 3.0}, valid_file=str(source), test_file=str(source)
    )
    raw = run_data_stage(ResolvedConfig.model_validate(document), DEFAULT_CATALOGUE)
    keys = KeySpecification.from_defaults()
    items = list(
        open_dataset(source, key_spec=keys, keep_isolated_atoms=True).iter_range()
    )
    if prepared_weights:
        items = [
            replace(
                item, weight=item.weight * (3.0 if item.config_type == "bulk" else 1.0)
            )
            for item in items
        ]
    destination = tmp_path / "prepared"
    write_shards(items, destination, shard_size=5, key_spec=keys, r_max=5.0)
    head.update(
        train_file=str(destination),
        valid_file=str(destination),
        test_file=str(destination),
    )
    if prepared_weights:
        head["config_type_weights"] = {}
    config = ResolvedConfig.model_validate(document)
    for _ in range(2):
        actual = run_data_stage(config, DEFAULT_CATALOGUE)
        for name in ("train_eval_loaders", "valid_loaders", "test_loaders"):
            expected_loader = getattr(raw, name)["default"]
            actual_loader = getattr(actual, name)["default"]
            expected = [item.weight for item in expected_loader.dataset.configurations]
            assert [
                item.weight for item in actual_loader.dataset.configurations
            ] == expected
