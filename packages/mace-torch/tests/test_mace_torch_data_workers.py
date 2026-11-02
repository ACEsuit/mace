"""Worker processes preserve batches; custom batches honor Torch's pin hook."""

from functools import partial

import pytest
import torch
from conftest import fp64_only
from mace_torch.data.batch import GraphDataset, TrainingBatch
from mace_torch.train import run_data_stage
from test_mace_torch_training_pipeline import CATALOGUE, configuration, write_dataset


def assert_same(actual, expected):
    for name in ("graph", "targets", "property_weights"):
        first, second = getattr(actual, name), getattr(expected, name)
        assert first.keys() == second.keys()
        for key, value in first.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, second[key]), (name, key)


@fp64_only
@pytest.mark.parametrize("mode", ["balanced", "proportional"])
def test_spawn_workers_preserve_training_and_evaluation_batches(
    tmp_path, monkeypatch, mode
):
    from mace_torch.train import loaders

    config = configuration(tmp_path, head_balancing=mode)
    settings = config.to_resolved_dict()
    settings["data"]["heads"]["default"]["test_file"] = str(tmp_path / "train.xyz")
    config = type(config).model_validate(settings)
    serial = run_data_stage(config, CATALOGUE)
    settings["data"]["num_workers"] = 1
    parallel = run_data_stage(type(config).model_validate(settings), CATALOGUE)
    expected_training = list(serial.train_loader.batches(0, drop_last=False))
    # Force spawn even on Linux versions whose default is fork.
    monkeypatch.setattr(
        loaders,
        "DataLoader",
        partial(torch.utils.data.DataLoader, multiprocessing_context="spawn"),
    )
    for name in ("valid_loaders", "train_eval_loaders", "test_loaders"):
        for loader in getattr(parallel, name).values():
            assert loader.num_workers == 1
            loader.multiprocessing_context = "spawn"
    pairs = [
        (
            expected_training,
            parallel.train_loader.batches(0, drop_last=False),
        )
    ]
    for name in ("valid_loaders", "train_eval_loaders", "test_loaders"):
        pairs.extend(
            (getattr(serial, name)[head], loader)
            for head, loader in getattr(parallel, name).items()
        )
    for serial_batches, worker_batches in pairs:
        for expected, actual in zip(serial_batches, worker_batches, strict=True):
            assert_same(actual, expected)


@fp64_only
def test_validation_indices_follow_the_work_directory(tmp_path):
    config = configuration(tmp_path)
    write_dataset(tmp_path / "train.xyz", count=60)
    settings = config.to_resolved_dict()
    settings["runtime"]["work_dir"] = str(tmp_path / "new-work-dir")
    data = run_data_stage(type(config).model_validate(settings), CATALOGUE)
    assert isinstance(data.valid_loaders["default"].dataset, GraphDataset)
    written = tmp_path / "new-work-dir/default_valid_indices_1.txt"
    indices = [int(value) for value in written.read_text().split()]
    assert (
        len(indices)
        == len(set(indices))
        == len(data.valid_loaders["default"].dataset)
        == 12
    )
    assert all(0 <= index < 60 for index in indices)
    assert not list(tmp_path.glob("*valid_indices*"))


def test_custom_batch_pins_every_tensor(monkeypatch):
    from torch.utils.data._utils.pin_memory import pin_memory

    batch = TrainingBatch(
        {"positions": torch.zeros(2, 3), "pbc": (False, False, False)},
        {"energy": torch.ones(1)},
        {"energy": torch.ones(1)},
    )
    seen = []

    def pin(value):
        seen.append(id(value))
        return value.clone()

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    result = pin_memory(batch)
    assert set(seen) == {
        id(batch.graph["positions"]),
        id(batch.targets["energy"]),
        id(batch.property_weights["energy"]),
    }
    assert result is not batch
    assert result.graph["pbc"] == batch.graph["pbc"]
    assert_same(result, batch)


@pytest.mark.gpu
def test_custom_batch_uses_page_locked_host_memory():
    from torch.utils.data._utils.pin_memory import pin_memory

    batch = TrainingBatch(
        {"positions": torch.zeros(2, 3)},
        {"energy": torch.ones(1)},
        {"energy": torch.ones(1)},
    )
    result = pin_memory(batch)
    assert result.graph["positions"].is_pinned()
    assert result.targets["energy"].is_pinned()
    assert result.property_weights["energy"].is_pinned()
    assert_same(result, batch)
