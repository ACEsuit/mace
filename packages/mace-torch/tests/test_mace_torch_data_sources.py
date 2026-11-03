"""Ordered sources and named test datasets reach the existing stage contracts."""

from dataclasses import asdict

import numpy as np
import pytest
from ase.io import read, write
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.data import KeySpecification, open_dataset
from mace_core.data_spec.shard_format import write_shards
from mace_torch.data import GraphDataset
from mace_torch.train import DataStageError, run_data_stage
from test_mace_torch_training_pipeline import CATALOGUE, configuration

pytestmark = fp64_only


def shards(source, destination):
    keys = KeySpecification.from_defaults()
    items = list(
        open_dataset(source, key_spec=keys, keep_isolated_atoms=True).iter_range()
    )
    write_shards(items, destination, shard_size=3, key_spec=keys)
    return destination


def settings_at(tmp_path):
    return configuration(tmp_path).to_resolved_dict()


def structures(data, split):
    if split == "train":
        return data.train_loader.datasets["default"].configurations
    return getattr(data, split)["default"].dataset.configurations


@pytest.mark.parametrize("split", ["train_file", "valid_file", "test_file"])
def test_empty_source_lists_are_refused(tmp_path, split):
    settings = settings_at(tmp_path)
    settings["data"]["heads"]["default"][split] = []
    with pytest.raises(ValueError, match=split):
        ResolvedConfig.model_validate(settings)


def test_mixed_sources_preserve_order_splits_e0s_and_statistics(tmp_path):
    settings = settings_at(tmp_path)
    frames = read(tmp_path / "train.xyz", index=":")
    write(tmp_path / "first.xyz", frames[:7])
    write(tmp_path / "second.xyz", frames[7:])
    prepared = shards(tmp_path / "second.xyz", tmp_path / "prepared")
    for split in ("valid_file", "test_file"):
        settings["data"]["heads"]["default"][split] = str(tmp_path / "train.xyz")
    baseline = run_data_stage(ResolvedConfig.model_validate(settings), CATALOGUE)
    paths = [str(tmp_path / "first.xyz"), str(prepared)]
    for split in ("train_file", "valid_file", "test_file"):
        settings["data"]["heads"]["default"][split] = paths
    config = ResolvedConfig.model_validate(settings)
    assert config.to_resolved_dict()["data"]["heads"]["default"]["train_file"] == paths
    assert ResolvedConfig.model_validate_json(config.model_dump_json()) == config
    actual = run_data_stage(config, CATALOGUE)
    assert asdict(actual.e0s) == asdict(baseline.e0s)
    assert asdict(actual.statistics) == asdict(baseline.statistics)
    for split in ("train", "valid_loaders", "test_loaders"):
        for first, second in zip(
            structures(actual, split), structures(baseline, split), strict=True
        ):
            np.testing.assert_array_equal(first.positions, second.positions)
            np.testing.assert_array_equal(
                first.properties["forces"], second.properties["forces"]
            )
            assert first.properties["energy"] == second.properties["energy"]
    # Splitting occurs over the concatenated structures, not once per source.
    settings["data"]["heads"]["default"]["valid_file"] = None
    split = run_data_stage(ResolvedConfig.model_validate(settings), CATALOGUE)
    settings["data"]["heads"]["default"]["train_file"] = str(tmp_path / "train.xyz")
    merged = run_data_stage(ResolvedConfig.model_validate(settings), CATALOGUE)
    for first, second in zip(
        structures(split, "valid_loaders"),
        structures(merged, "valid_loaders"),
        strict=True,
    ):
        np.testing.assert_array_equal(first.positions, second.positions)


def test_test_directory_keeps_sets_and_heads_separate(tmp_path):
    settings = settings_at(tmp_path)
    directory = tmp_path / "tests"
    directory.mkdir()
    frames = read(tmp_path / "train.xyz", index="2:")
    write(directory / "water.xyz", frames[:2])
    write(tmp_path / "held.xyz", frames[2:7])
    shards(tmp_path / "held.xyz", directory / "shards.v2")
    (directory / ".hidden").write_text("not a dataset")
    settings["data"]["test_dir"] = str(directory)
    settings["data"]["heads"] = {
        name: {
            **settings["data"]["heads"]["default"],
            "test_file": str(tmp_path / "held.xyz"),
        }
        for name in ("pbe", "pbe_d3")
    }
    actual = run_data_stage(ResolvedConfig.model_validate(settings), CATALOGUE)
    assert list(actual.test_loaders) == [
        "pbe",
        "pbe_d3",
        "pbe_shards.v2",
        "pbe_water",
        "pbe_d3_shards.v2",
        "pbe_d3_water",
    ]
    for name, loader in actual.test_loaders.items():
        head = "pbe_d3" if name.startswith("pbe_d3") else "pbe"
        expected = 2 if name.endswith("water") else 5
        assert isinstance(loader.dataset, GraphDataset)
        assert len(loader.dataset) == expected
        assert {item.head for item in loader.dataset.configurations} == {head}
        assert {int(value) for batch in loader for value in batch.graph["head"]} == {
            actual.heads.index(head)
        }


@pytest.mark.parametrize(
    "problem", ["missing", "empty", "unreadable", "collision", "report_collision"]
)
def test_test_directory_refuses_missing_data_and_name_collisions(tmp_path, problem):
    settings = settings_at(tmp_path)
    directory = tmp_path / "tests"
    settings["data"]["test_dir"] = str(directory)
    if problem != "missing":
        directory.mkdir()
    if problem == "unreadable":
        (directory / "broken.xyz").write_text("unreadable")
    if problem == "collision":
        for suffix in ("xyz", "extxyz"):
            write(
                directory / f"same.{suffix}", read(tmp_path / "train.xyz", index="2:4")
            )
    if problem == "report_collision":
        write(directory / "default.xyz", read(tmp_path / "train.xyz", index="2:4"))
        settings["data"]["heads"]["train"] = settings["data"]["heads"]["default"]
    with pytest.raises(DataStageError, match=r"test_dir|test set"):
        run_data_stage(ResolvedConfig.model_validate(settings), CATALOGUE)


def test_each_requested_source_must_be_readable(tmp_path):
    settings = settings_at(tmp_path)
    settings["data"]["heads"]["default"]["train_file"] = [
        str(tmp_path / "train.xyz"),
        str(tmp_path / "missing.xyz"),
    ]
    with pytest.raises(DataStageError, match=r"default.*missing\.xyz"):
        run_data_stage(ResolvedConfig.model_validate(settings), CATALOGUE)


def test_legacy_head_lists_use_the_native_source_contract(tmp_path):
    from mace_torch.cli.run_train import parse

    paths = [str(tmp_path / "one.xyz"), str(tmp_path / "two.xyz")]
    settings = {split: paths for split in ("train_file", "valid_file", "test_file")}
    config = parse(["--heads", repr({"water": settings}), "--test_dir", str(tmp_path)])
    assert config.to_resolved_dict()["data"]["heads"]["water"]["train_file"] == paths
    for split in settings:
        assert tuple(map(str, getattr(config.data.heads["water"], split))) == tuple(
            paths
        )
    assert config.data.test_dir == tmp_path
    with pytest.raises(ValueError, match="multi_processed_test"):
        parse(["--multi_processed_test", "true"])
    with pytest.raises(ValueError, match=r"heads.water.test_dir"):
        parse(["--heads", repr({"water": {**settings, "test_dir": str(tmp_path)}})])


@pytest.mark.parametrize("prepared", [False, True])
def test_source_lists_preserve_the_replay_ratio_guard(tmp_path, prepared):
    settings = settings_at(tmp_path)
    frames = read(tmp_path / "train.xyz", index=":")
    write(tmp_path / "small.xyz", frames[:4])
    big = (
        shards(tmp_path / "train.xyz", tmp_path / "prepared")
        if prepared
        else tmp_path / "train.xyz"
    )
    common = {
        "valid_file": str(tmp_path / "small.xyz"),
        "e0s": {"kind": "isolated_atoms"},
    }
    settings["data"]["heads"] = {
        "big": {**common, "train_file": [str(big)]},
        "small": {**common, "train_file": [str(tmp_path / "small.xyz")]},
    }
    settings["data"]["ratio_guard"] = {"reference": "big", "threshold": 0.5}
    data = run_data_stage(ResolvedConfig.model_validate(settings), CATALOGUE)
    assert len(data.train_loader.datasets["big"]) == 12
    assert len(data.train_loader.datasets["small"]) == (2 if prepared else 8)
