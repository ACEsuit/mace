"""Native multi-source loading against independently read frozen datasets."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from ase.io import write
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.train import run_data_stage, run_model_stage, run_train_stage

from tests.parity.test_data_backend_parity import (
    ParsedOnly,
    assert_same,
    fixture_databases,  # noqa: F401
    fixture_prepared,  # noqa: F401
    frames,
    frozen_hdf5,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*legacy HDF5 layout:FutureWarning")


def configuration(tmp_path, train, **data):
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(tmp_path)},
            "data": {
                "heads": {
                    "Default": {
                        "train_file": train,
                        "valid_file": train,
                        "test_file": train,
                        "e0s": {"kind": "table", "values": {1: 0.0, 8: 0.0}},
                    }
                },
                "pin_memory": False,
                **data,
            },
            "model": {
                "r_max": 5.0,
                "num_channels": 4,
                "num_interactions": 2,
                "max_ell": 1,
                "correlation": 1,
            },
            "training": {"max_num_epochs": 1, "batch_size": 4},
        }
    )


def read_file(path):
    reader = frozen_hdf5(path)
    return [reader[index] for index in range(len(reader))]


def read_shards(directory):
    return [item for path in sorted(directory.glob("*.h5")) for item in read_file(path)]


def test_mixed_file_lists_match_legacy_readers_and_train(
    prepared, databases, tmp_path, isolated, monkeypatch
):
    import mace.data.lmdb_dataset as frozen_module
    from mace.data import KeySpecification, load_from_xyz
    from mace.tools import AtomicNumberTable

    source = tmp_path / "structures.xyz"
    write(source, frames(3, 31))
    _, xyz = load_from_xyz(str(source), KeySpecification.from_defaults())
    expected = xyz + read_shards(prepared / "prep_train")
    monkeypatch.setattr(frozen_module, "AtomicData", ParsedOnly)
    reader = frozen_module.LMDBDataset(
        databases, r_max=5.0, z_table=AtomicNumberTable([1, 8])
    )
    expected += [reader[index] for index in range(len(reader))]
    del reader
    paths = [str(source), str(prepared / "prep_train"), databases]
    config = configuration(tmp_path, paths)
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    for dataset in (
        data.train_loader.datasets["Default"],
        data.valid_loaders["Default"].dataset,
        data.test_loaders["Default"].dataset,
    ):
        assert len(dataset) == len(expected)
        for index, (actual, frozen) in enumerate(
            zip(dataset.configurations, expected, strict=True)
        ):
            assert_same(actual, frozen, index)
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    trained = run_train_stage(config, built)
    assert len(trained.history) == 1
    assert np.isfinite(trained.history[0].valid_loss)
    assert trained.best_epoch == 0


@pytest.mark.parametrize("layout", ["flat", "sharded"])
def test_test_directory_matches_frozen_hdf5_sets(prepared, tmp_path, isolated, layout):
    directory = tmp_path / "tests"
    directory.mkdir()
    expected = {}
    if layout == "sharded":
        shutil.copytree(prepared / "prep_train", directory / "held_out")
        expected["Default_held_out"] = read_shards(directory / "held_out")
    else:
        for index, shard in enumerate(sorted((prepared / "prep_train").glob("*.h5"))):
            path = directory / f"held_out_{index}_test.h5"
            shutil.copy(shard, path)
            expected[f"Default_held_out_{index}_test"] = read_file(path)
    config = configuration(
        tmp_path, str(prepared / "prep_train"), test_dir=str(directory)
    )
    actual = run_data_stage(config, DEFAULT_CATALOGUE)
    assert set(actual.test_loaders) == {"Default", *expected}
    for name, items in expected.items():
        configurations = actual.test_loaders[name].dataset.configurations
        assert len(configurations) == len(items)
        for index, (ours, theirs) in enumerate(zip(configurations, items, strict=True)):
            assert_same(ours, theirs, index)
