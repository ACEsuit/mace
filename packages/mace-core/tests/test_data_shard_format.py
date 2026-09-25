"""The v2 shard format: what is written is what is read, and the manifest is
checked like a file format.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
from mace_core.data import DataBackendError, KeySpecification
from mace_core.data.backends.hdf5 import HDF5Backend
from mace_core.data_spec import (
    MANIFEST_NAME,
    SCHEMA_VERSION,
    SHARD_FORMAT,
    read_manifest,
    write_shards,
)
from mace_core_data_fixtures import STATISTICS, configurations, write_prepared


@pytest.fixture
def written(tmp_path):
    return write_prepared(tmp_path)


def assert_same(read, original):
    np.testing.assert_array_equal(read.atomic_numbers, original.atomic_numbers)
    np.testing.assert_array_equal(read.positions, original.positions)
    if original.cell is None:
        assert read.cell is None
    else:
        np.testing.assert_array_equal(read.cell, original.cell)
    assert read.pbc == original.pbc
    assert read.weight == original.weight
    assert read.config_type == original.config_type
    assert read.property_weights == original.property_weights
    assert read.properties.keys() == original.properties.keys()
    for name, value in original.properties.items():
        got = read.properties[name]
        if value is None or isinstance(value, (str, int, float)):
            assert got == value, name
            assert type(got) is type(value), name
        else:
            np.testing.assert_array_equal(got, value)
            assert got.dtype == np.asarray(value).dtype, name


def test_every_field_and_property_round_trips(written):
    """Including a missing cell and pbc, a missing label, an integer, a text
    label and a config type spelled ``None``: the frozen format turns that
    last one into a missing value."""
    backend = HDF5Backend.open(written, key_spec=KeySpecification.from_defaults())
    originals = configurations()
    assert originals[4].config_type == "None"
    assert originals[3].cell is None
    for index, original in enumerate(originals):
        assert_same(backend[index], original)


def test_the_manifest_is_real_json_with_real_types(written):
    payload = json.loads((written / MANIFEST_NAME).read_text())
    assert payload["schema_version"] == SCHEMA_VERSION == 1
    assert payload["shards"] == ["train_0.h5", "train_1.h5", "train_2.h5"]
    assert payload["shard_counts"] == [2, 2, 1]
    assert payload["atomic_numbers"] == [1, 8]
    assert payload["r_max"] == 5.0
    assert payload["e0_provenance"] == "isolated atoms in train.xyz"
    assert payload["key_specification"] == asdict(KeySpecification.from_defaults())
    statistics = payload["statistics"]
    assert statistics["atomic_energies"] == {"1": -13.6, "8": -2041.5}
    assert isinstance(statistics["avg_num_neighbors"], float)
    assert statistics["atomic_numbers"] == [1, 8]
    assert read_manifest(written).statistics == STATISTICS


def test_each_shard_says_what_it_is(written):
    with h5py.File(written / "train_0.h5", "r") as shard:
        assert shard.attrs["format"] == SHARD_FORMAT
        assert shard.attrs["schema_version"] == SCHEMA_VERSION
        assert sorted(shard["configs"].keys()) == ["0", "1"]


def test_nothing_that_reads_a_dataset_evaluates_python_literals():
    """The frozen tree reads its statistics back with ``ast.literal_eval``."""
    source = Path(__file__).resolve().parents[1] / "src" / "mace_core"
    files = [*source.glob("data/**/*.py"), *source.glob("data_spec/**/*.py")]
    assert len(files) > 10
    pattern = re.compile(r"^\s*(import ast|from ast import)|\beval\(", re.M)
    assert not [str(path) for path in files if pattern.search(path.read_text())]


def test_several_processes_write_the_same_dataset(tmp_path):
    keys = KeySpecification.from_defaults()
    items = configurations(7)
    write_shards(items, tmp_path / "one", shard_size=3, key_spec=keys)
    write_shards(items, tmp_path / "three", shard_size=3, key_spec=keys, processes=3)
    one = HDF5Backend.open(tmp_path / "one", key_spec=keys)
    three = HDF5Backend.open(tmp_path / "three", key_spec=keys)
    assert one.metadata() == three.metadata()
    assert three.metadata().shard_counts == [3, 3, 1]
    for index, original in enumerate(items):
        assert_same(three[index], original)
        assert_same(one[index], original)


def test_the_element_table_defaults_to_the_elements_present(tmp_path):
    keys = KeySpecification.from_defaults()
    manifest = write_shards(configurations(2), tmp_path, shard_size=5, key_spec=keys)
    assert manifest.atomic_numbers == [1, 8]
    assert manifest.statistics is None
    assert manifest.shards == ["shard_0.h5"]


def test_a_prepared_directory_is_not_written_over(written):
    with pytest.raises(FileExistsError, match=MANIFEST_NAME):
        write_shards(
            configurations(),
            written,
            shard_size=2,
            key_spec=KeySpecification.from_defaults(),
        )


@pytest.mark.parametrize("shard_size, processes", [(0, 1), (2, 0)])
def test_the_writer_refuses_nonsense_sizes(tmp_path, shard_size, processes):
    with pytest.raises(ValueError):
        write_shards(
            configurations(),
            tmp_path,
            shard_size=shard_size,
            key_spec=KeySpecification.from_defaults(),
            processes=processes,
        )


def _rewrite(directory: Path, **changes) -> None:
    path = directory / MANIFEST_NAME
    payload = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            del payload[key]
        else:
            payload[key] = value
    path.write_text(json.dumps(payload))


def test_a_manifest_from_another_schema_version_is_refused(written):
    _rewrite(written, schema_version=2)
    with pytest.raises(DataBackendError, match="schema_version 2"):
        read_manifest(written)


def test_a_manifest_with_an_unknown_field_is_refused(written):
    _rewrite(written, compression="gzip")
    with pytest.raises(DataBackendError, match="compression"):
        read_manifest(written)


def test_a_manifest_missing_a_field_is_refused(written):
    _rewrite(written, shard_counts=None)
    with pytest.raises(DataBackendError, match="shard_counts"):
        read_manifest(written)


def test_shard_lists_that_disagree_are_refused(written):
    _rewrite(written, shard_counts=[2, 3])
    with pytest.raises(DataBackendError, match="3 shards are listed with 2 counts"):
        read_manifest(written)


def test_a_directory_without_a_manifest_is_named(tmp_path):
    with pytest.raises(DataBackendError) as caught:
        read_manifest(tmp_path)
    assert caught.value.source.endswith(MANIFEST_NAME)


def test_the_manifest_path_opens_the_dataset(written):
    keys = KeySpecification.from_defaults()
    assert len(HDF5Backend.open(written / MANIFEST_NAME, key_spec=keys)) == 5


def test_an_unknown_property_kind_is_an_error_not_a_guess(written):
    with h5py.File(written / "train_0.h5", "a") as shard:
        shard["configs/0/properties/energy"].attrs["kind"] = "complex"
    backend = HDF5Backend.open(written, key_spec=KeySpecification.from_defaults())
    with pytest.raises(DataBackendError, match="complex"):
        backend[0]
