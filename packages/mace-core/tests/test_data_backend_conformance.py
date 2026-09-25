"""The HDF5, legacy HDF5 and LMDB backends, through the one harness.

Each backend passes the checks every backend passes, and then the three things
the frozen readers get wrong: a length that counts structures that do not
exist, a bad read that turns into ``None`` in the batch, and statistics kept as
stringified Python.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import h5py
import numpy as np
import pytest
from mace_core.data import (
    Configuration,
    DataBackendError,
    KeySpecification,
    data_backend_conformance,
)
from mace_core.data.backends.hdf5 import HDF5Backend
from mace_core.data.backends.hdf5_legacy import (
    LegacyHDF5Backend,
    read_legacy_statistics,
)
from mace_core.data.backends.lmdb import ASEDBBackend
from mace_core.elements import DefaultKeys
from mace_core_data_fixtures import (
    STATISTICS,
    lmdb_atoms,
    lmdb_row,
    write_legacy_directory,
    write_legacy_file,
    write_lmdb,
    write_lmdb_source,
    write_prepared,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*legacy HDF5 layout:FutureWarning")


@pytest.fixture
def key_spec():
    return KeySpecification.from_defaults()


@pytest.fixture
def written(tmp_path):
    return write_prepared(tmp_path)


@pytest.fixture
def legacy_file(tmp_path):
    return write_legacy_file(tmp_path)


@pytest.fixture
def legacy_directory(tmp_path):
    return write_legacy_directory(tmp_path)


@pytest.fixture
def lmdb_source(tmp_path):
    return write_lmdb_source(tmp_path)


@pytest.fixture(params=["hdf5", "hdf5-legacy file", "hdf5-legacy directory", "lmdb"])
def opened(request, key_spec):
    """Each backend over its fixture, with the number of structures it holds."""
    if request.param == "hdf5":
        return HDF5Backend.open(
            request.getfixturevalue("written"), key_spec=key_spec
        ), 5
    if request.param == "hdf5-legacy file":
        source = request.getfixturevalue("legacy_file")
        return LegacyHDF5Backend.open(source, key_spec=key_spec), 5
    if request.param == "hdf5-legacy directory":
        source = request.getfixturevalue("legacy_directory")
        return LegacyHDF5Backend.open(source, key_spec=key_spec), 5
    source = request.getfixturevalue("lmdb_source")
    return ASEDBBackend.open(source, key_spec=key_spec), 9


def test_every_backend_passes_the_conformance_harness(opened):
    backend, length = opened
    data_backend_conformance(backend, expected_length=length)


def test_every_backend_yields_configurations(opened):
    backend, length = opened
    assert all(type(item) is Configuration for item in backend.iter_range())
    assert all(type(backend[index]) is Configuration for index in range(length))


def test_every_backend_survives_a_pickle_after_it_has_read(opened):
    """The handles opened by a read are what cannot be pickled."""
    backend, length = opened
    before = [backend[index].positions for index in range(length)]
    revived = pickle.loads(pickle.dumps(backend))
    for index in range(length):
        np.testing.assert_array_equal(revived[index].positions, before[index])


def test_the_head_is_the_one_asked_for(opened, key_spec):
    backend, _ = opened
    reopened = type(backend).open(backend.source, key_spec=key_spec, head="water")
    assert {item.head for item in reopened.iter_range()} == {"water"}


# ---------------------------------------------------------------------------
# Uneven shards
# ---------------------------------------------------------------------------


def test_uneven_shards_are_counted_exactly(
    written, legacy_file, legacy_directory, key_spec
):
    """Shards of (2, 2, 1) hold five structures. The frozen reader multiplies
    the group count by the first group's size and reports six."""
    v2 = HDF5Backend.open(written, key_spec=key_spec)
    assert v2.metadata().shard_counts == [2, 2, 1]
    assert len(v2) == 5
    assert len(LegacyHDF5Backend.open(legacy_file, key_spec=key_spec)) == 5
    assert len(LegacyHDF5Backend.open(legacy_directory, key_spec=key_spec)) == 5


def test_the_legacy_file_reads_its_groups_in_order(legacy_file, key_spec):
    backend = LegacyHDF5Backend.open(legacy_file, key_spec=key_spec)
    assert [item.properties["energy"] for item in backend.iter_range()] == [
        -10.0,
        -11.0,
        -12.0,
        -13.0,
        -14.0,
    ]


# ---------------------------------------------------------------------------
# A bad read raises, naming the source and the index
# ---------------------------------------------------------------------------


def test_a_damaged_v2_structure_raises_naming_its_shard_and_index(written, key_spec):
    with h5py.File(written / "train_1.h5", "a") as shard:
        del shard["configs/1/positions"]
    backend = HDF5Backend.open(written, key_spec=key_spec)
    backend[2]
    with pytest.raises(DataBackendError) as caught:
        backend[3]
    assert caught.value.index == 3
    assert caught.value.source.endswith("train_1.h5")
    with pytest.raises(DataBackendError):
        list(backend.iter_range())


def test_a_v2_shard_that_is_not_hdf5_raises_naming_it(written, key_spec):
    (written / "train_2.h5").write_bytes(b"not an hdf5 file")
    backend = HDF5Backend.open(written, key_spec=key_spec)
    with pytest.raises(DataBackendError) as caught:
        backend[4]
    assert caught.value.source.endswith("train_2.h5")


def test_a_missing_v2_shard_is_refused_at_open(written, key_spec):
    (written / "train_2.h5").unlink()
    with pytest.raises(DataBackendError, match="does not exist"):
        HDF5Backend.open(written, key_spec=key_spec)


def test_a_damaged_legacy_structure_raises_naming_its_file_and_index(
    legacy_directory, key_spec
):
    with h5py.File(legacy_directory / "train_1.h5", "a") as handle:
        del handle["config_batch_0/config_0/positions"]
    backend = LegacyHDF5Backend.open(legacy_directory, key_spec=key_spec)
    with pytest.raises(DataBackendError) as caught:
        backend[2]
    assert caught.value.index == 2
    assert caught.value.source.endswith("train_1.h5")


def test_a_file_that_is_not_the_legacy_layout_is_refused(written, key_spec):
    with pytest.raises(DataBackendError, match="not the legacy layout"):
        LegacyHDF5Backend.open(written / "train_0.h5", key_spec=key_spec)


def test_a_damaged_lmdb_row_raises_naming_its_database_and_index(tmp_path, key_spec):
    path = tmp_path / "damaged.aselmdb"
    write_lmdb(path, [lmdb_row(atoms) for atoms in lmdb_atoms(3, 5)])
    import lmdb

    environment = lmdb.open(str(path), subdir=False, map_size=2**24)
    with environment.begin(write=True) as txn:
        txn.put(b"2", b"not a compressed row")
    environment.close()

    backend = ASEDBBackend.open(path, key_spec=key_spec)
    backend[0]
    with pytest.raises(DataBackendError) as caught:
        backend[1]
    assert caught.value.index == 1
    assert caught.value.source == str(path)
    assert "row 2" in str(caught.value)
    backend[2]


def test_a_missing_lmdb_source_is_named(tmp_path, key_spec):
    with pytest.raises(DataBackendError) as caught:
        ASEDBBackend.open(
            f"{tmp_path}/a.aselmdb:{tmp_path}/b.aselmdb", key_spec=key_spec
        )
    assert caught.value.source.endswith("a.aselmdb")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_the_v2_statistics_are_the_ones_written(written, key_spec):
    assert HDF5Backend.open(written, key_spec=key_spec).statistics() == STATISTICS


def test_the_legacy_statistics_file_is_read_beside_its_shards(
    legacy_directory, key_spec
):
    """``<prefix>train/`` sits beside ``<prefix>statistics.json``, whose
    energies and element table are stringified Python."""
    statistics = LegacyHDF5Backend.open(
        legacy_directory, key_spec=key_spec
    ).statistics()
    assert statistics == STATISTICS
    assert type(statistics.atomic_energies[1]) is float


def test_the_legacy_statistics_parse_follows_a_named_energies_file(tmp_path):
    energies = tmp_path / "e0s.json"
    energies.write_text(json.dumps("{1: -13.6, 8: -2041.5}"))
    path = tmp_path / "statistics.json"
    path.write_text(
        json.dumps(
            {
                "atomic_energies": str(energies),
                "avg_num_neighbors": 1.75,
                "mean": 0.125,
                "std": 0.875,
                "atomic_numbers": [1, 8],
                "r_max": 5.0,
            }
        )
    )
    assert read_legacy_statistics(path) == STATISTICS


@pytest.mark.parametrize(
    "energies",
    ["{1: -13.6, 8: __import__('os')}", "{1: -13.6, 8: [1, 2]}", "[1, 8]", "{1 -13.6}"],
)
def test_the_legacy_statistics_parse_accepts_only_numbers(tmp_path, energies):
    """Where the frozen tree calls ``ast.literal_eval``, only the literal it
    writes is accepted."""
    path = tmp_path / "statistics.json"
    path.write_text(
        json.dumps(
            {
                "atomic_energies": energies,
                "avg_num_neighbors": 1.0,
                "mean": 0.0,
                "std": 1.0,
                "atomic_numbers": "[1, 8]",
                "r_max": 5.0,
            }
        )
    )
    with pytest.raises(DataBackendError, match=r"statistics\.json"):
        read_legacy_statistics(path)


def test_lmdb_carries_no_statistics(lmdb_source, key_spec):
    backend = ASEDBBackend.open(lmdb_source, key_spec=key_spec)
    assert backend.statistics() is None
    assert backend.metadata() is None


# ---------------------------------------------------------------------------
# LMDB rows
# ---------------------------------------------------------------------------


def test_an_lmdb_row_s_calculator_labels_become_its_properties(tmp_path, key_spec):
    frames = lmdb_atoms(2, 11)
    path = tmp_path / "labels.aselmdb"
    write_lmdb(path, [lmdb_row(atoms) for atoms in frames])
    backend = ASEDBBackend.open(path, key_spec=key_spec)
    for atoms, item in zip(frames, backend.iter_range(), strict=True):
        results = atoms.calc.results
        assert item.properties["energy"] == pytest.approx(results["energy"], abs=0)
        np.testing.assert_array_equal(item.properties["forces"], results["forces"])
        np.testing.assert_array_equal(item.properties["stress"], results["stress"])
        assert item.property_weights["energy"] == 1.0
        np.testing.assert_array_equal(item.positions, atoms.positions)
        assert item.pbc == (True, True, True)


def test_lmdb_reads_the_parts_of_a_source_in_sorted_order(lmdb_source, key_spec):
    """The vendored reader sorts the colon-separated parts, then a directory's
    databases, so an index means one structure in both readers."""
    parts = lmdb_source.split(":")
    shuffled = ":".join(reversed(parts))
    forwards = ASEDBBackend.open(lmdb_source, key_spec=key_spec)
    backwards = ASEDBBackend.open(shuffled, key_spec=key_spec)
    for index in range(len(forwards)):
        np.testing.assert_array_equal(
            forwards[index].positions, backwards[index].positions
        )


def test_a_deleted_lmdb_row_is_skipped(tmp_path, key_spec):
    path = tmp_path / "deleted.aselmdb"
    frames = lmdb_atoms(3, 21)
    write_lmdb(path, [lmdb_row(atoms) for atoms in frames])
    import zlib

    import lmdb

    environment = lmdb.open(str(path), subdir=False, map_size=2**24)
    with environment.begin(write=True) as txn:
        txn.delete(b"2")
        txn.put(b"deleted_ids", zlib.compress(json.dumps([2]).encode()))
    environment.close()
    backend = ASEDBBackend.open(path, key_spec=key_spec)
    assert len(backend) == 2
    np.testing.assert_array_equal(backend[1].positions, frames[2].positions)


# ---------------------------------------------------------------------------
# No framework
# ---------------------------------------------------------------------------

PROBE = """
import json, sys, warnings
warnings.simplefilter("ignore")
from mace_core.data import KeySpecification, open_dataset
keys = KeySpecification.from_defaults()
for source in sys.argv[1:]:
    open_dataset(source, key_spec=keys)[0]
print(json.dumps(sorted({m.split(".", 1)[0] for m in sys.modules})))
"""


def test_reading_every_format_imports_no_framework(written, legacy_file, lmdb_source):
    """In a fresh interpreter, since this session may have imported anything."""
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(written), str(legacy_file), lmdb_source],
        capture_output=True,
        text=True,
        check=True,
    )
    reached = set(json.loads(result.stdout.splitlines()[-1]))
    assert not reached & {"torch", "jax", "jaxlib", "e3nn", "mace"}
    assert {"h5py", "lmdb"} <= reached


def test_the_promotion_writes_the_default_file_keys():
    """Guards the key names the promotion copies to against the key table."""
    assert DefaultKeys.ENERGY.value == "REF_energy"
    assert DefaultKeys.FORCES.value == "REF_forces"
    assert DefaultKeys.STRESS.value == "REF_stress"
