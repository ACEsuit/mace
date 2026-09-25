"""``format="auto"`` tells v2 shards from the legacy layout from LMDB, and a
source that more than one backend or no backend claims is a hard error.
"""

from __future__ import annotations

import shutil

import pytest
from ase import Atoms
from ase.io import write
from mace_core.data import (
    AmbiguousFormatError,
    KeySpecification,
    available_backends,
    open_dataset,
)
from mace_core.data.backends.hdf5 import HDF5Backend
from mace_core.data.backends.hdf5_legacy import LegacyHDF5Backend
from mace_core.data.backends.lmdb import ASEDBBackend
from mace_core.data.backends.xyz import XYZBackend
from mace_core.data.registry import SNIFF_PRIORITY
from mace_core.data_spec import MANIFEST_NAME
from mace_core_data_fixtures import (
    write_legacy_directory,
    write_legacy_file,
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


def test_the_three_backends_are_registered_and_load():
    found = available_backends()
    for name in ("hdf5", "hdf5-legacy", "lmdb"):
        assert found[name].loaded, found[name].reason
    assert found["hdf5"].factory is HDF5Backend
    assert found["hdf5-legacy"].factory is LegacyHDF5Backend
    assert found["lmdb"].factory is ASEDBBackend


def test_the_sniff_priority_is_the_declared_one():
    assert SNIFF_PRIORITY == ("xyz", "hdf5", "hdf5-legacy", "lmdb")


@pytest.mark.parametrize(
    "fixture, backend",
    [
        ("written", HDF5Backend),
        ("legacy_file", LegacyHDF5Backend),
        ("legacy_directory", LegacyHDF5Backend),
        ("lmdb_source", ASEDBBackend),
    ],
)
def test_auto_opens_each_format_with_its_backend(request, key_spec, fixture, backend):
    source = request.getfixturevalue(fixture)
    assert type(open_dataset(source, key_spec=key_spec)) is backend


def test_auto_opens_a_v2_dataset_by_its_manifest(written, key_spec):
    assert type(open_dataset(written / MANIFEST_NAME, key_spec=key_spec)) is HDF5Backend


def test_a_v2_shard_on_its_own_is_claimed_by_nobody(written, key_spec):
    """A shard is part of a dataset, not a dataset. The legacy backend must not
    claim it just because it is an ``.h5`` file."""
    with pytest.raises(AmbiguousFormatError, match="no registered data backend"):
        open_dataset(written / "train_0.h5", key_spec=key_spec)


def test_the_xyz_backend_still_claims_its_files(tmp_path, key_spec):
    path = tmp_path / "frames.xyz"
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=[5, 5, 5], pbc=True)
    atoms.info["REF_energy"] = -1.0
    write(path, [atoms], format="extxyz")
    assert type(open_dataset(path, key_spec=key_spec)) is XYZBackend


def test_a_directory_two_backends_claim_is_refused_naming_both(
    tmp_path, legacy_file, lmdb_source, key_spec
):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    shutil.copy(legacy_file, mixed / "legacy.h5")
    shutil.copy(lmdb_source.split(":")[-1], mixed / "rows.aselmdb")
    with pytest.raises(AmbiguousFormatError) as caught:
        open_dataset(mixed, key_spec=key_spec)
    assert "'hdf5-legacy'" in str(caught.value)
    assert "'lmdb'" in str(caught.value)


def test_a_source_nobody_claims_is_refused(tmp_path, key_spec):
    path = tmp_path / "notes.txt"
    path.write_text("nothing here")
    with pytest.raises(AmbiguousFormatError, match="no registered data backend"):
        open_dataset(path, key_spec=key_spec)
    with pytest.raises(AmbiguousFormatError):
        open_dataset(tmp_path / "missing.aselmdb", key_spec=key_spec)


def test_a_named_format_skips_the_sniff(legacy_file, key_spec):
    backend = open_dataset(legacy_file, format="hdf5-legacy", key_spec=key_spec)
    assert type(backend) is LegacyHDF5Backend


def test_opening_the_legacy_layout_warns(legacy_file, key_spec):
    with pytest.warns(FutureWarning, match="legacy HDF5 layout"):
        LegacyHDF5Backend.open(legacy_file, key_spec=key_spec)
