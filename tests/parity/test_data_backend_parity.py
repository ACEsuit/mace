"""The shard and database backends, against the frozen tree's own files.

Every dataset here is written by the frozen tree: its preprocessing script, its
HDF5 writer and its vendored LMDB database. The frozen readers read them back
as their configurations, with graph construction stubbed out so that what is
compared is the parsed structure, and the rewrite's backends have to read the
same structures.

Two differences are asserted rather than tolerated. A legacy file whose last
group is short has a length the frozen reader overstates, and the rewrite
counts it. The frozen statistics file is stringified Python, which the frozen
tree evaluates, and the rewrite parses to the same numbers without evaluating
anything.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from mace_core.data import KeySpecification, open_dataset
from mace_core.data.backends.hdf5_legacy import LegacyHDF5Backend
from mace_core.data.backends.lmdb import ASEDBBackend

from tests.helpers import preprocess_data, run_mace_train

pytestmark = pytest.mark.filterwarnings("ignore:.*legacy HDF5 layout:FutureWarning")


class ParsedOnly:
    """Stands in for ``AtomicData`` so a frozen reader returns its
    configuration rather than a graph."""

    @staticmethod
    def from_config(config, **_):
        return config


def frames(count: int, seed: int) -> list[Atoms]:
    generator = np.random.default_rng(seed)
    water = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    made = []
    for _ in range(count):
        atoms = water.copy()
        atoms.positions += generator.normal(scale=0.05, size=(3, 3))
        atoms.info["REF_energy"] = float(generator.normal())
        atoms.arrays["REF_forces"] = generator.normal(size=(3, 3))
        atoms.info["REF_stress"] = generator.normal(size=6)
        made.append(atoms)
    return made


def assert_same(ours, theirs, where):
    np.testing.assert_array_equal(ours.atomic_numbers, theirs.atomic_numbers)
    np.testing.assert_array_equal(ours.positions, theirs.positions)
    np.testing.assert_array_equal(ours.cell, theirs.cell)
    assert tuple(ours.pbc) == tuple(bool(flag) for flag in theirs.pbc), where
    assert ours.weight == theirs.weight, where
    assert ours.config_type == theirs.config_type, where
    shared = set(theirs.properties)
    assert shared <= set(ours.properties), where
    for name in shared:
        mine, frozen = ours.properties[name], theirs.properties[name]
        if frozen is None:
            assert mine is None, (where, name)
        else:
            np.testing.assert_array_equal(np.asarray(mine), np.asarray(frozen))
        assert ours.property_weights[name] == theirs.property_weights[name], (
            where,
            name,
        )


# ---------------------------------------------------------------------------
# HDF5, written by the frozen preprocessing script
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", name="prepared")
def fixture_prepared(tmp_path_factory):
    """A real ``mace_prepare_data`` run: two train shards, statistics."""
    directory = tmp_path_factory.mktemp("prepared")
    isolated = [Atoms("O", cell=np.eye(3) * 6.0), Atoms("H", cell=np.eye(3) * 6.0)]
    for atoms, energy in zip(isolated, (-2041.5, -13.6), strict=True):
        atoms.info["REF_energy"] = energy
        atoms.info["config_type"] = "IsolatedAtom"
    write(directory / "train.xyz", isolated + frames(9, 1), format="extxyz")
    run_mace_train(
        {
            "train_file": directory / "train.xyz",
            "r_max": 5.0,
            "num_process": 2,
            "valid_fraction": 0.2,
            "h5_prefix": directory / "prep_",
            "compute_statistics": None,
            "seed": 3,
        },
        script=preprocess_data,
    )
    return directory


def frozen_hdf5(path: Path):
    from mace.data.hdf5_dataset import HDF5Dataset
    from mace.tools import AtomicNumberTable

    return HDF5Dataset(
        str(path),
        r_max=5.0,
        z_table=AtomicNumberTable([1, 8]),
        atomic_dataclass=ParsedOnly,
    )


def test_the_prepared_shards_read_as_the_frozen_reader_reads_them(prepared, isolated):
    train = prepared / "prep_train"
    ours = LegacyHDF5Backend.open(train, key_spec=KeySpecification.from_defaults())
    readers = [frozen_hdf5(path) for path in sorted(train.glob("*.h5"))]
    theirs = [reader[index] for reader in readers for index in range(len(reader))]
    assert len(ours) == len(theirs) > 0
    for index, frozen in enumerate(theirs):
        assert_same(ours[index], frozen, index)


def test_auto_takes_the_prepared_directories_for_the_legacy_layout(prepared):
    keys = KeySpecification.from_defaults()
    for split in ("prep_train", "prep_val"):
        assert type(open_dataset(prepared / split, key_spec=keys)) is LegacyHDF5Backend


def test_the_prepared_statistics_parse_to_what_the_frozen_tree_evaluates(prepared):
    """``mace/cli/run_train.py:319-334`` evaluates these two fields."""
    payload = json.loads((prepared / "prep_statistics.json").read_text())
    assert isinstance(payload["atomic_energies"], str)
    energies = ast.literal_eval(payload["atomic_energies"])
    numbers = ast.literal_eval(payload["atomic_numbers"])

    ours = LegacyHDF5Backend.open(
        prepared / "prep_train", key_spec=KeySpecification.from_defaults()
    ).statistics()
    assert ours is not None
    assert ours.atomic_energies == {int(z): float(e) for z, e in energies.items()}
    assert ours.atomic_numbers == numbers
    assert ours.avg_num_neighbors == payload["avg_num_neighbors"]
    assert ours.mean == payload["mean"]
    assert ours.std == payload["std"]
    assert ours.r_max == payload["r_max"]


def test_a_short_last_group_is_counted_where_the_frozen_reader_overstates(
    tmp_path, isolated
):
    """Groups of (2, 2, 1) written by the frozen writer hold five structures.
    The frozen reader reports six, and the sixth does not exist."""
    from mace.data import KeySpecification as LegacyKeys
    from mace.data import config_from_atoms
    from mace.data.utils import save_configurations_as_HDF5

    configs = [
        config_from_atoms(atoms, key_specification=LegacyKeys.from_defaults())
        for atoms in frames(5, 2)
    ]
    path = tmp_path / "uneven.h5"
    with h5py.File(path, "w") as handle:
        for group, run in ((2, configs[4:5]), (1, configs[2:4]), (0, configs[0:2])):
            save_configurations_as_HDF5(run, None, handle)
            if group:
                handle.move("config_batch_0", f"config_batch_{group}")

    frozen = frozen_hdf5(path)
    assert len(frozen) == 6
    with pytest.raises(KeyError):
        frozen[5]

    ours = LegacyHDF5Backend.open(path, key_spec=KeySpecification.from_defaults())
    assert len(ours) == 5
    for index in range(5):
        assert_same(ours[index], frozen[index], index)


# ---------------------------------------------------------------------------
# LMDB, written by the vendored fairchem database
# ---------------------------------------------------------------------------


@pytest.fixture(name="databases")
def fixture_databases(tmp_path):
    """Three directories of two databases of two rows, labels on a
    single-point calculator, as the frozen tree's own test writes them."""
    from mace.tools.fairchem_dataset.lmdb_dataset_tools import LMDBDatabase

    generator = np.random.default_rng(7)
    directories = []
    for folder in range(3):
        directory = tmp_path / f"folder_{folder}"
        directory.mkdir()
        for index in range(2):
            database = LMDBDatabase(directory / f"data_{index}.aselmdb", readonly=False)
            for atoms in frames(2, int(generator.integers(1 << 30))):
                labelled = Atoms(
                    atoms.numbers, atoms.positions, cell=atoms.cell, pbc=atoms.pbc
                )
                labelled.calc = SinglePointCalculator(
                    labelled,
                    energy=atoms.info["REF_energy"],
                    forces=atoms.arrays["REF_forces"],
                    stress=atoms.info["REF_stress"],
                )
                database.write(labelled)
            database.close()
        directories.append(str(directory))
    return ":".join(directories)


def test_lmdb_rows_read_as_the_frozen_reader_reads_them(
    databases, isolated, monkeypatch
):
    import mace.data.lmdb_dataset as frozen_module
    from mace.tools import AtomicNumberTable

    # LMDB refuses to open a file twice in one process, so each side reads
    # everything and lets go before the other opens.
    backend = ASEDBBackend.open(databases, key_spec=KeySpecification.from_defaults())
    ours = list(backend.iter_range())
    backend.close()

    monkeypatch.setattr(frozen_module, "AtomicData", ParsedOnly)
    reader = frozen_module.LMDBDataset(
        databases, r_max=5.0, z_table=AtomicNumberTable([1, 8])
    )
    theirs = [reader[index] for index in range(len(reader))]
    del reader

    assert len(ours) == len(theirs) == 12
    for index, frozen in enumerate(theirs):
        assert frozen.properties["energy"] is not None
        assert frozen.properties["forces"] is not None
        assert_same(ours[index], frozen, index)


def test_auto_takes_a_colon_joined_source_for_lmdb(databases):
    keys = KeySpecification.from_defaults()
    assert type(open_dataset(databases, key_spec=keys)) is ASEDBBackend
