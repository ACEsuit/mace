"""Tiny datasets in every on-disk format the data backends read.

Each fixture writes its format by hand, from the layout the format defines,
rather than through the backend under test. A reader tested only against its
own writer agrees with itself and nothing else. The frozen tree's own writers
are exercised against these readers in ``tests/parity``.

A module of its own, with a name no other package's tests use, because the
package test directories are collected in one session and are not packages:
two modules named alike would be one module.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import h5py
import lmdb
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db.row import AtomsRow
from mace_core.data import Configuration, DatasetStatistics, KeySpecification
from mace_core.data_spec import write_shards

STATISTICS = DatasetStatistics(
    atomic_energies={1: -13.6, 8: -2041.5},
    avg_num_neighbors=1.75,
    mean=0.125,
    std=0.875,
    atomic_numbers=[1, 8],
    r_max=5.0,
)


def configurations(count: int = 5) -> list[Configuration]:
    """Water-sized structures with every kind of value a property can hold."""
    generator = np.random.default_rng(3)
    items = []
    for index in range(count):
        atoms = 2 + index % 3
        items.append(
            Configuration(
                atomic_numbers=np.array([8] + [1] * (atoms - 1)),
                positions=generator.normal(size=(atoms, 3)),
                properties={
                    "energy": -10.0 - index,
                    "forces": generator.normal(size=(atoms, 3)),
                    "stress": generator.normal(size=6) if index % 2 == 0 else None,
                    "dipole": None,
                    "total_spin": index,
                    "label": f"frame {index}",
                },
                property_weights={
                    "energy": 1.0,
                    "forces": 0.5 * index,
                    "stress": 1.0 if index % 2 == 0 else 0.0,
                    "dipole": 0.0,
                    "total_spin": 1.0,
                    "label": 1.0,
                },
                cell=np.eye(3) * (8.0 + index) if index != 3 else None,
                pbc=(True, True, index % 2 == 0) if index != 3 else None,
                weight=1.0 + index,
                config_type="None" if index == 4 else f"type{index % 2}",
            )
        )
    return items


def write_prepared(tmp_path: Path) -> Path:
    """Five structures as v2 shards of (2, 2, 1), with statistics."""
    directory = tmp_path / "prepared"
    write_shards(
        configurations(),
        directory,
        shard_size=2,
        key_spec=KeySpecification.from_defaults(),
        prefix="train",
        statistics=STATISTICS,
        e0_provenance="isolated atoms in train.xyz",
    )
    return directory


def _write_value(value):
    """The frozen tree's ``write_value``: the value, or the text ``None``."""
    return value if value is not None else "None"


def write_legacy_group(group: h5py.Group, items: list[Configuration]) -> None:
    """``config_{j}`` subgroups in the frozen tree's layout
    (``mace/data/utils.py:594``)."""
    for index, item in enumerate(items):
        entry = group.create_group(f"config_{index}")
        entry["atomic_numbers"] = _write_value(item.atomic_numbers)
        entry["positions"] = _write_value(item.positions)
        properties = entry.create_group("properties")
        for name, value in item.properties.items():
            properties[name] = _write_value(value)
        entry["cell"] = _write_value(item.cell)
        entry["pbc"] = _write_value(item.pbc)
        entry["weight"] = _write_value(item.weight)
        weights = entry.create_group("property_weights")
        for name, value in item.property_weights.items():
            weights[name] = _write_value(value)
        entry["config_type"] = _write_value(item.config_type)


def legacy_items() -> list[Configuration]:
    """The fixture structures without the text label, which the frozen writer
    would store as bytes and its reader give back as a string either way."""
    items = configurations()
    for item in items:
        del item.properties["label"]
        del item.property_weights["label"]
    return items


def write_legacy_file(tmp_path: Path) -> Path:
    """One legacy file whose groups hold (2, 2, 1) structures. The frozen
    reader reports six."""
    path = tmp_path / "legacy.h5"
    items = legacy_items()
    with h5py.File(path, "w") as handle:
        handle.attrs["drop_last"] = True
        for group, run in enumerate((items[0:2], items[2:4], items[4:5])):
            write_legacy_group(handle.create_group(f"config_batch_{group}"), run)
    return path


def write_legacy_directory(tmp_path: Path) -> Path:
    """A sharded legacy directory, the layout ``mace_prepare_data`` writes:
    one ``config_batch_0`` per file, beside a stringified statistics file."""
    directory = tmp_path / "prep_train"
    directory.mkdir()
    items = legacy_items()
    for index, run in enumerate((items[0:2], items[2:4], items[4:5])):
        with h5py.File(directory / f"train_{index}.h5", "w") as handle:
            handle.attrs["drop_last"] = False
            write_legacy_group(handle.create_group("config_batch_0"), run)
    (tmp_path / "prep_statistics.json").write_text(
        json.dumps(
            {
                "atomic_energies": "{1: np.float64(-13.6), 8: -2041.5}",
                "avg_num_neighbors": 1.75,
                "mean": 0.125,
                "std": 0.875,
                "atomic_numbers": "[1, 8]",
                "r_max": 5.0,
            }
        )
    )
    return directory


def lmdb_row(atoms: Atoms) -> dict:
    """A row as a fairchem database stores it: ase's row fields, as JSON."""
    row = AtomsRow(atoms)
    stored = {
        key: value
        for key, value in row.__dict__.items()
        if not key.startswith("_") and key != "id"
    }
    stored["cell"] = np.asarray(stored["cell"])
    if atoms.info:
        stored["key_value_pairs"] = dict(atoms.info)
    return stored


def _json(value):
    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item))

    return zlib.compress(json.dumps(value, default=default).encode())


def write_lmdb(path: Path, rows: list[dict]) -> None:
    environment = lmdb.open(str(path), subdir=False, map_size=2**24)
    with environment.begin(write=True) as txn:
        for identifier, row in enumerate(rows, start=1):
            txn.put(str(identifier).encode("ascii"), _json(row))
        txn.put(b"nextid", _json(len(rows) + 1))
    environment.close()


def lmdb_atoms(count: int, seed: int) -> list[Atoms]:
    """Structures whose labels live in a single-point calculator, the way a
    fairchem corpus stores them."""
    generator = np.random.default_rng(seed)
    frames = []
    for _ in range(count):
        atoms = Atoms(
            "OH2",
            positions=generator.normal(scale=0.5, size=(3, 3)),
            cell=np.eye(3) * 6.0,
            pbc=True,
        )
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=float(generator.uniform(-15.0, -5.0)),
            forces=generator.normal(size=(3, 3)),
            stress=generator.normal(size=6),
        )
        frames.append(atoms)
    return frames


def write_lmdb_source(tmp_path: Path) -> str:
    """Two directories of two databases each plus one loose database, joined
    with colons: 2 + 2 + 1 + 1 + 3 = 9 rows."""
    parts = []
    seed = 0
    for folder, counts in (("a", (2, 2)), ("b", (1, 1))):
        directory = tmp_path / folder
        directory.mkdir()
        for index, count in enumerate(counts):
            seed += 1
            rows = [lmdb_row(atoms) for atoms in lmdb_atoms(count, seed)]
            write_lmdb(directory / f"data_{index}.aselmdb", rows)
        parts.append(str(directory))
    loose = tmp_path / "loose.aselmdb"
    write_lmdb(loose, [lmdb_row(atoms) for atoms in lmdb_atoms(3, 99)])
    parts.append(str(loose))
    return ":".join(parts)
