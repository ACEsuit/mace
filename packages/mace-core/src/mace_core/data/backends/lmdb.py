"""The LMDB backend: fairchem-style ASE databases, read-only.

OMat24 and OMol25 are published in this format, so it is read as it is rather
than converted. A database is one LMDB file whose values are ASE database rows,
JSON compressed with zlib, keyed by row id. Two more keys hold bookkeeping:
``nextid``, one past the last id written, and ``deleted_ids``.

A source is a database file (``.aselmdb`` or ``.lmdb``), a directory of them,
or several of either joined with ``:``. The parts are read in sorted order, and
a directory's databases in sorted order within it, which is the order the
frozen tree's vendored reader uses (``mace/tools/fairchem_dataset``), so an
index means the same structure in both.

A row becomes a :class:`~mace_core.data.configuration.Configuration` the way an
XYZ structure does, through :func:`~mace_core.data.xyz.configuration_from_atoms`
and the key specification. Before that, the energy, forces and stress a row
stores as calculator results are copied to the default file keys
(``REF_energy``, ``REF_forces``, ``REF_stress``), as the frozen reader does
(``mace/data/lmdb_dataset.py:39-45``). That is what makes a row parse to the
same configuration as the same structure written to an XYZ file.

A row that cannot be read raises :class:`~mace_core.data.backend.DataBackendError`
naming the database and the index. The frozen reader prints the error and puts
``None`` into the batch (``mace/data/lmdb_dataset.py:30-36``).
"""

from __future__ import annotations

import bisect
import json
import os
import zlib
from collections.abc import Iterator
from itertools import accumulate
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db.row import AtomsRow

from mace_core.data.backend import DataBackendError, DatasetManifest, DatasetStatistics
from mace_core.data.configuration import DEFAULT_HEAD, Configuration
from mace_core.data.keys import KeySpecification
from mace_core.data.xyz import configuration_from_atoms
from mace_core.elements import DefaultKeys

__all__ = ["ASEDBBackend"]

#: The suffixes a database file carries.
SUFFIXES = (".aselmdb", ".lmdb")

#: Stored by ase at the top of a row rather than in its info.
_CALCULATOR_RESULTS = ("energy", "forces", "stress", "free_energy")

#: Where the three labels the frozen reader promotes are copied to.
_PROMOTED = {
    "energy": ("info", DefaultKeys.ENERGY.value),
    "forces": ("arrays", DefaultKeys.FORCES.value),
    "stress": ("info", DefaultKeys.STRESS.value),
}


def _decode_arrays(value: Any) -> Any:
    """Turn ase's ``{"__ndarray__": [shape, dtype, flat]}`` blobs into arrays."""
    if isinstance(value, dict):
        if "__ndarray__" in value:
            shape, dtype, flat = value["__ndarray__"]
            return np.asarray(flat, dtype=dtype).reshape(shape)
        return {key: _decode_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_arrays(item) for item in value]
    return value


def _stored(txn: lmdb.Transaction, key: str) -> Any:
    raw = txn.get(key.encode("ascii"))
    return None if raw is None else json.loads(zlib.decompress(raw))


def _databases(source: str) -> list[Path]:
    paths: list[Path] = []
    for part in sorted(source.split(":")):
        path = Path(part)
        if path.is_dir():
            found = sorted(
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in SUFFIXES
            )
            if not found:
                raise DataBackendError(
                    part, reason=f"the directory holds no {' or '.join(SUFFIXES)} file"
                )
            paths.extend(found)
        elif path.is_file():
            paths.append(path)
        else:
            raise DataBackendError(part, reason="no such file or directory")
    return paths


#: One environment per database per process. LMDB refuses to open a file
#: twice in one process, which a backend and its unpickled copy would do, and
#: an environment must not cross a fork, hence the process id in the key.
_ENVIRONMENTS: dict[tuple[int, str], lmdb.Environment] = {}


def _environment(path: Path) -> lmdb.Environment:
    key = (os.getpid(), str(path.resolve()))
    environment = _ENVIRONMENTS.get(key)
    if environment is None:
        environment = lmdb.open(
            str(path), subdir=False, readonly=True, lock=False, meminit=False
        )
        _ENVIRONMENTS[key] = environment
    return environment


def atoms_from_row(row: dict[str, Any], row_id: int = 0) -> Atoms:
    """One stored row as ase atoms, with its labels where an XYZ file has them.

    The row's key-value pairs and data land in ``atoms.info``, extra per-atom
    arrays in ``atoms.arrays``, the calculator results on a single-point
    calculator, and energy, forces and stress also under the default file
    keys.
    """
    fields = dict(row, id=row_id)
    parsed = AtomsRow(_decode_arrays(fields))
    atoms = parsed.toatoms()

    data = _decode_arrays(dict(parsed.data)) if parsed.get("data") else {}
    extra_arrays = data.pop("__arrays__", {})
    extra_info = data.pop("__info__", {})
    atoms.info.update(data)
    atoms.info.update(_decode_arrays(dict(parsed.key_value_pairs)))

    results = {}
    for name in _CALCULATOR_RESULTS:
        value = parsed.get(name)
        if value is not None:
            results[name] = _decode_arrays(value)
            atoms.info[name] = results[name]
    if results:
        atoms.calc = SinglePointCalculator(atoms, **results)

    for name, value in extra_arrays.items():
        atoms.new_array(name, np.asarray(value))
    atoms.info.update(extra_info)

    if atoms.calc is not None:
        for name, (store, key) in _PROMOTED.items():
            if name in atoms.calc.results:
                target = atoms.info if store == "info" else atoms.arrays
                target[key] = atoms.calc.results[name]
    return atoms


class ASEDBBackend:
    """One or more fairchem-style LMDB databases, as one dataset."""

    name = "lmdb"

    def __init__(
        self,
        source: str,
        databases: list[Path],
        ids: list[list[int]],
        key_spec: KeySpecification,
        head: str,
    ) -> None:
        self.source = source
        self.head = head
        self._databases = databases
        self._ids = ids
        self._ends = list(accumulate(len(run) for run in ids))
        self._key_spec = key_spec

    @classmethod
    def sniff(cls, source: str) -> bool:
        for part in source.split(":"):
            path = Path(part)
            if path.is_file():
                if path.suffix.lower() not in SUFFIXES:
                    return False
            elif path.is_dir():
                if not any(
                    child.suffix.lower() in SUFFIXES for child in path.iterdir()
                ):
                    return False
            else:
                return False
        return True

    @classmethod
    def open(
        cls,
        source: str | Path,
        *,
        key_spec: KeySpecification,
        head: str = DEFAULT_HEAD,
        keep_isolated_atoms: bool = False,
    ) -> ASEDBBackend:
        """Open every database and list its row ids.

        The rows are fairchem's training structures and are read as they are,
        so ``keep_isolated_atoms`` changes nothing.
        """
        text = str(source)
        databases = _databases(text)
        ids: list[list[int]] = []
        for path in databases:
            try:
                with _environment(path).begin() as txn:
                    next_id = _stored(txn, "nextid") or 1
                    deleted = set(_stored(txn, "deleted_ids") or [])
            except (lmdb.Error, zlib.error, ValueError) as failure:
                raise DataBackendError(
                    str(path), reason=f"not a readable ASE LMDB database: {failure}"
                ) from failure
            ids.append([row for row in range(1, int(next_id)) if row not in deleted])
        return cls(text, databases, ids, key_spec, head)

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> Configuration:
        if not 0 <= index < len(self):
            raise DataBackendError(
                self.source, index, f"the dataset holds {len(self)} configurations"
            )
        database = bisect.bisect_right(self._ends, index)
        local = index - (self._ends[database - 1] if database else 0)
        row_id = self._ids[database][local]
        path = str(self._databases[database])
        try:
            with _environment(self._databases[database]).begin() as txn:
                row = _stored(txn, str(row_id))
            if row is None:
                raise KeyError(f"row {row_id} is missing")
            atoms = atoms_from_row(row, row_id)
        except Exception as failure:
            raise DataBackendError(
                path, index, f"row {row_id}: {failure!r}"
            ) from failure
        return configuration_from_atoms(atoms, self._key_spec, head_name=self.head)

    def iter_range(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[Configuration]:
        end = len(self) if stop is None else min(stop, len(self))
        for index in range(start, end, step):
            yield self[index]

    def statistics(self) -> DatasetStatistics | None:
        """None. A database carries no precomputed statistics."""
        return None

    def metadata(self) -> DatasetManifest | None:
        """None. A database carries no manifest."""
        return None

    def close(self) -> None:
        """Close this process's environments for these databases.

        Needed only when something else in the process opens the same files,
        which LMDB refuses while they are open here. A later read reopens them.
        """
        for path in self._databases:
            environment = _ENVIRONMENTS.pop((os.getpid(), str(path.resolve())), None)
            if environment is not None:
                environment.close()

    def __getstate__(self) -> dict:
        """Plain data only. Environments are shared per process and reopened
        in the one this lands in."""
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
