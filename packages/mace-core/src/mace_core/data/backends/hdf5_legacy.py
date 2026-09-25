"""Reading HDF5 files the frozen tree prepared, until they are re-prepared.

Read-only, and it warns on every open. A dataset in this layout is not
converted: unlike a checkpoint it can simply be prepared again, into the v2
shards of :mod:`mace_core.data_spec.shard_format`. This backend goes when the
legacy data layer does.

The layout is two levels, ``config_batch_{i}/config_{j}``, one file or a
directory of files (``mace/data/utils.py:594``). Two things are read
differently from the frozen reader, on purpose.

**The length is counted.** The frozen reader multiplies the number of groups
by the size of the first group (``mace/data/hdf5_dataset.py:19-21``), so a file
whose last group is short reports structures that do not exist, and reading one
fails or returns another structure. Here every group is counted.

**A directory's files are read in sorted order.** The frozen reader
concatenates them in whatever order ``glob`` returns
(``mace/data/hdf5_dataset.py:84-92``), which is the directory's order and
differs between filesystems.

The directory form is also what the frozen tree's ``--multi_processed_test``
flag declares by hand: a source is opened as one file or as a directory of
them according to what it is, so there is nothing to declare.

The frozen tree's ``statistics.json`` holds its element table and energies as
stringified Python (``mace/cli/preprocess_data.py:250-260``), read back there
with ``ast.literal_eval``. :func:`read_legacy_statistics` parses exactly that
grammar and nothing more.
"""

from __future__ import annotations

import bisect
import json
import re
import warnings
from collections.abc import Iterator
from itertools import accumulate
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from mace_core.data.backend import DataBackendError, DatasetManifest, DatasetStatistics
from mace_core.data.configuration import DEFAULT_HEAD, Configuration
from mace_core.data.keys import KeySpecification
from mace_core.data_spec.shard_format import MANIFEST_NAME

__all__ = ["LegacyHDF5Backend", "read_legacy_statistics"]

#: The suffixes the frozen tree writes and this backend claims.
SUFFIXES = (".h5", ".hdf5")

_GROUP = re.compile(r"config_batch_(\d+)")
_ENTRY = re.compile(r"config_(\d+)")


def _sorted_numbered(names: Any, pattern: re.Pattern[str]) -> list[str]:
    found = []
    for name in names:
        match = pattern.fullmatch(name)
        if match:
            found.append((int(match.group(1)), name))
    return [name for _, name in sorted(found)]


def _unpack(value: Any) -> Any:
    """What the frozen reader's ``unpack_value`` gives back, without its text
    ``"None"`` sentinel being mistaken for anything but a missing value."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str) and value == "None":
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _flags(values: Any) -> tuple[bool, bool, bool]:
    x, y, z = (bool(flag) for flag in values)
    return x, y, z


def _files(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in SUFFIXES
        )
    return [source]


class LegacyHDF5Backend:
    """The frozen tree's HDF5 layout, one file or a directory of them."""

    name = "hdf5-legacy"

    def __init__(
        self,
        source: str,
        files: list[Path],
        index: list[list[tuple[str, str]]],
        head: str,
    ) -> None:
        self.source = source
        self.head = head
        self._files = files
        self._index = index
        self._ends = list(accumulate(len(entries) for entries in index))
        self._handles: dict[int, h5py.File] = {}

    @classmethod
    def sniff(cls, source: str) -> bool:
        """A ``.h5`` file outside a v2 dataset, or a directory of them with no
        manifest."""
        path = Path(source)
        if path.is_file():
            return (
                path.suffix.lower() in SUFFIXES
                and not (path.parent / MANIFEST_NAME).is_file()
            )
        if path.is_dir() and not (path / MANIFEST_NAME).is_file():
            return any(child.suffix.lower() in SUFFIXES for child in path.iterdir())
        return False

    @classmethod
    def open(
        cls,
        source: str | Path,
        *,
        key_spec: KeySpecification,
        head: str = DEFAULT_HEAD,
        keep_isolated_atoms: bool = False,
    ) -> LegacyHDF5Backend:
        """Open the file or directory, counting every structure it holds.

        The frozen tree's shards hold convention names already and carry no
        isolated atoms, so neither ``key_spec`` nor ``keep_isolated_atoms``
        changes what is read.
        """
        warnings.warn(
            f"{source} is in the legacy HDF5 layout, which is read-only and "
            f"will stop being read when the legacy data layer is removed. "
            f"Prepare the dataset again to get v2 shards.",
            FutureWarning,
            stacklevel=2,
        )
        text = str(source)
        path = Path(text)
        if not path.exists():
            raise DataBackendError(text, reason="no such file or directory")
        files = _files(path)
        if not files:
            raise DataBackendError(text, reason="the directory holds no .h5 files")
        index: list[list[tuple[str, str]]] = []
        for file in files:
            try:
                with h5py.File(file, "r") as handle:
                    groups = _sorted_numbered(handle.keys(), _GROUP)
                    if not groups:
                        raise DataBackendError(
                            str(file),
                            reason=(
                                "no config_batch_ groups; this is not the legacy layout"
                            ),
                        )
                    index.append(
                        [
                            (group, entry)
                            for group in groups
                            for entry in _sorted_numbered(handle[group].keys(), _ENTRY)
                        ]
                    )
            except OSError as failure:
                raise DataBackendError(
                    str(file), reason=f"not a readable HDF5 file: {failure}"
                ) from failure
        return cls(text, files, index, head)

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def _handle(self, file: int) -> h5py.File:
        handle = self._handles.get(file)
        if handle is None:
            handle = h5py.File(self._files[file], "r")
            self._handles[file] = handle
        return handle

    def __getitem__(self, index: int) -> Configuration:
        if not 0 <= index < len(self):
            raise DataBackendError(
                self.source, index, f"the dataset holds {len(self)} configurations"
            )
        file = bisect.bisect_right(self._ends, index)
        local = index - (self._ends[file - 1] if file else 0)
        group, entry = self._index[file][local]
        try:
            return self._read(self._handle(file)[group][entry])
        except (KeyError, OSError, ValueError, TypeError) as failure:
            raise DataBackendError(
                str(self._files[file]), index, f"{group}/{entry}: {failure!r}"
            ) from failure

    def _read(self, group: h5py.Group) -> Configuration:
        pbc = _unpack(group["pbc"][()])
        return Configuration(
            atomic_numbers=group["atomic_numbers"][()],
            positions=group["positions"][()],
            properties={
                name: _unpack(value[()]) for name, value in group["properties"].items()
            },
            property_weights={
                name: float(_unpack(value[()]))
                for name, value in group["property_weights"].items()
            },
            cell=_unpack(group["cell"][()]),
            pbc=None if pbc is None else _flags(pbc),
            weight=float(_unpack(group["weight"][()])),
            config_type=_unpack(group["config_type"][()]),
            head=self.head,
        )

    def iter_range(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[Configuration]:
        end = len(self) if stop is None else min(stop, len(self))
        for index in range(start, end, step):
            yield self[index]

    def statistics(self) -> DatasetStatistics | None:
        """The ``statistics.json`` the frozen tree wrote beside these shards,
        if there is one where it writes it.

        That is the directory itself, or its parent under the preparation
        prefix: ``<prefix>train/`` sits beside ``<prefix>statistics.json``.
        Anywhere else, name the file to :func:`read_legacy_statistics`.
        """
        path = Path(self.source)
        directory = path if path.is_dir() else path.parent
        candidates = [directory / "statistics.json"]
        for split in ("train", "val", "test"):
            if directory.name.endswith(split):
                prefix = directory.name[: -len(split)]
                candidates.append(directory.parent / f"{prefix}statistics.json")
        for candidate in candidates:
            if candidate.is_file():
                return read_legacy_statistics(candidate)
        return None

    def metadata(self) -> DatasetManifest | None:
        """None. The legacy layout carries no manifest."""
        return None

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


# The frozen tree writes ``str(dict)`` and ``str(list)``. The numbers in them
# are Python floats and ints, or numpy scalars, which numpy 2 prints as
# ``np.float64(-13.6)``.
_NUMBER = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|inf|nan)"
_SCALAR = rf"(?:np\.\w+\(\s*{_NUMBER}\s*\)|{_NUMBER})"
_PAIR = re.compile(rf"\s*({_SCALAR})\s*:\s*({_SCALAR})\s*")
_ITEM = re.compile(rf"\s*({_SCALAR})\s*")


def _scalar(token: str) -> float:
    inner = re.fullmatch(rf"np\.\w+\(\s*({_NUMBER})\s*\)", token)
    return float(inner.group(1) if inner else token)


def _split(text: str, opening: str, closing: str, what: str) -> list[str]:
    stripped = text.strip()
    if not (stripped.startswith(opening) and stripped.endswith(closing)):
        raise ValueError(f"{what} is not a {opening}...{closing} literal: {text!r}")
    body = stripped[1:-1].strip()
    if not body:
        return []
    return body.rstrip(",").split(",")


def _energies(text: str) -> dict[int, float]:
    energies: dict[int, float] = {}
    for part in _split(text, "{", "}", "atomic_energies"):
        match = _PAIR.fullmatch(part)
        if match is None:
            raise ValueError(f"atomic_energies entry {part!r} is not Z: energy")
        energies[int(_scalar(match.group(1)))] = _scalar(match.group(2))
    return energies


def _numbers(text: str) -> list[int]:
    numbers = []
    for part in _split(text, "[", "]", "atomic_numbers"):
        match = _ITEM.fullmatch(part)
        if match is None:
            raise ValueError(f"atomic_numbers entry {part!r} is not a number")
        numbers.append(int(_scalar(match.group(1))))
    return numbers


def read_legacy_statistics(path: str | Path) -> DatasetStatistics:
    """Read a ``statistics.json`` in the frozen tree's format.

    ``atomic_energies`` and ``atomic_numbers`` are accepted either as the
    stringified literals the frozen tree writes or as real JSON. The
    ``atomic_energies`` value may also name a further JSON file holding the
    same literal, which the frozen training script follows
    (``mace/cli/run_train.py:319-334``).

    Raises:
        DataBackendError: The file cannot be read, or a field does not parse.
            Names the file and the field.
    """
    source = str(path)
    try:
        payload = json.loads(Path(path).read_text())
        energies = payload["atomic_energies"]
        if isinstance(energies, str) and energies.endswith(".json"):
            energies = json.loads(Path(energies).read_text())
        if isinstance(energies, str):
            energies = _energies(energies)
        numbers = payload["atomic_numbers"]
        if isinstance(numbers, str):
            numbers = _numbers(numbers)
        return DatasetStatistics(
            atomic_energies={int(z): float(e) for z, e in dict(energies).items()},
            avg_num_neighbors=float(payload["avg_num_neighbors"]),
            mean=float(payload["mean"]),
            std=float(payload["std"]),
            atomic_numbers=sorted(int(z) for z in numbers),
            r_max=float(payload["r_max"]),
        )
    except (OSError, KeyError, TypeError, ValueError) as failure:
        raise DataBackendError(
            source, reason=f"not a legacy statistics file: {failure!r}"
        ) from failure
