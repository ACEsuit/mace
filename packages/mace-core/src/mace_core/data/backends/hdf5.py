"""The HDF5 backend: a prepared dataset of v2 shards, read lazily.

The format is :mod:`mace_core.data_spec.shard_format`, and nothing about it is
decided here. What this adds is access: an exact length from the manifest, a
configuration by index without reading the others, and a sequential run that
reads each shard once.

A source is the dataset directory, or the path of its ``dataset.json``.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator
from itertools import accumulate
from pathlib import Path

import h5py

from mace_core.data.backend import DataBackendError, DatasetManifest, DatasetStatistics
from mace_core.data.configuration import DEFAULT_HEAD, Configuration
from mace_core.data.keys import KeySpecification
from mace_core.data_spec.shard_format import (
    MANIFEST_NAME,
    read_configuration,
    read_manifest,
)

__all__ = ["HDF5Backend"]


def _dataset_directory(source: str | Path) -> Path:
    path = Path(source)
    return path.parent if path.name == MANIFEST_NAME else path


class HDF5Backend:
    """A directory of v2 shards and its manifest.

    Structures are served as they were written. The isolated-atom structures
    are whatever the writer was given, so ``keep_isolated_atoms`` has nothing
    to act on: they are separated from the training structures before a
    dataset is prepared, not after.

    The key specification is recorded in the manifest at write time and the
    shards hold convention names, so the one passed to :meth:`open` renames
    nothing.
    """

    name = "hdf5"

    def __init__(self, directory: Path, manifest: DatasetManifest, head: str) -> None:
        self.source = str(directory)
        self.head = head
        self._directory = directory
        self._manifest = manifest
        self._ends = list(accumulate(manifest.shard_counts))
        self._handles: dict[int, h5py.File] = {}

    @classmethod
    def sniff(cls, source: str) -> bool:
        path = Path(source)
        if path.name == MANIFEST_NAME:
            return path.is_file()
        return (path / MANIFEST_NAME).is_file()

    @classmethod
    def open(
        cls,
        source: str | Path,
        *,
        key_spec: KeySpecification,
        head: str = DEFAULT_HEAD,
        keep_isolated_atoms: bool = False,
    ) -> HDF5Backend:
        directory = _dataset_directory(source)
        manifest = read_manifest(directory)
        for name in manifest.shards:
            if not (directory / name).is_file():
                raise DataBackendError(
                    str(directory / name),
                    reason="the manifest lists this shard and it does not exist",
                )
        return cls(directory, manifest, head)

    def __len__(self) -> int:
        return self._manifest.length

    def _shard(self, shard: int) -> h5py.File:
        handle = self._handles.get(shard)
        if handle is None:
            path = self._directory / self._manifest.shards[shard]
            try:
                handle = h5py.File(path, "r")
            except OSError as failure:
                raise DataBackendError(
                    str(path), reason=f"not a readable HDF5 file: {failure}"
                ) from failure
            self._handles[shard] = handle
        return handle

    def _locate(self, index: int) -> tuple[int, int]:
        if not 0 <= index < len(self):
            raise DataBackendError(
                self.source, index, f"the dataset holds {len(self)} configurations"
            )
        shard = bisect.bisect_right(self._ends, index)
        start = self._ends[shard - 1] if shard else 0
        return shard, index - start

    def _read(
        self, configs: h5py.Group, shard: int, local: int, index: int
    ) -> Configuration:
        try:
            return read_configuration(configs[str(local)], head=self.head)
        except (KeyError, OSError, ValueError, TypeError) as failure:
            raise DataBackendError(
                str(self._directory / self._manifest.shards[shard]),
                index,
                f"configuration {local} of this shard: {failure!r}",
            ) from failure

    def _configs(self, shard: int) -> h5py.Group:
        try:
            return self._shard(shard)["configs"]
        except KeyError as failure:
            raise DataBackendError(
                str(self._directory / self._manifest.shards[shard]),
                reason="no configs group; this is not a v2 shard",
            ) from failure

    def __getitem__(self, index: int) -> Configuration:
        shard, local = self._locate(index)
        return self._read(self._configs(shard), shard, local, index)

    def iter_range(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[Configuration]:
        """A run in index order, looking each shard up once rather than per item."""
        end = len(self) if stop is None else min(stop, len(self))
        current, configs = -1, None
        for index in range(start, end, step):
            shard, local = self._locate(index)
            if shard != current or configs is None:
                current, configs = shard, self._configs(shard)
            yield self._read(configs, shard, local, index)

    def statistics(self) -> DatasetStatistics | None:
        """The statistics embedded at write time. Never recomputed."""
        return self._manifest.statistics

    def metadata(self) -> DatasetManifest:
        """The manifest. A v2 dataset always has one."""
        return self._manifest

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict:
        """Open files cannot be pickled, so they are dropped and reopened."""
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
