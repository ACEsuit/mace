"""Structures already in hand, presented as a backend.

The statistics are taken over the *training split*, after the isolated-atom
structures have been read and possibly removed, after the validation set has
been taken out and after several heads' files have been joined. That set exists
only in memory and corresponds to no file, so without this the one statistics
implementation could not be pointed at the thing it is supposed to measure.

It is deliberately not a format. It claims no source and refuses to be opened
from one: a list of configurations is what another backend produced, and a
second way to read a file is exactly what the registry exists to prevent.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from mace_core.data.backend import DataBackendError, DatasetManifest, DatasetStatistics
from mace_core.data.configuration import Configuration
from mace_core.data.keys import KeySpecification

__all__ = ["InMemoryBackend"]


class InMemoryBackend:
    """A sequence of configurations, behind the backend Protocol."""

    name = "memory"

    def __init__(
        self, configurations: Sequence[Configuration], source: str = "<memory>"
    ) -> None:
        self.source = source
        self._configurations = list(configurations)

    @classmethod
    def sniff(cls, source: str) -> bool:
        """Never. There is no file this reads."""
        return False

    @classmethod
    def open(
        cls,
        source: str | Path,
        *,
        key_spec: KeySpecification,
        head: str = "Default",
        keep_isolated_atoms: bool = False,
    ) -> InMemoryBackend:
        raise DataBackendError(
            str(source),
            reason=(
                "the in-memory backend holds configurations another backend "
                "already parsed; it does not read files. Open the file with "
                "its own backend and pass the configurations here."
            ),
        )

    def __len__(self) -> int:
        return len(self._configurations)

    def __getitem__(self, index: int) -> Configuration:
        try:
            return self._configurations[index]
        except IndexError as failure:
            raise DataBackendError(
                self.source,
                index,
                f"the dataset holds {len(self._configurations)} configurations",
            ) from failure

    def iter_range(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[Configuration]:
        end = len(self._configurations) if stop is None else stop
        for index in range(start, end, step):
            yield self[index]

    def statistics(self) -> DatasetStatistics | None:
        """None. Nothing precomputed them, which is why they are being asked
        for."""
        return None

    def metadata(self) -> DatasetManifest | None:
        return None

    def __getstate__(self) -> dict:
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
