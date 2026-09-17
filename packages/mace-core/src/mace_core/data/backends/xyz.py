"""The XYZ backend: anything ase can read, held in memory.

Parsing is not reimplemented here. It delegates to
:func:`mace_core.data.xyz.read_configurations`, so the property-key variants,
the weights and the isolated-atom detection are inherited rather than written
twice and then drifting apart.

In memory because that is what the format is: an extended-XYZ file has no
index, so random access means having parsed it. A format that can be read
lazily gets its own backend rather than this one growing a mode.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from mace_core.data.backend import DataBackendError, DatasetManifest, DatasetStatistics
from mace_core.data.configuration import Configuration
from mace_core.data.keys import KeySpecification
from mace_core.data.xyz import read_configurations

__all__ = ["XYZBackend"]

#: What ase reads and this backend therefore claims. Checked as a suffix and
#: nothing more: sniffing must not parse the file.
SUFFIXES = (".xyz", ".extxyz")


class XYZBackend:
    """An ase-readable file of structures."""

    name = "xyz"

    def __init__(
        self,
        source: str,
        configurations: list[Configuration],
        isolated_atom_energies: dict[int, float],
    ) -> None:
        self.source = source
        self._configurations = configurations
        self.isolated_atom_energies = isolated_atom_energies

    @classmethod
    def sniff(cls, source: str) -> bool:
        return Path(source).suffix.lower() in SUFFIXES

    @classmethod
    def open(
        cls,
        source: str | Path,
        *,
        key_spec: KeySpecification,
        head: str = "Default",
    ) -> XYZBackend:
        text = str(source)
        if not Path(text).is_file():
            raise DataBackendError(text, reason="no such file")
        try:
            parsed = read_configurations(
                text,
                key_spec,
                head_name=head,
                extract_isolated_atom_energies=True,
            )
        except DataBackendError:
            raise
        except Exception as failure:
            raise DataBackendError(text, reason=repr(failure)) from failure
        return cls(text, parsed.configurations, parsed.isolated_atom_energies)

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
        """None. An XYZ file carries no precomputed statistics, so the caller
        computes them."""
        return None

    def metadata(self) -> DatasetManifest | None:
        """None. The manifest belongs to a written shard, which is DATA-2's."""
        return None

    def __getstate__(self) -> dict:
        """Everything here is already plain data, so pickling is the default.

        Stated rather than omitted because the Protocol requires a backend to
        survive being sent to a dataloader worker, and a backend that holds an
        open handle has to drop it here.
        """
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
