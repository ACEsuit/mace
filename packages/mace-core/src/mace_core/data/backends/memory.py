"""A dataset read once and kept in memory.

A run may hold a dataset in memory instead of reading its backend on every
access. :meth:`InMemoryBackend.load` reads every structure once and returns a
backend that stands in for the one it read: same length, same order, same
isolated-atom energies, so whatever indexes the dataset does not tell the two
apart.

It is not a format. It has no entry point, ``format="auto"`` never matches it,
and :meth:`InMemoryBackend.open` refuses: a file is opened by its own backend,
and a second way to read one is what the registry exists to prevent.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from mace_core.data.backend import DataBackend, DataBackendError
from mace_core.data.configuration import DEFAULT_HEAD, Configuration
from mace_core.data.keys import KeySpecification

__all__ = ["InMemoryBackend"]


class InMemoryBackend:
    """Configurations already parsed, behind the backend contract.

    Args:
        configurations: The structures, in order. The list is copied; the
            configurations are not.
        source: What errors name.
        isolated_atom_energies: ``{atomic number: eV}`` to report, as the
            backend these configurations came from reported them, stored as
            ``int`` and ``float``.
    """

    def __init__(
        self,
        configurations: Sequence[Configuration],
        *,
        source: str = "<memory>",
        isolated_atom_energies: Mapping[int, float] | None = None,
    ) -> None:
        self.source = source
        self._configurations = list(configurations)
        self._isolated_atom_energies = {
            int(z): float(energy)
            for z, energy in (isolated_atom_energies or {}).items()
        }

    @classmethod
    def load(cls, backend: DataBackend) -> InMemoryBackend:
        """Every structure of ``backend``, read once through ``iter_range``."""
        return cls(
            list(backend.iter_range()),
            source=backend.source,
            isolated_atom_energies=backend.isolated_atom_energies(),
        )

    @classmethod
    def sniff(cls, source: str) -> bool:
        """Never: there is no file this reads."""
        return False

    @classmethod
    def open(
        cls,
        source: str,
        *,
        key_spec: KeySpecification,
        head: str = DEFAULT_HEAD,
        config_type_weights: Mapping[str, float] | None = None,
        keep_isolated_atoms: bool = False,
        require_labels: bool = True,
    ) -> InMemoryBackend:
        """Refused. Open the file with its own backend and pass it to :meth:`load`."""
        raise DataBackendError(
            str(source),
            reason=(
                "the in-memory backend reads no files. Open the source with its "
                "own backend and pass that backend to InMemoryBackend.load."
            ),
        )

    def __len__(self) -> int:
        return len(self._configurations)

    def __getitem__(self, index: int) -> Configuration:
        if not 0 <= index < len(self._configurations):
            raise DataBackendError(
                self.source,
                index,
                f"out of range; the dataset holds {len(self._configurations)} "
                f"structures",
            )
        return self._configurations[index]

    def iter_range(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[Configuration]:
        if step < 1:
            raise DataBackendError(
                self.source, reason=f"iter_range takes a positive step, not {step}"
            )
        stop = len(self._configurations) if stop is None else stop
        for index in range(start, stop, step):
            yield self[index]

    def isolated_atom_energies(self) -> dict[int, float]:
        return dict(self._isolated_atom_energies)
