"""The one interface the data pipeline reads datasets through.

A dataset backend turns a source (a file, a directory of shards, a database)
into a sized, indexable sequence of
:class:`~mace_core.data.configuration.Configuration`. It never builds graphs:
neighbour search, collation and padding happen once, above every backend, so
the same backend feeds every framework.

**Random access and an exact length are the whole contract.** They are what a
map-style loader, a seeded train/validation split and a distributed sampler
need. Which rank reads which index is the sampler's business; a backend never
knows about ranks.

**A read that fails raises.** A backend never hands back ``None`` or a
placeholder for a structure it could not read: :class:`DataBackendError`
names the source and, when the failure is one structure's, its index.

**A capability some formats have and others lack is a protocol of its own**,
checked with ``isinstance``, so adding one never changes :class:`DataBackend`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Protocol, runtime_checkable

from mace_core.data.configuration import DEFAULT_HEAD, Configuration
from mace_core.data.keys import KeySpecification

__all__ = [
    "DataBackend",
    "DataBackendError",
]


class DataBackendError(ValueError):
    """A dataset could not be opened, or one of its structures read.

    Args:
        source: The source as the backend was given it.
        index: The structure that failed, or ``None`` when the source as a
            whole did.
        reason: What went wrong and, where there is one, the fix.
    """

    def __init__(self, source: str, index: int | None = None, reason: str = "") -> None:
        self.source = source
        self.index = index
        where = f"{source!r}" if index is None else f"structure {index} of {source!r}"
        super().__init__(f"{where}: {reason}" if reason else where)


@runtime_checkable
class DataBackend(Protocol):
    """A dataset as a sized, indexable sequence of configurations.

    A backend is picklable, so a loader worker or another rank can receive
    it: a backend holding an open file handle drops it in ``__getstate__`` and
    reopens it on the next read.

    The head is not read from the source. Every structure a backend yields
    carries the ``head`` its :meth:`open` was given, so one opened backend
    serves one head.
    """

    #: The source the backend was opened on, as given. Errors name it.
    source: str

    @classmethod
    def sniff(cls, source: str) -> bool:
        """Whether ``source`` is in this backend's format.

        Cheap: a suffix or a few magic bytes, never a parse. ``format="auto"``
        calls the ``sniff`` of every registered backend and requires exactly
        one to answer ``True``.
        """
        ...

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
    ) -> DataBackend:
        """Open ``source``, resolving file keys to convention names.

        Args:
            source: Whatever the format reads from: a file, a directory.
            key_spec: Which file key each convention name is read from.
            head: The head every structure of this dataset trains.
            config_type_weights: Per-config-type multiplier on the structure
                weight. It acts where structures are parsed from the source. A
                format that stores configurations already parsed applied it
                when it was written, and refuses any value here other than
                ``None``, since applying it again would square it.
            keep_isolated_atoms: Keep the single-atom structures labelled
                ``IsolatedAtom`` among the configurations as well as in
                :meth:`isolated_atom_energies`. Refused like
                ``config_type_weights`` by a format that stores parsed
                configurations.
            require_labels: Refuse a dataset in which no structure it yields
                carries an energy, forces or a dipole; isolated-atom
                references taken out of it do not count. ``False`` reads an
                unlabelled dataset, such as structures to evaluate.

        Raises:
            DataBackendError: if the source cannot be read, yields no
                labelled structure while ``require_labels`` is set, or refuses
                an option.
        """
        ...

    def __len__(self) -> int:
        """The exact number of structures, known without reading them."""
        ...

    def __getitem__(self, index: int) -> Configuration:
        """The structure at ``index``, ``0 <= index < len(self)``.

        Raises:
            DataBackendError: with the index, for an index out of range or a
                structure that cannot be read.
        """
        ...

    def __iter__(self) -> Iterator[Configuration]:
        """Every structure, in order, as ``iter_range()`` yields them.

        Declared rather than left to Python's fallback through
        :meth:`__getitem__`, which stops only on an ``IndexError`` and would
        read past the end into a :class:`DataBackendError`.
        """
        ...

    def iter_range(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[Configuration]:
        """The structures at ``range(start, stop, step)``, in that order.

        ``stop=None`` is the length and ``step`` is positive. Every index is
        in range, as for :meth:`__getitem__`. A format stored in shards reads them in
        sequence here instead of seeking per structure.
        """
        ...

    def isolated_atom_energies(self) -> dict[int, float]:
        """``{atomic number: energy in eV}`` from the isolated-atom references.

        Empty when the source has none. A conflict between two references of
        one element was already refused when the source was opened.
        """
        ...
