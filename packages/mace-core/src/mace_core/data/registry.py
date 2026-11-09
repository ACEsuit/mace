"""Discover and resolve data backends through entry points.

``format="auto"`` asks in ``SNIFF_PRIORITY`` order and requires exactly one
backend to claim the source. Listings retain import failures; named resolution
raises for unavailable providers. Conflicting registrations raise
``DuplicateDataBackendError``; identical registrations coalesce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any

from mace_core._entry_points import load_entry_point, unique_entries
from mace_core.data.backend import DataBackend, DataBackendError
from mace_core.data.keys import KeySpecification

__all__ = [
    "ENTRY_POINT_GROUP",
    "SNIFF_PRIORITY",
    "AmbiguousFormatError",
    "DiscoveredDataBackend",
    "DuplicateDataBackendError",
    "UnknownBackendError",
    "available_backends",
    "open_dataset",
]

#: Third-party wheels declare into exactly this group. One backend is one
#: module plus one line of metadata, with no edit anywhere in this repository.
ENTRY_POINT_GROUP = "mace.data_backends"

#: The order ``format="auto"`` asks in. Declared rather than discovered,
#: because a resolution order that depended on installation order would make
#: the same path open as different formats on two machines. A name missing
#: from here is asked last, in alphabetical order.
SNIFF_PRIORITY: tuple[str, ...] = ("xyz", "hdf5", "hdf5-legacy", "lmdb")


class UnknownBackendError(DataBackendError):
    """A format was named and no backend is registered under it."""


class AmbiguousFormatError(DataBackendError):
    """``format="auto"`` matched zero backends, or more than one."""


class DuplicateDataBackendError(RuntimeError):
    """Different data backends registered under one name."""


@dataclass(frozen=True)
class DiscoveredDataBackend:
    name: str
    loaded: bool
    reason: str = ""
    factory: Any = field(default=None, repr=False, compare=False)


def _entries() -> dict[str, EntryPoint]:
    return unique_entries(
        entry_points(group=ENTRY_POINT_GROUP),
        ENTRY_POINT_GROUP,
        DuplicateDataBackendError,
    )


def _discover(entries: dict[str, EntryPoint]) -> dict[str, DiscoveredDataBackend]:
    found = {}
    for name, entry in entries.items():
        factory, reason = load_entry_point(entry)
        found[name] = DiscoveredDataBackend(name, not reason, reason, factory)
    return found


def available_backends() -> dict[str, DiscoveredDataBackend]:
    """Every registered backend, loaded or not.

    A broken optional backend must not break ``import``, so this never raises
    for one that failed to load. The record says why.
    """
    return _discover(_entries())


def _ordered(names: list[str]) -> list[str]:
    ranked = {name: position for position, name in enumerate(SNIFF_PRIORITY)}
    return sorted(names, key=lambda name: (ranked.get(name, len(ranked)), name))


def open_dataset(
    source: str | Path,
    *,
    format: str = "auto",
    key_spec: KeySpecification,
    head: str = "Default",
    keep_isolated_atoms: bool = False,
) -> DataBackend:
    """Open ``source`` with a named backend, or by asking which one claims it.

    Args:
        source: The path.
        format: A registered backend name, or ``"auto"``.
        key_spec: Which keys carry which property.
        head: Which head the configurations belong to.
        keep_isolated_atoms: Leave the single-atom reference structures in.

    Raises:
        UnknownBackendError: A named format nobody registered. Lists what there
            is.
        AmbiguousFormatError: ``"auto"`` matched none, or several. Names the
            candidates either way. There is deliberately no fall-through: a
            path that matches nothing is a mistake, and guessing at it is how
            the frozen tree turns a typo into an attempt to open a database.
        DataBackendError: The backend was found and the source could not be
            read.
    """
    entries = _entries()
    text = str(source)

    if format != "auto":
        if format not in entries:
            raise UnknownBackendError(
                text,
                reason=(
                    f"no data backend is registered as {format!r}. The "
                    f"registered names are {sorted(entries)}, from the "
                    f"entry point group {ENTRY_POINT_GROUP!r}."
                ),
            )
        factory, reason = load_entry_point(entries[format])
        if reason:
            raise UnknownBackendError(
                text,
                reason=(
                    f"the data backend {format!r} is registered but did not "
                    f"import: {reason}"
                ),
            )
        return factory.open(
            source,
            key_spec=key_spec,
            head=head,
            keep_isolated_atoms=keep_isolated_atoms,
        )

    discovered = _discover(entries)
    claimed = [
        name
        for name in _ordered(list(discovered))
        if discovered[name].loaded and discovered[name].factory.sniff(text)
    ]
    if not claimed:
        raise AmbiguousFormatError(
            text,
            reason=(
                f"no registered data backend claims this source. The "
                f"registered names are {sorted(discovered)}. Name the format "
                f"explicitly if it is one of them; there is no last resort to "
                f"fall through to, on purpose."
            ),
        )
    if len(claimed) > 1:
        raise AmbiguousFormatError(
            text,
            reason=(
                f"{claimed} all claim this source. Name the one you mean. Two "
                f"backends claiming the same path is a bug in one of their "
                f"sniff methods, and picking the first would hide it."
            ),
        )
    return discovered[claimed[0]].factory.open(
        source, key_spec=key_spec, head=head, keep_isolated_atoms=keep_isolated_atoms
    )
