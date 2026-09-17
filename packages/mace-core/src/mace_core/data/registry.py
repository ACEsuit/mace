"""Finding a data backend, with no fall-through anywhere.

The frozen tree selects a format by sniffing the path name through four
successive guesses and then a final "attempting to load as LMDB"
(``mace/tools/run_train_utils.py:32-162``). A typo in a path is therefore not
an error: it is an attempt to read a file that does not exist as a database,
and the failure arrives far from the cause.

Here resolution is explicit or it fails. ``format="auto"`` asks every
registered backend whether it claims the source, in a declared priority order,
and **zero matches or more than one match are both hard errors naming the
candidates**. There is no last resort.

Discovery and resolution differ in the same way as for kernel backends: a
backend whose import fails is recorded so that listing works on a machine
missing an optional dependency, and asking for it by name raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from mace_core.data.backend import DataBackend, DataBackendError
from mace_core.data.keys import KeySpecification

__all__ = [
    "ENTRY_POINT_GROUP",
    "SNIFF_PRIORITY",
    "AmbiguousFormatError",
    "DiscoveredDataBackend",
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


@dataclass(frozen=True)
class DiscoveredDataBackend:
    name: str
    loaded: bool
    reason: str = ""
    factory: Any = field(default=None, repr=False, compare=False)


def _discover() -> dict[str, DiscoveredDataBackend]:
    found: dict[str, DiscoveredDataBackend] = {}
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        try:
            factory = entry.load()
        except Exception as failure:
            found[entry.name] = DiscoveredDataBackend(entry.name, False, repr(failure))
        else:
            found[entry.name] = DiscoveredDataBackend(entry.name, True, "", factory)
    return found


def available_backends() -> dict[str, DiscoveredDataBackend]:
    """Every registered backend, loaded or not.

    A broken optional backend must not break ``import``, so this never raises
    for one that failed to load. The record says why.
    """
    return _discover()


def _ordered(names: list[str]) -> list[str]:
    ranked = {name: position for position, name in enumerate(SNIFF_PRIORITY)}
    return sorted(names, key=lambda name: (ranked.get(name, len(ranked)), name))


def open_dataset(
    source: str | Path,
    *,
    format: str = "auto",
    key_spec: KeySpecification,
    head: str = "Default",
) -> DataBackend:
    """Open ``source`` with a named backend, or by asking which one claims it.

    Args:
        source: The path.
        format: A registered backend name, or ``"auto"``.
        key_spec: Which keys carry which property.
        head: Which head the configurations belong to.

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
    discovered = _discover()
    text = str(source)

    if format != "auto":
        if format not in discovered:
            raise UnknownBackendError(
                text,
                reason=(
                    f"no data backend is registered as {format!r}. The "
                    f"registered names are {sorted(discovered)}, from the "
                    f"entry point group {ENTRY_POINT_GROUP!r}."
                ),
            )
        record = discovered[format]
        if not record.loaded:
            raise UnknownBackendError(
                text,
                reason=(
                    f"the data backend {format!r} is registered but did not "
                    f"import: {record.reason}"
                ),
            )
        return record.factory.open(source, key_spec=key_spec, head=head)

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
    return discovered[claimed[0]].factory.open(source, key_spec=key_spec, head=head)
