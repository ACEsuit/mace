"""Which backend opens a dataset: by name, or by asking every backend.

Backends register through the entry point group ``mace.data_backends``, so a
format can live in its own distribution and is found with no edit here.

**There is no fall-through.** ``format="auto"`` asks every registered backend
whether it recognises the source and needs exactly one yes: none, or more than
one, is an error that names the candidates. A source that nothing claims is
never handed to a backend that might half-read it.

**Discovery records failures; resolution raises.** Listing imports every
registered backend and reports the ones that fail with the reason, so one
broken optional format cannot break the rest. Asking for a backend by name
imports that one only, and raises with the reason when it is unavailable,
since substituting another would hide that the request was not met.

**A name is registered once.** Two entry points registering different objects
under one name raise rather than resolve by install order.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from mace_core.data.backend import DataBackend
from mace_core.data.configuration import DEFAULT_HEAD
from mace_core.data.keys import KeySpecification

__all__ = [
    "DATA_BACKEND_GROUP",
    "DataBackendNotAvailableError",
    "DiscoveredDataBackend",
    "DuplicateDataBackendError",
    "available_data_backends",
    "open_dataset",
]

#: The entry point group dataset backends register in.
DATA_BACKEND_GROUP = "mace.data_backends"

#: The ``format`` that asks every backend to sniff, so no backend can be
#: registered under it.
AUTO = "auto"


class DataBackendNotAvailableError(RuntimeError):
    """No backend could be resolved for a dataset."""


class DuplicateDataBackendError(RuntimeError):
    """Two entry points registered different backends under one name."""


@dataclass(frozen=True)
class DiscoveredDataBackend:
    """One registered backend, and whether it imported.

    Attributes:
        name: The registered name.
        loaded: Whether importing it succeeded here.
        reason: Why it did not, when it did not.
    """

    name: str
    loaded: bool
    reason: str = ""
    backend_class: Any = field(default=None, repr=False, compare=False)


def _describe(entry: EntryPoint) -> str:
    distribution = entry.dist.name if entry.dist is not None else "unknown"
    return f"{entry.value} (from {distribution})"


def _registered() -> dict[str, EntryPoint]:
    """Every entry point in the group by name, read from metadata only."""
    by_name: dict[str, EntryPoint] = {}
    for entry in entry_points(group=DATA_BACKEND_GROUP):
        seen = by_name.setdefault(entry.name, entry)
        if seen.value != entry.value:
            raise DuplicateDataBackendError(
                f"two entry points in {DATA_BACKEND_GROUP!r} register the backend "
                f"name {entry.name!r}: {_describe(seen)} and {_describe(entry)}. "
                f"Neither is "
                f"chosen, since the choice would follow install order. Uninstall "
                f"one of the two distributions."
            )
    return by_name


def _load(name: str, entry: EntryPoint) -> DiscoveredDataBackend:
    if name == AUTO:
        return DiscoveredDataBackend(
            name,
            False,
            f"{AUTO!r} is reserved for sniffing; register under another name",
        )
    try:
        backend_class = entry.load()
    except Exception as failure:
        return DiscoveredDataBackend(name, False, repr(failure))
    missing = [
        m for m in ("sniff", "open") if not callable(getattr(backend_class, m, None))
    ]
    if missing:
        return DiscoveredDataBackend(
            name, False, f"{entry.value} is not a dataset backend: it has no {missing}"
        )
    return DiscoveredDataBackend(name, True, backend_class=backend_class)


def _claims(discovered: DiscoveredDataBackend, source: str) -> bool | str:
    """Whether the backend claims ``source``, or why its ``sniff`` failed."""
    try:
        return bool(discovered.backend_class.sniff(source))
    except Exception as failure:
        return repr(failure)


def available_data_backends() -> list[DiscoveredDataBackend]:
    """Every registered backend, loaded or not, sorted by name.

    Imports every one of them to find the ones that fail.

    Raises:
        DuplicateDataBackendError: if two entry points register different
            backends under one name.
    """
    return [_load(name, entry) for name, entry in sorted(_registered().items())]


def _named(name: str) -> type[DataBackend]:
    registered = _registered()
    if name not in registered:
        raise DataBackendNotAvailableError(
            f"no dataset backend is registered as {name!r}. The registered names "
            f"are {sorted(registered)}, from the entry point group "
            f"{DATA_BACKEND_GROUP!r}."
        )
    discovered = _load(name, registered[name])
    if not discovered.loaded:
        raise DataBackendNotAvailableError(
            f"the dataset backend {name!r} is registered but is not usable: "
            f"{discovered.reason}. No other backend is used in its place."
        )
    return discovered.backend_class


def _sniffed(source: str) -> type[DataBackend]:
    discovered = available_data_backends()
    answers = {d.name: _claims(d, source) for d in discovered if d.loaded}
    claimed = [d for d in discovered if answers.get(d.name) is True]
    if len(claimed) == 1:
        return claimed[0].backend_class
    failed = [f"{d.name} ({d.reason})" for d in discovered if not d.loaded]
    failed += [
        f"{n} (sniff raised {a})" for n, a in answers.items() if isinstance(a, str)
    ]
    unavailable = f" Backends that are not usable: {failed}." if failed else ""
    if claimed:
        raise DataBackendNotAvailableError(
            f"{source!r} is claimed by more than one dataset backend: "
            f"{[d.name for d in claimed]}. Name the one to use with format=."
            + unavailable
        )
    raise DataBackendNotAvailableError(
        f"no dataset backend recognises {source!r}. Check that the path exists, "
        f"or name a backend with format=; the registered ones are "
        f"{[d.name for d in discovered]}." + unavailable
    )


def open_dataset(
    source: str | os.PathLike[str],
    *,
    key_spec: KeySpecification,
    format: str = "auto",
    head: str = DEFAULT_HEAD,
    config_type_weights: Mapping[str, float] | None = None,
    keep_isolated_atoms: bool = False,
    require_labels: bool = True,
) -> DataBackend:
    """Open ``source`` with the backend ``format`` names.

    Args:
        source: The dataset to open.
        key_spec: Which file key each convention name is read from.
        format: A registered backend name, which imports that backend only,
            or ``"auto"``, which asks every registered backend's ``sniff`` and
            requires exactly one to claim the source.
        head, config_type_weights, keep_isolated_atoms, require_labels: Passed
            to the backend's ``open``; see
            :meth:`~mace_core.data.backend.DataBackend.open`.

    Raises:
        DataBackendNotAvailableError: if ``format`` names no registered
            backend or one that is not usable, if ``"auto"`` finds no backend
            or more than one claiming the source, or if what the backend
            opened is not a dataset backend.
        DuplicateDataBackendError: if two entry points register different
            backends under one name.
        DataBackendError: if the backend refuses to open the source.
    """
    path = os.fspath(source)
    backend_class = _sniffed(path) if format == AUTO else _named(format)
    backend = backend_class.open(
        path,
        key_spec=key_spec,
        head=head,
        config_type_weights=config_type_weights,
        keep_isolated_atoms=keep_isolated_atoms,
        require_labels=require_labels,
    )
    if not isinstance(backend, DataBackend):
        raise DataBackendNotAvailableError(
            f"{backend_class.__qualname__}.open returned {type(backend).__name__}, "
            f"which is not a dataset backend."
        )
    return backend
