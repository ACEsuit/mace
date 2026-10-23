"""Neighbour-list backends: the protocol, the registry and the config section.

A backend finds the edges and nothing else. It receives the
:class:`~mace_core.neighbors.geometry.SearchGeometry` the graph builder already
computed and returns ``(edge_index, unit_shifts)``; the effective cell and the
shifts are filled in afterwards by
:func:`~mace_core.neighbors.neighborhood.get_neighborhood`, the same way
whichever backend ran. Swapping backends is therefore a choice about speed and
robustness and never about the numbers: every registered backend has to
reproduce the brute-force edge set.

Backends register through the entry point group ``mace.neighbor_backends``,
the same way kernel backends do, so one can live in its own distribution. The
providers registered there return numpy arrays and import no framework.

**Discovery records failures; resolution raises.** A backend whose import fails
is listed with the reason. Asking for it by name raises with that reason,
because substituting another one would hide that the request was not met.

**A name is registered once.** Two entry points under one name would resolve by
install order. Two that name the same object are one registration; two that
name different objects raise wherever discovery runs.

**Resolve once.** Discovery reads installed metadata, which costs milliseconds.
A graph builder resolves its backend when it is set up and hands the object to
every :func:`~mace_core.neighbors.neighborhood.get_neighborhood` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import field_validator

from mace_core.config.base import ConfigSection
from mace_core.neighbors.geometry import SearchGeometry

__all__ = [
    "DEFAULT_NEIGHBOR_BACKEND",
    "NEIGHBOR_BACKEND_GROUP",
    "DiscoveredNeighborBackend",
    "DuplicateNeighborBackendError",
    "NeighborBackendNotAvailableError",
    "NeighborListBackend",
    "NeighborListConfig",
    "available_neighbor_backends",
    "get_neighbor_backend",
]

#: The entry point group framework-free neighbour-list providers register in.
NEIGHBOR_BACKEND_GROUP = "mace.neighbor_backends"

#: The backend a graph builder uses when the config names none.
DEFAULT_NEIGHBOR_BACKEND = "matscipy"


@runtime_checkable
class NeighborListBackend(Protocol):
    """Finds every directed pair of atoms closer than the cutoff.

    An entry point names a class; the registry instantiates it with no
    arguments.
    """

    def search(
        self, positions: np.ndarray, cutoff: float, geometry: SearchGeometry
    ) -> tuple[np.ndarray, np.ndarray]:
        """The edge set of one structure.

        Args:
            positions: ``[n_atoms, 3]`` float64 Cartesian positions, Angstrom.
                Read-only to the backend.
            cutoff: Strict bound: a pair is an edge when its distance is
                ``< cutoff``.
            geometry: The periodicity and the search box to bin into. A
                backend that does not bin may ignore ``search_cell``, but
                images exist only along axes where ``geometry.pbc`` is true.

        Returns:
            ``edge_index``, ``[2, n_edges]`` integers, row 0 the sender ``i``
            and row 1 the receiver ``j``; and ``unit_shifts``, ``[n_edges, 3]``
            integers ``n`` such that ``positions[j] - positions[i] +
            n @ cell`` is the edge vector. Every pair appears once per
            direction. The zero-shift self pair may be included or not;
            :func:`~mace_core.neighbors.neighborhood.get_neighborhood` drops
            it either way. Edge order is unspecified.
        """
        ...


class NeighborListConfig(ConfigSection):
    """Which neighbour-list backend builds the graphs.

    Resolved when the graph builder is set up, never inside a model forward.

    Attributes:
        kind: A name registered in the ``mace.neighbor_backends`` entry point
            group. ``matscipy`` and ``brute_force`` ship with ``mace-core``.
    """

    kind: str = DEFAULT_NEIGHBOR_BACKEND

    @field_validator("kind")
    @classmethod
    def _named(cls, kind: str) -> str:
        if not kind.strip():
            raise ValueError(
                "neighbor_list.kind is empty. Name a registered backend, such "
                f"as {DEFAULT_NEIGHBOR_BACKEND!r}."
            )
        return kind


class NeighborBackendNotAvailableError(RuntimeError):
    """A neighbour-list backend was asked for by name and could not be built."""


class DuplicateNeighborBackendError(RuntimeError):
    """Two entry points registered different backends under one name."""


@dataclass(frozen=True)
class DiscoveredNeighborBackend:
    """One entry point, and whether it loaded.

    Attributes:
        name: The registered name.
        loaded: Whether importing it succeeded on this machine.
        reason: Why it did not, when it did not.
    """

    name: str
    loaded: bool
    reason: str = ""
    factory: Any = field(default=None, repr=False, compare=False)


def _describe(entry: Any) -> str:
    distribution = getattr(entry, "dist", None)
    origin = getattr(distribution, "name", None) or "an unknown distribution"
    return f"{entry.name} = {getattr(entry, 'value', '?')} (from {origin})"


def _discover() -> list[DiscoveredNeighborBackend]:
    by_name: dict[str, Any] = {}
    for entry in entry_points(group=NEIGHBOR_BACKEND_GROUP):
        seen = by_name.get(entry.name)
        if seen is None:
            by_name[entry.name] = entry
        elif getattr(seen, "value", None) != getattr(entry, "value", None):
            raise DuplicateNeighborBackendError(
                f"two entry points in {NEIGHBOR_BACKEND_GROUP!r} register the "
                f"backend name {entry.name!r}: {_describe(seen)} and "
                f"{_describe(entry)}. Neither is chosen, because the choice would "
                f"follow install order. Uninstall one of the two distributions, "
                f"or register one under another name."
            )
    found = []
    for name, entry in sorted(by_name.items()):
        try:
            factory = entry.load()
        except Exception as failure:
            found.append(DiscoveredNeighborBackend(name, False, repr(failure)))
        else:
            found.append(DiscoveredNeighborBackend(name, True, "", factory))
    return found


def available_neighbor_backends() -> list[DiscoveredNeighborBackend]:
    """Every registered neighbour-list backend, loaded or not, sorted by name.

    Raises:
        DuplicateNeighborBackendError: if two entry points register different
            objects under one name.
    """
    return _discover()


def get_neighbor_backend(name: str = DEFAULT_NEIGHBOR_BACKEND) -> NeighborListBackend:
    """The backend registered under ``name``, instantiated.

    Raises:
        NeighborBackendNotAvailableError: if no backend of that name is
            registered, if it failed to import, or if what it names does not
            implement :class:`NeighborListBackend`. The message says which.
        DuplicateNeighborBackendError: if two entry points register different
            objects under one name.
    """
    by_name = {backend.name: backend for backend in _discover()}
    if name not in by_name:
        raise NeighborBackendNotAvailableError(
            f"no neighbour-list backend is registered as {name!r}. The "
            f"registered names are {sorted(by_name)}, from the entry point "
            f"group {NEIGHBOR_BACKEND_GROUP!r}."
        )
    discovered = by_name[name]
    if not discovered.loaded:
        raise NeighborBackendNotAvailableError(
            f"the neighbour-list backend {name!r} is registered but did not "
            f"import on this machine: {discovered.reason}. No other backend is "
            f"substituted."
        )
    backend = discovered.factory()
    if not isinstance(backend, NeighborListBackend):
        raise NeighborBackendNotAvailableError(
            f"the entry point {name!r} built a {type(backend).__name__}, which "
            f"has no search(positions, cutoff, geometry) method."
        )
    return backend
