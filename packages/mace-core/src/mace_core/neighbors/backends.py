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

**Resolving a name imports one backend.** Listing imports every registered
backend, because that is how it learns which ones fail; a backend that sets up
a device at import pays for it there. :func:`get_neighbor_backend` reads the
metadata of all of them and imports only the one it names.

**One default per framework; anything else is asked for by name.** A config
section that names no backend gets the default of the framework building the
graphs, which it passes to :func:`resolve_neighbor_backend`; ``brute_force``
when no framework is involved. A named backend is used as named. Either is
built or refused, never replaced by another. Only ``brute_force`` is
guaranteed by ``mace-core``. ``matscipy`` is its extra of the same name;
``vesin`` is registered by mace-torch, because importing vesin also imports
torch whenever torch is installed. Each framework package requires the
library of its default. The choice never changes a number, since every
backend returns the same edge set, so it is not part of a model's state.

**Resolve once.** A graph builder resolves its backend when it is set up and
hands the object to every
:func:`~mace_core.neighbors.neighborhood.get_neighborhood` call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import Field, field_validator, model_validator

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
    "resolve_neighbor_backend",
]

#: The entry point group framework-free neighbour-list providers register in.
NEIGHBOR_BACKEND_GROUP = "mace.neighbor_backends"

#: The backend used when the config names none and no framework passes its
#: own default. It ships with ``mace-core`` and needs nothing beyond numpy.
DEFAULT_NEIGHBOR_BACKEND = "brute_force"


@runtime_checkable
class NeighborListBackend(Protocol):
    """Finds every directed pair of atoms closer than the cutoff.

    An entry point names a class or a factory; the registry calls it with the
    config section's ``settings`` as keyword arguments, and with none when
    there are none.
    """

    def search(
        self, positions: np.ndarray, cutoff: float, geometry: SearchGeometry
    ) -> tuple[np.ndarray, np.ndarray]:
        """The edge set of one structure.

        Args:
            positions: ``[n_atoms, 3]`` float64 Cartesian positions, Angstrom.
                Read-only to the backend.
            cutoff: Strict bound: a pair is an edge when its float64 squared
                length is ``< cutoff * cutoff``. Comparing a square-rooted
                length instead drops a pair whose length rounds onto the
                cutoff, such as the face diagonals of a unit cube against
                ``cutoff = np.sqrt(2)``.
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
            group, or ``None`` for the default of the framework building the
            graphs (see :func:`resolve_neighbor_backend`). ``brute_force`` ships with
            ``mace-core``, ``matscipy`` with its extra of the same name, and
            ``vesin`` with mace-torch, where it takes ``n_threads``.
        settings: Keyword arguments for the backend's factory. Validated by
            the backend when it is built, since a backend in its own
            distribution declares settings this schema cannot know. Only with
            a ``kind``, because settings belong to one backend.
    """

    kind: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _named(cls, kind: str | None) -> str | None:
        if kind is not None and not kind.strip():
            raise ValueError(
                "neighbor_list.kind is empty. Name a registered backend, such "
                f"as {DEFAULT_NEIGHBOR_BACKEND!r}, or leave it out for the "
                f"framework's default."
            )
        return kind

    @model_validator(mode="after")
    def _settings_need_a_kind(self) -> NeighborListConfig:
        if self.settings and self.kind is None:
            raise ValueError(
                f"neighbor_list.settings {sorted(self.settings)} are given "
                f"without neighbor_list.kind. Settings belong to one backend; "
                f"name it."
            )
        return self


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


def _registered() -> dict[str, Any]:
    """Every entry point in the group by name, read from metadata only.

    Raises:
        DuplicateNeighborBackendError: if two entry points register different
            objects under one name.
    """
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
    return by_name


def _load(name: str, entry: Any) -> DiscoveredNeighborBackend:
    try:
        factory = entry.load()
    except Exception as failure:
        return DiscoveredNeighborBackend(name, False, repr(failure))
    return DiscoveredNeighborBackend(name, True, "", factory)


def available_neighbor_backends() -> list[DiscoveredNeighborBackend]:
    """Every registered neighbour-list backend, loaded or not, sorted by name.

    Imports every registered backend to find the ones that fail. Use
    :func:`get_neighbor_backend` to resolve one; it imports only that one.

    Raises:
        DuplicateNeighborBackendError: if two entry points register different
            objects under one name.
    """
    return [_load(name, entry) for name, entry in sorted(_registered().items())]


def get_neighbor_backend(
    name: str, settings: Mapping[str, Any] | None = None
) -> NeighborListBackend:
    """The backend registered under ``name``, built with ``settings``.

    Only that backend is imported.

    Args:
        name: A registered backend name, usually ``NeighborListConfig.kind``.
        settings: Keyword arguments for its factory, usually
            ``NeighborListConfig.settings``. ``None`` passes none.

    Raises:
        NeighborBackendNotAvailableError: if no backend of that name is
            registered, if it failed to import, if its factory refused the
            settings, or if what it built does not implement
            :class:`NeighborListBackend`. The message says which.
        DuplicateNeighborBackendError: if two entry points register different
            objects under one name.
    """
    registered = _registered()
    if name not in registered:
        raise NeighborBackendNotAvailableError(
            f"no neighbour-list backend is registered as {name!r}. The "
            f"registered names are {sorted(registered)}, from the entry point "
            f"group {NEIGHBOR_BACKEND_GROUP!r}."
        )
    discovered = _load(name, registered[name])
    if not discovered.loaded:
        raise NeighborBackendNotAvailableError(
            f"the neighbour-list backend {name!r} is registered but did not "
            f"import on this machine: {discovered.reason}. No other backend is "
            f"substituted."
        )
    settings = dict(settings or {})
    try:
        backend = discovered.factory(**settings)
    except (TypeError, ValueError) as failure:
        raise NeighborBackendNotAvailableError(
            f"the neighbour-list backend {name!r} could not be built with the "
            f"settings {settings!r}: {failure}"
        ) from failure
    if not isinstance(backend, NeighborListBackend):
        raise NeighborBackendNotAvailableError(
            f"the entry point {name!r} built a {type(backend).__name__}, which "
            f"has no search(positions, cutoff, geometry) method."
        )
    return backend


def resolve_neighbor_backend(
    config: NeighborListConfig | None = None, default: str = DEFAULT_NEIGHBOR_BACKEND
) -> NeighborListBackend:
    """The backend a graph builder uses: the one the config names, or ``default``.

    Args:
        config: The ``neighbor_list`` section. ``None`` names no backend.
        default: The framework's default, used when the config names none.

    Raises:
        NeighborBackendNotAvailableError: if the backend is not registered,
            did not import or refused its settings, as
            :func:`get_neighbor_backend` does. No other backend is tried.
        DuplicateNeighborBackendError: if two entry points register different
            objects under one name.
    """
    config = config or NeighborListConfig()
    if config.kind is None:
        return get_neighbor_backend(default)
    return get_neighbor_backend(config.kind, config.settings)
