"""Neighbour-list providers, config and entry-point resolution.

Host providers receive ``SearchGeometry`` and return numpy ``(edge_index,
unit_shifts)`` arrays matching the all-pairs edge set. Framework groups add
device providers; their framework owns the tensor protocol. The core decides
the effective cell through ``search_geometry`` for both kinds.

Listing imports all providers and records failures; named resolution imports
only the requested provider and raises on failure. Identical registrations
coalesce; conflicting names raise before imports.

Graph builders resolve a provider once and reuse it. Each framework supplies
its default; standalone core defaults to ``all_pairs``. Core optionally provides
``matscipy``; mace-torch provides ``vesin`` because importing it may import torch.
Provider choice changes performance, not the edge set or model state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import Field, ValidationError, field_validator, model_validator

from mace_core._entry_points import load_entry_point, unique_entries
from mace_core.config.base import ConfigSection
from mace_core.edges import NeighborListCapabilities
from mace_core.neighbors.geometry import SearchGeometry

__all__ = [
    "DEFAULT_NEIGHBOR_BACKEND",
    "NEIGHBOR_BACKEND_GROUP",
    "BatchedNeighborListBackend",
    "DiscoveredNeighborBackend",
    "DuplicateNeighborBackendError",
    "NeighborBackendNotAvailableError",
    "NeighborListBackend",
    "NeighborListConfig",
    "available_neighbor_backends",
    "get_neighbor_backend",
    "neighbor_capabilities",
    "resolve_neighbor_backend",
]

#: The entry point group framework-free neighbour-list providers register in.
NEIGHBOR_BACKEND_GROUP = "mace.neighbor_backends"

#: The backend used when the config names none and no framework passes its
#: own default. It ships with ``mace-core`` and needs nothing beyond numpy.
DEFAULT_NEIGHBOR_BACKEND = "all_pairs"


@runtime_checkable
class NeighborListBackend(Protocol):
    """Finds every directed pair of atoms closer than the cutoff.

    An entry point names a class or a factory. One that takes settings
    declares them as ``settings_section``, a
    :class:`~mace_core.config.ConfigSection` subclass: the registry validates
    the config section's ``settings`` against it and calls the factory with
    the validated section. A factory without one is called with no arguments
    and accepts no settings.
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


@runtime_checkable
class BatchedNeighborListBackend(Protocol):
    """Optional, stateless search returning each structure's ordinary search order."""

    def search_batch(
        self,
        positions: Sequence[np.ndarray],
        cutoff: float,
        geometries: Sequence[SearchGeometry],
    ) -> Sequence[tuple[np.ndarray, np.ndarray]]: ...


def neighbor_capabilities(backend: Any, name: str) -> NeighborListCapabilities:
    """Read a provider's declaration; old host providers guarantee no ordering."""
    method = getattr(backend, "capabilities", None)
    capabilities = method() if method is not None else NeighborListCapabilities(name)
    if not isinstance(capabilities, NeighborListCapabilities):
        raise NeighborBackendNotAvailableError(
            f"neighbour provider {name!r} must declare NeighborListCapabilities."
        )
    if capabilities.stateful and callable(getattr(backend, "search_batch", None)):
        raise NeighborBackendNotAvailableError(
            f"neighbour provider {name!r} is stateful and cannot offer "
            "search_batch; keep one instance per system."
        )
    return capabilities


class NeighborListConfig(ConfigSection):
    """Which neighbour-list backend builds the graphs.

    Resolved when the graph builder is set up, never inside a model forward.

    Attributes:
        kind: A name registered in ``mace.neighbor_backends`` or its framework
            group, or ``None`` for the default of the framework building the
            graphs (see :func:`resolve_neighbor_backend`). ``all_pairs``
            ships with ``mace-core``, ``matscipy`` with its extra of the same
            name, and ``vesin`` with mace-torch, where it takes ``n_threads``.
        settings: The backend's settings, validated against the section the
            backend declares (``settings_section``) when it is resolved. They
            stay a mapping here because a backend in its own distribution
            declares fields this schema cannot know. A backend without a
            section accepts none, and settings are only accepted with a
            ``kind``, because they belong to one backend.
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
        framework: The native framework, or ``None`` for a host provider.
    """

    name: str
    loaded: bool
    reason: str = ""
    factory: Any = field(default=None, repr=False, compare=False)
    framework: str | None = None


def _registered(framework: str | None = None) -> dict[str, Any]:
    registered = unique_entries(
        entry_points(group=NEIGHBOR_BACKEND_GROUP),
        NEIGHBOR_BACKEND_GROUP,
        DuplicateNeighborBackendError,
    )
    if framework is not None:
        group = f"{NEIGHBOR_BACKEND_GROUP}.{framework}"
        native = unique_entries(
            entry_points(group=group), group, DuplicateNeighborBackendError
        )
        overlap = set(registered) & set(native)
        if overlap:
            raise DuplicateNeighborBackendError(
                f"neighbour provider names {sorted(overlap)} occur in both "
                f"{NEIGHBOR_BACKEND_GROUP!r} and {group!r}; use distinct names."
            )
        registered.update(native)
    return registered


def _load(name: str, entry: Any) -> DiscoveredNeighborBackend:
    factory, reason = load_entry_point(entry)
    group = getattr(entry, "group", NEIGHBOR_BACKEND_GROUP)
    framework = (
        None
        if group == NEIGHBOR_BACKEND_GROUP
        else group.removeprefix(NEIGHBOR_BACKEND_GROUP + ".")
    )
    return DiscoveredNeighborBackend(name, not reason, reason, factory, framework)


def available_neighbor_backends(
    framework: str | None = None,
) -> list[DiscoveredNeighborBackend]:
    """Every registered neighbour-list backend, loaded or not, sorted by name.

    Imports every registered backend to find the ones that fail. Use
    :func:`get_neighbor_backend` to resolve one; it imports only that one.
    With ``framework``, include its native group alongside the host group.

    Raises:
        DuplicateNeighborBackendError: if two entry points register different
            objects under one name.
    """
    return [
        _load(name, entry) for name, entry in sorted(_registered(framework).items())
    ]


def get_neighbor_backend(
    name: str,
    settings: Mapping[str, Any] | None = None,
    *,
    framework: str | None = None,
) -> NeighborListBackend:
    """The backend registered under ``name``, built with ``settings``.

    Only that backend is imported.

    Args:
        name: A registered backend name, usually ``NeighborListConfig.kind``.
        settings: Its settings, usually ``NeighborListConfig.settings``,
            validated against the section the backend declares. ``None`` and
            an empty mapping give the section's defaults.
        framework: Include ``mace.neighbor_backends.<framework>``. The caller
            is responsible for using that framework's search protocol.

    Raises:
        NeighborBackendNotAvailableError: if no backend of that name is
            registered, if it failed to import, if the settings are not valid
            for it, or if what it built does not implement
            :class:`NeighborListBackend`. The message says which.
        DuplicateNeighborBackendError: if two entry points register different
            objects under one name.
    """
    registered = _registered(framework)
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
    backend = _build(name, discovered.factory, dict(settings or {}))
    if not isinstance(backend, NeighborListBackend):
        raise NeighborBackendNotAvailableError(
            f"the entry point {name!r} built a {type(backend).__name__}, which "
            f"has no search(positions, cutoff, geometry) method."
        )
    neighbor_capabilities(backend, name)
    return backend


def _build(name: str, factory: Any, settings: dict[str, Any]) -> Any:
    """Call ``factory`` with its settings validated against its own section."""
    section = getattr(factory, "settings_section", None)
    if section is None:
        if settings:
            raise NeighborBackendNotAvailableError(
                f"the neighbour-list backend {name!r} takes no settings, and "
                f"{sorted(settings)} were given."
            )
        return factory()
    if not (isinstance(section, type) and issubclass(section, ConfigSection)):
        raise NeighborBackendNotAvailableError(
            f"the neighbour-list backend {name!r} declares a settings_section "
            f"that is not a ConfigSection subclass: {section!r}."
        )
    try:
        validated = section.model_validate(settings)
    except ValidationError as failure:
        raise NeighborBackendNotAvailableError(
            f"the settings of the neighbour-list backend {name!r} are not "
            f"valid: {failure}"
        ) from failure
    return factory(validated)


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
