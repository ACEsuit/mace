"""Discover kernel backends in one entry-point group per framework.

Listings retain import failures; resolving an unavailable backend raises.
Identical registrations coalesce, while conflicting names raise before imports.
No backend is substituted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Literal

from mace_core._entry_points import load_entry_point, unique_entries

__all__ = [
    "ENTRY_POINT_GROUPS",
    "BackendNotAvailableError",
    "DiscoveredBackend",
    "DuplicateBackendError",
    "available_backends",
    "get_backend",
]

Framework = Literal["torch", "jax"]

#: One group per framework. A torch backend is not a jax backend, and the
#: separation is in the group name rather than in a field nobody checks.
ENTRY_POINT_GROUPS: dict[str, str] = {
    "torch": "mace.kernel_backends.torch",
    "jax": "mace.kernel_backends.jax",
}


class BackendNotAvailableError(RuntimeError):
    """A backend was asked for by name and could not be delivered."""


class DuplicateBackendError(RuntimeError):
    """Two entry points registered different backends under one name."""


@dataclass(frozen=True)
class DiscoveredBackend:
    """One entry point, and whether it loaded.

    Attributes:
        name: The registered name.
        framework: Which framework's group it came from.
        loaded: Whether importing it succeeded on this machine.
        reason: Why it did not, when it did not. Kept so that asking for it by
            name can say what went wrong instead of only that it is missing.
    """

    name: str
    framework: str
    loaded: bool
    reason: str = ""
    factory: Any = field(default=None, repr=False, compare=False)


def _discover(framework: str) -> list[DiscoveredBackend]:
    group = ENTRY_POINT_GROUPS.get(framework)
    if group is None:
        raise ValueError(
            f"{framework!r} is not a framework this registry knows. The "
            f"frameworks are {sorted(ENTRY_POINT_GROUPS)}."
        )
    found = []
    for entry in unique_entries(
        entry_points(group=group), group, DuplicateBackendError
    ).values():
        factory, reason = load_entry_point(entry)
        found.append(
            DiscoveredBackend(entry.name, framework, not reason, reason, factory)
        )
    return sorted(found, key=lambda backend: backend.name)


def available_backends(framework: str = "torch") -> list[DiscoveredBackend]:
    """List every registered backend, including import failures.

    Raises:
        DuplicateBackendError: Different objects register under the same name.
    """
    return _discover(framework)


def get_backend(name: str, framework: str = "torch") -> Any:
    """The backend registered under ``name``, built.

    Raises:
        BackendNotAvailableError: If no backend of that name is registered, or
            if one is and it failed to import. The message distinguishes the
            two, lists what is available, and quotes the import failure, since
            "not found" and "found but broken" call for different fixes.
        DuplicateBackendError: If two entry points register different objects
            under one name, whichever name was asked for.
    """
    discovered = _discover(framework)
    by_name = {backend.name: backend for backend in discovered}
    if name not in by_name:
        raise BackendNotAvailableError(
            f"no {framework} kernel backend is registered as {name!r}. The "
            f"registered names are {sorted(by_name)}, from the entry point "
            f"group {ENTRY_POINT_GROUPS[framework]!r}."
        )
    backend = by_name[name]
    if not backend.loaded:
        raise BackendNotAvailableError(
            f"the {framework} kernel backend {name!r} is registered but did "
            f"not import on this machine: {backend.reason}. It is not being "
            f"substituted, because another backend is another set of numbers."
        )
    return backend.factory()
