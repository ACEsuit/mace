"""Discover kernel backends in one entry-point group per framework.

Listings retain import failures; resolving an unavailable backend raises.
Identical registrations coalesce, while conflicting names raise before imports.
No backend is substituted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Literal

from mace_core._entry_points import (
    load_entry_point,
    resolve_entry_point,
    unique_entries,
)

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


def _entries(framework: str) -> dict[str, EntryPoint]:
    group = ENTRY_POINT_GROUPS.get(framework)
    if group is None:
        raise ValueError(
            f"{framework!r} is not a framework this registry knows. The "
            f"frameworks are {sorted(ENTRY_POINT_GROUPS)}."
        )
    return unique_entries(entry_points(group=group), group, DuplicateBackendError)


def _discover(framework: str) -> list[DiscoveredBackend]:
    found = []
    for name, entry in sorted(_entries(framework).items()):
        factory, reason = load_entry_point(entry)
        found.append(DiscoveredBackend(name, framework, not reason, reason, factory))
    return found


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
    entries = _entries(framework)
    factory = resolve_entry_point(
        entries,
        name,
        group=ENTRY_POINT_GROUPS[framework],
        description=f"{framework} kernel backend",
        unavailable=BackendNotAvailableError,
    )
    return factory()
