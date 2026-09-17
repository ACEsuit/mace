"""Finding backends, and the difference between discovery and resolution.

Backends register through entry points, one group per framework, so a backend
can live in its own distribution and be found without anything in this package
naming it.

**Discovery records failures; resolution raises.** A backend whose import fails
because a CUDA library is missing is an ordinary fact about this machine, not a
reason to stop: it is recorded and listed. Asking for that backend by name is a
different matter and raises, with the recorded reason, because the caller named
something that cannot be delivered and a fallback would silently change the
numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Literal

__all__ = [
    "ENTRY_POINT_GROUPS",
    "BackendNotAvailableError",
    "DiscoveredBackend",
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
    for entry in entry_points(group=group):
        try:
            factory = entry.load()
        except Exception as failure:
            found.append(DiscoveredBackend(entry.name, framework, False, repr(failure)))
        else:
            found.append(DiscoveredBackend(entry.name, framework, True, "", factory))
    return sorted(found, key=lambda backend: backend.name)


def available_backends(framework: str = "torch") -> list[DiscoveredBackend]:
    """Every registered backend for a framework, loaded or not.

    Nothing here raises for a backend that failed to import. A machine without
    a CUDA runtime is expected to carry entry points it cannot load, and a
    listing that refused to run on such a machine would be useless exactly
    where it is most wanted.
    """
    return _discover(framework)


def get_backend(name: str, framework: str = "torch") -> Any:
    """The backend registered under ``name``, built.

    Raises:
        BackendNotAvailableError: If no backend of that name is registered, or
            if one is and it failed to import. The message distinguishes the
            two, lists what is available, and quotes the import failure, since
            "not found" and "found but broken" call for different fixes.
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
