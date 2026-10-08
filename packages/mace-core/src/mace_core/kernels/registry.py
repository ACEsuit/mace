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

**A name is registered once.** Two entry points under one name would otherwise
resolve by install order, so a third-party distribution could shadow
``reference`` and change the numbers with nothing said. Two declarations that
name the same object are the same registration and are accepted; two that name
different objects raise wherever discovery runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Literal

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


def _describe(entry: Any) -> str:
    """An entry point as ``name = value`` plus the distribution declaring it."""
    distribution = getattr(entry, "dist", None)
    origin = getattr(distribution, "name", None) or "an unknown distribution"
    return f"{entry.name} = {getattr(entry, 'value', '?')} (from {origin})"


def _refuse_duplicates(entries: list[Any], group: str) -> list[Any]:
    """The entries with identical repeats dropped, or a raise on a conflict."""
    by_name: dict[str, Any] = {}
    for entry in entries:
        seen = by_name.get(entry.name)
        if seen is None:
            by_name[entry.name] = entry
            continue
        if getattr(seen, "value", None) == getattr(entry, "value", None):
            continue
        raise DuplicateBackendError(
            f"two entry points in {group!r} register the backend name "
            f"{entry.name!r}: {_describe(seen)} and {_describe(entry)}. "
            f"Neither is chosen, because the choice would follow install order "
            f"and a different backend is a different set of numbers. Uninstall "
            f"one of the two distributions, or register one under another name."
        )
    return list(by_name.values())


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
    for entry in _refuse_duplicates(list(entry_points(group=group)), group):
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

    Raises:
        DuplicateBackendError: If two entry points register different objects
            under one name. That is not a fact about this machine but an
            ambiguity no listing can resolve.
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
