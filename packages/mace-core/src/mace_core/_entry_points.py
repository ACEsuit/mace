"""Shared entry-point metadata checks and optional-import handling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib.metadata import EntryPoint
from typing import Any


def unique_entries(
    entries: Iterable[EntryPoint], group: str, conflict: type[RuntimeError]
) -> dict[str, EntryPoint]:
    """Coalesce identical registrations; reject conflicting names before imports."""
    by_name: dict[str, EntryPoint] = {}
    for entry in entries:
        seen = by_name.get(entry.name)
        if seen is not None and seen.value != entry.value:
            declarations = sorted((_describe(seen), _describe(entry)))
            raise conflict(
                f"two entry points in {group!r} register the name {entry.name!r}: "
                f"{' and '.join(declarations)}. Uninstall one of the two "
                "distributions, or register one under another name."
            )
        by_name.setdefault(entry.name, entry)
    return by_name


def _describe(entry: EntryPoint) -> str:
    origin = (
        getattr(getattr(entry, "dist", None), "name", None) or "an unknown distribution"
    )
    return f"{entry.name} = {entry.value} (from {origin})"


def load_entry_point(entry: EntryPoint) -> tuple[Any, str]:
    """Return the loaded object and an empty reason, or None and the import error."""
    try:
        return entry.load(), ""
    except Exception as failure:
        return None, repr(failure)


def resolve_entry_point(
    entries: Mapping[str, EntryPoint],
    name: str,
    *,
    group: str,
    description: str,
    unavailable: type[RuntimeError],
) -> Any:
    """Load one named registration without importing any other provider."""
    if name not in entries:
        raise unavailable(
            f"no {description} is registered as {name!r}. The registered names "
            f"are {sorted(entries)}, from the entry point group {group!r}."
        )
    factory, reason = load_entry_point(entries[name])
    if reason:
        raise unavailable(
            f"the {description} {name!r} is registered but did not import "
            f"on this machine: {reason}. It is not being substituted."
        )
    return factory
