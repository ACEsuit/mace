"""Discover commands declared in the ``mace.commands`` entry-point group.

Names are dotted command paths (``train``, ``model.export-config``), pointing
at a :class:`~mace_core.cli.command.Command`. Duplicate paths and command/group
collisions raise. Failed imports remain listed so help works without optional
dependencies and command execution can report the cause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points

from mace_core._entry_points import load_entry_point
from mace_core.cli.command import Command

__all__ = ["ENTRY_POINT_GROUP", "CommandConflictError", "DiscoveredCommand", "discover"]

#: The group every package declares its commands into.
ENTRY_POINT_GROUP = "mace.commands"


class CommandConflictError(RuntimeError):
    """Two installed packages declare the same command."""


@dataclass(frozen=True)
class DiscoveredCommand:
    """One declared command, loaded or not.

    Attributes:
        path: The words that run it, such as ``("model", "export-config")``.
        distribution: The package that declares it.
        command: The command, when it loaded.
        reason: Why it did not load, when it did not.
    """

    path: tuple[str, ...]
    distribution: str
    command: Command | None = field(default=None, compare=False)
    reason: str = ""


def discover() -> dict[tuple[str, ...], DiscoveredCommand]:
    """Every declared command, by path.

    Raises:
        CommandConflictError: Two packages declare one path, or one path is
            both a command and a group of commands. Names the packages.
    """
    found: dict[tuple[str, ...], DiscoveredCommand] = {}
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        path = tuple(entry.name.split("."))
        distribution = entry.dist.name if entry.dist is not None else "<unknown>"
        if path in found:
            raise CommandConflictError(
                f"'mace {' '.join(path)}' is declared by both "
                f"{found[path].distribution} and {distribution}. Uninstall one."
            )
        command, reason = load_entry_point(entry)
        if reason:
            found[path] = DiscoveredCommand(path, distribution, None, reason)
            continue
        if not isinstance(command, Command):
            found[path] = DiscoveredCommand(
                path,
                distribution,
                None,
                f"{entry.value} is a {type(command).__name__}, not a Command",
            )
            continue
        found[path] = DiscoveredCommand(path, distribution, command)
    for path, record in found.items():
        for other in found:
            if other != path and other[: len(path)] == path:
                raise CommandConflictError(
                    f"'mace {' '.join(path)}' is a command in "
                    f"{record.distribution} and a group of commands in "
                    f"{found[other].distribution}"
                )
    return found
