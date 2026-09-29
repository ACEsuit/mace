"""Finding the commands the installed packages declare.

A package declares a command as an entry point in the ``mace.commands`` group,
named by its path with dots between the words (``train``,
``model.export-config``) and pointing at a
:class:`~mace_core.cli.command.Command`. Nothing here names a package, so an
implementation is installed or it is not, and a third party adds a command the
same way.

Two packages declaring one command is refused rather than resolved by
installation order, since the same command line would then run different code
on two machines. A command whose import fails is still listed, so ``--help``
works on a machine missing a dependency, and running it says why it cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points

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
        try:
            command = entry.load()
        except Exception as failure:
            found[path] = DiscoveredCommand(path, distribution, None, repr(failure))
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
