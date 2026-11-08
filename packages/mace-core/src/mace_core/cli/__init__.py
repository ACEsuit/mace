"""The ``mace`` command line.

One program whatever is installed. Its subcommands come from the packages that
implement them, through the ``mace.commands`` entry point group
(:mod:`mace_core.cli.registry`); this package declares none of its own, so it
imports no framework to show ``--help``.

Each command declares its arguments and owns its configuration loading
(:mod:`mace_core.cli.command`). Configuration errors are reported as a message
and exit status 2, the way argparse reports its own.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from mace_core.cli.command import Command
from mace_core.cli.registry import (
    ENTRY_POINT_GROUP,
    CommandConflictError,
    DiscoveredCommand,
    discover,
)
from mace_core.config.base import ConfigError

__all__ = [
    "ENTRY_POINT_GROUP",
    "Command",
    "CommandConflictError",
    "DiscoveredCommand",
    "build_parser",
    "discover",
    "format_error",
    "main",
]

#: Where the chosen command is kept on the parsed arguments.
_CHOSEN = "_mace_command"


def build_parser(
    commands: dict[tuple[str, ...], DiscoveredCommand],
) -> argparse.ArgumentParser:
    """The parser for every command given, nested by the words of their paths."""
    parser = argparse.ArgumentParser(
        prog="mace", description="Train, evaluate and manage models."
    )
    groups: dict[tuple[str, ...], argparse._SubParsersAction] = {
        (): parser.add_subparsers(title="commands", metavar="COMMAND")
    }
    for path in sorted(commands):
        for depth in range(1, len(path)):
            prefix = path[:depth]
            if prefix not in groups:
                group = groups[prefix[:-1]].add_parser(
                    prefix[-1], help=f"{prefix[-1]} commands"
                )
                group.set_defaults(**{_CHOSEN: group})
                groups[prefix] = group.add_subparsers(
                    title="commands", metavar="COMMAND"
                )
        record = commands[path]
        help_line = (
            record.command.help
            if record.command is not None
            else f"not available: {record.reason}"
        )
        leaf = groups[path[:-1]].add_parser(path[-1], help=help_line)
        if record.command is not None:
            record.command.add_to(leaf)
        leaf.set_defaults(**{_CHOSEN: record})
    parser.set_defaults(**{_CHOSEN: parser})
    return parser


def format_error(error: Exception) -> str:
    """Report configuration locations without dumping inputs or library URLs."""
    if isinstance(error, ValidationError):
        messages = []
        for detail in error.errors(include_url=False, include_input=False):
            location = ".".join(str(part) for part in detail["loc"])
            message = detail["msg"].removeprefix("Value error, ")
            messages.append(f"{location}: {message}" if location else message)
        return "\n  ".join(messages)
    return str(error)


def main(argv: Sequence[str] | None = None) -> int:
    """The console entry point. Returns a process exit status."""
    try:
        commands = discover()
    except CommandConflictError as error:
        print(f"mace: {error}", file=sys.stderr)
        return 2
    parser = build_parser(commands)
    arguments = parser.parse_args(argv)
    chosen = getattr(arguments, _CHOSEN)
    if isinstance(chosen, argparse.ArgumentParser):
        chosen.print_help(sys.stderr)
        return 2
    if chosen.command is None:
        print(
            f"mace {' '.join(chosen.path)}: declared by {chosen.distribution} "
            f"and not available: {chosen.reason}",
            file=sys.stderr,
        )
        return 2
    command: Command = chosen.command
    try:
        return command.run(arguments)
    except (ConfigError, ValidationError, ValueError, OSError) as error:
        print(f"mace {' '.join(chosen.path)}: {format_error(error)}", file=sys.stderr)
        return 2
