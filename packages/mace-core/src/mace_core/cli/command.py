"""What a command is, and how its flags reach its configuration.

A command is declared by whichever package implements it: training by the
torch package, inference by torch and later by jax. This package owns only the
shape of a command and the ``mace`` program that finds and runs them, so the
command line is one program whatever is installed.

A command that runs from a configuration declares its schema and a few
explicit flags. The configuration is read from one file, each flag given on
the command line is written into what was read at the path it names, and the
result is validated once. There is no general override syntax: a setting
without a flag is set in the file.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mace_core.config.base import ConfigError, ReforgeBaseConfig, read_config_file

__all__ = ["Command", "ConfigFlag", "set_value"]


def set_value(document: MutableMapping[str, Any], path: str, value: Any) -> None:
    """Write ``value`` at the dotted ``path`` of a parsed configuration.

    Sections missing on the way are created. The value replaces whatever was
    at the path, and nothing beside it changes.

    Raises:
        ConfigError: A section on the way holds something that is not a
            mapping, since writing through it would throw that value away.
    """
    *sections, key = path.split(".")
    target = document
    walked: list[str] = []
    for section in sections:
        walked.append(section)
        existing = target.setdefault(section, {})
        if not isinstance(existing, MutableMapping):
            raise ConfigError(
                f"cannot set {path}: {'.'.join(walked)} holds "
                f"{type(existing).__name__}, not a section"
            )
        target = existing
    target[key] = value


@dataclass(frozen=True)
class ConfigFlag:
    """One command-line flag that sets one configuration value.

    Args:
        option: The flag, such as ``--seed``.
        path: The dotted configuration path it sets, such as
            ``runtime.seed``. Ignored when ``write`` is given.
        help: The line ``--help`` shows.
        type: Turns the command-line text into the value. The schema validates
            the value afterwards like any value read from a file.
        write: Places the value itself, for a flag whose path depends on what
            the file holds. Called with the parsed document and the value.
    """

    option: str
    path: str
    help: str
    type: Callable[[str], Any] = str
    write: Callable[[MutableMapping[str, Any], Any], None] | None = None

    @property
    def dest(self) -> str:
        return "flag_" + self.option.lstrip("-").replace("-", "_")

    def apply(self, document: MutableMapping[str, Any], value: Any) -> None:
        if self.write is not None:
            self.write(document, value)
        else:
            set_value(document, self.path, value)


@dataclass(frozen=True)
class Command:
    """A subcommand of ``mace``.

    Args:
        help: One line, shown in ``mace --help``.
        run: Runs the command on the parsed arguments and returns the exit
            status. For a command with a schema, ``arguments.configuration``
            is the validated configuration and ``arguments.config`` the file
            it was read from.
        schema: The configuration the command runs from. Given, the command
            takes ``--config`` and its ``flags``.
        flags: The explicit flags, each setting one value of the schema.
        arguments: Adds the command's other arguments to its parser.
    """

    help: str
    run: Callable[[argparse.Namespace], int]
    schema: type[ReforgeBaseConfig] | None = None
    flags: tuple[ConfigFlag, ...] = ()
    arguments: Callable[[argparse.ArgumentParser], object] | None = field(default=None)

    def add_to(self, parser: argparse.ArgumentParser) -> None:
        """Declare everything this command takes on its parser."""
        if self.schema is not None:
            parser.add_argument(
                "--config",
                type=Path,
                default=None,
                help="TOML, YAML or JSON. Without one, the run is the schema's "
                "defaults plus the flags given.",
            )
            for flag in self.flags:
                parser.add_argument(
                    flag.option,
                    dest=flag.dest,
                    metavar=flag.option.lstrip("-").upper(),
                    type=flag.type,
                    help=flag.help,
                )
        elif self.flags:
            raise TypeError("a command without a schema has no configuration to flag")
        if self.arguments is not None:
            self.arguments(parser)

    def configuration(self, arguments: argparse.Namespace) -> ReforgeBaseConfig:
        """The configuration the parsed arguments ask for: the file, with the
        flags given written into it, validated once."""
        if self.schema is None:
            raise TypeError("this command has no schema")
        path = getattr(arguments, "config", None)
        document = read_config_file(path) if path is not None else {}
        for flag in self.flags:
            value = getattr(arguments, flag.dest, None)
            if value is not None:
                flag.apply(document, value)
        return self.schema.from_dict(document)
