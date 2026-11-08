"""A command declares its arguments and executes the parsed namespace."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["Command"]


@dataclass(frozen=True)
class Command:
    """One subcommand, registered by the package that implements it."""

    help: str
    run: Callable[[argparse.Namespace], int]
    arguments: Callable[[argparse.ArgumentParser], object] | None = None

    def add_to(self, parser: argparse.ArgumentParser) -> None:
        if self.arguments is not None:
            self.arguments(parser)
