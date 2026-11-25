"""Head names: the one default, and what a name has to be.

A head is one level of theory a model is fitted to. Its name keys every
per-head setting and value, so it lives here, where the configuration, the
data layer and the output layer can all read it without importing each other.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints

__all__ = ["DEFAULT_HEAD", "HeadName"]

#: The head of a model fitted to one level of theory.
DEFAULT_HEAD = "Default"

#: A head name: at least one character that is not whitespace.
HeadName = Annotated[str, StringConstraints(pattern=r"\S")]
