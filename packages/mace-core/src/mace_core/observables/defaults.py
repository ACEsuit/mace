"""Loading a catalogue from a declarations file.

The shipped file is packaged data rather than a Python literal, because the
whole point of the declarative spec is that a property can be added without
touching code -- including the code that holds the defaults.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from mace_core.observables.spec import ObservableCatalogue

__all__ = [
    "DEFAULTS_RESOURCE",
    "load_catalogue",
    "load_default_catalogue",
]

#: Where the shipped declarations live, relative to the package root.
DEFAULTS_RESOURCE = "defaults/observables.yaml"


def _catalogue_from_text(text: str, source: str) -> ObservableCatalogue:
    document: Any = yaml.safe_load(text)
    if document is None:
        document = {}
    if not isinstance(document, dict):
        raise ValueError(
            f"{source}: an observable declarations file must be a mapping with "
            f"`inputs` and `observables` keys, not a "
            f"{type(document).__name__}."
        )
    return ObservableCatalogue.model_validate(document)


def load_catalogue(path: str | Path) -> ObservableCatalogue:
    """Load and validate a declarations file from disk."""
    path = Path(path)
    return _catalogue_from_text(path.read_text(encoding="utf-8"), str(path))


def load_default_catalogue() -> ObservableCatalogue:
    """The shipped declarations: energy plus its position and cell derivatives."""
    resource = files("mace_core").joinpath(DEFAULTS_RESOURCE)
    return _catalogue_from_text(resource.read_text(encoding="utf-8"), DEFAULTS_RESOURCE)
