"""Check schema-author rules once in the suite, outside configuration loading."""

from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from importlib import import_module
from pkgutil import iter_modules
from typing import Any, get_args, get_origin

import mace_core.config
from mace_core.config.base import ConfigSection
from pydantic import BaseModel


def _types_in(annotation: Any) -> Iterator[type]:
    """Every class an annotation mentions, containers included."""
    outer = get_origin(annotation) or annotation
    if isinstance(outer, type):
        yield outer
    for argument in get_args(annotation):
        yield from _types_in(argument)


def _check_section_fields(section: type[ConfigSection]) -> None:
    """Refuse, with a `TypeError`, what pydantic allows on a field but a config
    section cannot have: a field that would not load back from the export, or
    a child that would ignore unknown keys."""
    if section.model_computed_fields:
        name = next(iter(section.model_computed_fields))
        raise TypeError(
            f"{section.__name__}.{name} is a computed field; it would not load back"
        )
    for name, field in section.model_fields.items():
        where = f"{section.__name__}.{name}"
        if field.alias or field.validation_alias or field.serialization_alias:
            raise TypeError(f"{where} has an alias; a config key is its field name")
        if field.exclude:
            raise TypeError(f"{where} is excluded from dumps; it would not load back")
        for held in _types_in(field.annotation):
            # Any set type, `set`, `frozenset` or an abstract one, dumps in an
            # order that varies between runs, so the export would not be stable.
            if issubclass(held, AbstractSet):
                raise TypeError(f"{where} is typed as a set; order varies. Use a list")
            if issubclass(held, BaseModel) and not issubclass(held, ConfigSection):
                raise TypeError(
                    f"{where} holds {held.__name__}, which is not a ConfigSection; "
                    "unknown keys under it would be dropped"
                )


def test_production_config_sections_preserve_validation_and_round_trips():
    for module in iter_modules(mace_core.config.__path__):
        import_module(f"mace_core.config.{module.name}")
    pending: list[type[ConfigSection]] = [ConfigSection]
    seen = set()
    while pending:
        section = pending.pop()
        if section in seen:
            continue
        seen.add(section)
        pending.extend(section.__subclasses__())
        if not section.__module__.startswith("mace_core."):
            continue
        section.model_rebuild()
        assert section.model_config.get("extra") == "forbid", section
        assert section.model_config.get("frozen"), section
        _check_section_fields(section)
