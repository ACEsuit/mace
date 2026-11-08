"""The config base: one file, validated once by pydantic, exported as a dict.

Everything the schema rejects, unknown keys included, is pydantic's
`ValidationError`; `ConfigError` is only for a file that cannot be read or
parsed. Nothing here knows about a command line: that is `cli`.
"""

import json
import os
import tomllib
from collections.abc import Iterator, Mapping
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any, Never, Self, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict

__all__ = ["BaseConfig", "ConfigError", "ConfigSection", "read_config_file"]


def _immutable(*args: Any, **kwargs: Any) -> Never:
    raise TypeError(
        "configuration containers are immutable; use model_copy(update=...)"
    )


class _FrozenDict(dict):
    """A JSON-compatible mapping whose ordinary mutation operations are refused."""

    __setitem__ = __delitem__ = __ior__ = _immutable
    clear = pop = popitem = setdefault = update = _immutable

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        return self

    def __reduce__(self):
        # DataLoader workers may receive a configuration through spawn.
        return type(self), (dict(self),)


class _FrozenList(list):
    """Keep list serialization and equality while refusing in-place changes."""

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _immutable
    append = clear = extend = insert = pop = remove = reverse = sort = _immutable

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        return self

    def __reduce__(self):
        return type(self), (list(self),)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


class ConfigError(ValueError):
    """A config file, or a command-line override (`cli`), that cannot be read or
    parsed. Schema errors are pydantic's."""


def read_config_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse one TOML, YAML or JSON config file, chosen by its (case-folded)
    extension, into a dict; an empty or comment-only file is `{}`. Raises
    `ConfigError` for an unknown extension, an unreadable file, a parse failure
    or a top level that is not a mapping."""
    path = Path(path)
    parsers = {
        ".toml": tomllib.loads,
        ".yaml": yaml.safe_load,
        ".yml": yaml.safe_load,
        ".json": json.loads,
    }
    parse = parsers.get(path.suffix.lower())
    if parse is None:
        raise ConfigError(
            f"cannot read config file {path}: unknown extension {path.suffix!r}; "
            "use .toml, .yaml, .yml or .json"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"cannot read config file {path}: {error}") from error
    try:
        document = parse(text)
    except (ValueError, yaml.YAMLError) as error:  # toml, json: ValueError
        raise ConfigError(f"cannot parse config file {path}: {error}") from error
    if document is None:  # YAML reads an empty or comment-only file as None
        return {}
    if not isinstance(document, dict):
        raise ConfigError(
            f"config file {path} must be a mapping of keys to values at the top "
            f"level, not {type(document).__name__}"
        )
    return document


class ConfigSection(BaseModel):
    """A validated config node with immutable attributes and nested containers.

    Unknown keys are errors. ``model_copy(update=...)`` validates the updated
    section, including its cross-field constraints. Dumps remain ordinary
    editable dictionaries and lists.

    Subclasses declare fields only. A subclass that would undo either rule, or
    hold what does not load back from the export (a set, an alias, an excluded
    or computed field, a plain `BaseModel`), is a `TypeError` when it is
    defined, or on `load` if a forward reference left it incomplete. Declare a
    forward-referenced class at module level: pydantic cannot resolve one
    local to a function.
    """

    # inf/nan as the JSON constants Infinity and NaN, not pydantic's default
    # null, which would turn a value into a different one.
    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="constants")

    def model_post_init(self, context: Any) -> None:
        for name in type(self).model_fields:
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Copy with validated changes, retaining which fields the user supplied.

        Pydantic's default copy trusts updates. A config copy is another public
        construction path and must enforce the same constraints as its constructor.
        Existing nested sections retain their own explicit-field provenance.
        """
        if not update:
            return super().model_copy(deep=deep)
        source = super().model_copy(deep=True) if deep else self
        values = {name: getattr(source, name) for name in type(self).model_fields}
        values.update(update)
        result = type(self).model_validate(_validation_input(values))
        for name in type(self).model_fields:
            _preserve_explicit_fields(getattr(result, name), values[name])
        object.__setattr__(
            result, "__pydantic_fields_set__", self.model_fields_set | update.keys()
        )
        return result

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        extra = cls.model_config.get("extra")
        if extra != "forbid":
            raise TypeError(
                f"{cls.__name__} sets extra={extra!r}; a section keeps "
                "extra='forbid' so an unknown key stays an error"
            )
        if not cls.model_config.get("frozen"):
            raise TypeError(
                f"{cls.__name__} sets frozen=False; a section stays frozen "
                "so a validated config is not changed behind the validation"
            )
        # A forward reference leaves the field types unknown for now;
        # `_check_schema` checks the section on load instead.
        if cls.__pydantic_complete__:
            _check_section_fields(cls)


def _validation_input(value: Any) -> Any:
    """Before-validators receive the same plain values as a config file."""
    if isinstance(value, ConfigSection):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return {key: _validation_input(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_validation_input(item) for item in value)
    return value


def _preserve_explicit_fields(value: Any, original: Any) -> None:
    """Keep nested user/default provenance when a resolved section is copied."""
    if isinstance(value, ConfigSection) and isinstance(original, ConfigSection):
        object.__setattr__(
            value, "__pydantic_fields_set__", original.model_fields_set.copy()
        )
        for name in type(value).model_fields:
            _preserve_explicit_fields(getattr(value, name), getattr(original, name))
    elif isinstance(value, Mapping) and isinstance(original, Mapping):
        for key in value.keys() & original.keys():
            _preserve_explicit_fields(value[key], original[key])
    elif isinstance(value, (list, tuple)) and isinstance(original, (list, tuple)):
        for item, source in zip(value, original, strict=False):
            _preserve_explicit_fields(item, source)


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


def _check_schema(root: type[ConfigSection]) -> None:
    """Run `_check_section_fields` on the root and every section under it.

    A section is checked when it is defined, unless a forward reference left
    its field types unknown then. Such a section is resolved and checked here,
    on load. The others are checked a second time, which costs less than
    tracking which ones were skipped."""
    to_visit, seen = [root], set()
    while to_visit:
        section = to_visit.pop()
        if section in seen:
            continue
        seen.add(section)
        if not section.__pydantic_complete__:
            section.model_rebuild()  # resolves the forward reference or raises
        _check_section_fields(section)
        for field in section.model_fields.values():
            for held in _types_in(field.annotation):
                if issubclass(held, ConfigSection):
                    to_visit.append(held)


class BaseConfig(ConfigSection):
    """Base class for the root of a config schema: subclass it and declare the
    fields, then `load` a file or `from_dict` a parsed one."""

    @classmethod
    def load(cls, config_file: str | os.PathLike[str]) -> Self:
        """Build the config from one TOML, YAML or JSON file."""
        return cls.from_dict(read_config_file(config_file))

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Build the config from an already parsed file, such as one a command
        line applied its overrides to."""
        _check_schema(cls)
        return cls.model_validate(document)

    def to_resolved_dict(self) -> dict[str, Any]:
        """A JSON-compatible dict that loads back to this config, defaults
        filled in."""
        return self.model_dump(mode="json")

    def to_user_dict(self) -> dict[str, Any]:
        """A JSON-compatible dict that loads back to this config, holding only
        the fields that were set."""
        return self.model_dump(mode="json", exclude_unset=True)
