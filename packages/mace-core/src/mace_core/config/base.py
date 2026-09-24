"""One config file, one validation, two exports.

`ReforgeBaseConfig.load(config_file)` reads one TOML, YAML or JSON file into a
dict (`read_config_file`, public) and validates it once with pydantic
(`from_dict`, public, for a caller that edits the dict first: `cli` does, for a
command line). Everything the schema rejects, unknown keys included, is
pydantic's `ValidationError` at its dotted location; `ConfigError` is raised
only for a file that cannot be read or parsed. `to_resolved_dict` is the full
JSON-native dump, itself a reloadable config file; `to_user_dict` holds only
what was set. A kinds field (a discriminated union on `kind`) is an ordinary
pydantic feature written in the tagged form, `loss: {kind: huber, delta: 0.1}`;
nothing here knows about it. Nothing here knows about a command line either;
the `--a.b value` grammar is `cli`'s.
"""

import json
import os
import sys
from collections.abc import Iterator, Mapping
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = ["ConfigError", "ConfigSection", "ReforgeBaseConfig", "read_config_file"]


class ConfigError(ValueError):
    """A config file, or a command-line override (`cli`), that cannot be read or
    parsed. Schema errors are pydantic's."""


def read_config_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse one TOML, YAML or JSON config file, chosen by its (case-folded)
    extension, into a dict; an empty or comment-only file is `{}`. Raises
    `ConfigError` for an unknown extension, an unreadable file, a parse failure,
    a top level that is not a mapping, or a value that contains itself (a YAML
    anchor inside itself), which no schema could export again."""
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
    except (ValueError, yaml.YAMLError, RecursionError) as error:  # toml, json: Value
        raise ConfigError(f"cannot parse config file {path}: {error}") from error
    if document is None:  # YAML reads an empty or comment-only file as None
        return {}
    if not isinstance(document, dict):
        raise ConfigError(
            f"config file {path} must be a mapping of keys to values at the top "
            f"level, not {type(document).__name__}"
        )
    try:  # the stdlib's cycle detector; shared siblings pass, only a cycle fails
        json.dumps(document, default=str, skipkeys=True)
    except ValueError as error:
        raise ConfigError(
            f"config file {path} contains a value that refers to itself"
        ) from error
    return document


class ConfigSection(BaseModel):
    """A node of a config tree: unknown keys are errors at every level.

    Subclasses declare fields only. The definition-time checks keep unknown
    keys fatal (every reachable model is a section, `extra="forbid"` cannot be
    reopened) and the resolved export a fixed point (no sets, aliases,
    excluded or computed fields). A class left incomplete by a forward
    reference is checked on every `load` of a root that reaches it instead;
    a forward reference to a class local to a function cannot be resolved
    that way (pydantic cannot see the function's scope), so declare such
    classes at module level.
    """

    # inf/nan as the JSON constants Infinity and NaN, not pydantic's default
    # null, which would turn a value into a different one.
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="constants")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        extra = cls.model_config.get("extra")
        if extra != "forbid":
            raise TypeError(
                f"{cls.__name__} sets extra={extra!r}; a section keeps "
                "extra='forbid' so an unknown key stays an error"
            )
        if cls.__pydantic_complete__:  # else a forward reference: `load` checks it
            _check_field_declarations(cls)


def _leaf_types(annotation: Any) -> Iterator[Any]:
    """Every class or origin an annotation reaches through `Annotated`, unions and
    list, tuple or dict parameters. Non-type arguments (Literal values, `Field`
    metadata) come out too; callers test `isinstance(leaf, type)`."""
    yield get_origin(annotation) or annotation
    for argument in get_args(annotation):
        yield from _leaf_types(argument)


def _check_field_declarations(section: type[ConfigSection]) -> None:
    """The checks on one class's own fields; a violation is a `TypeError`."""
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
        for leaf in _leaf_types(field.annotation):
            if not isinstance(leaf, type):
                continue
            # Any set type, `set`, `frozenset` or an abstract one, dumps in an
            # order that varies between runs, so the export would not be stable.
            if issubclass(leaf, AbstractSet):
                raise TypeError(f"{where} is typed as a set; order varies. Use a list")
            if issubclass(leaf, BaseModel) and not issubclass(leaf, ConfigSection):
                raise TypeError(
                    f"{where} holds {leaf.__name__}, which is not a ConfigSection; "
                    "unknown keys under it would be dropped"
                )


def _check_sections_reached_by(root: type[ConfigSection]) -> None:
    """Resolve any section the root's tree reaches that a forward reference left
    incomplete at definition, and check every reached section. Checking again on
    each load is a few attribute reads per class; it saves remembering which
    classes were checked."""
    to_visit, seen = [root], set()
    while to_visit:
        section = to_visit.pop()
        if section in seen:
            continue
        seen.add(section)
        if not section.__pydantic_complete__:
            section.model_rebuild()  # resolves the forward reference or raises
        _check_field_declarations(section)
        for field in section.model_fields.values():
            for leaf in _leaf_types(field.annotation):
                if isinstance(leaf, type) and issubclass(leaf, ConfigSection):
                    to_visit.append(leaf)


class ReforgeBaseConfig(ConfigSection):
    """The root of a config tree: `load` a file, or `from_dict` a parsed one."""

    @classmethod
    def load(cls, config_file: str | os.PathLike[str]) -> Self:
        """Build the config from one TOML, YAML or JSON file."""
        return cls.from_dict(read_config_file(config_file))

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Build the config from a parsed document, checking the sections the
        class reaches first. This is `load` for a caller that edits the parsed
        file before validating it, such as a command line writing its flags into
        the dict `read_config_file` returned. The document is not written into
        (values under an `Any`-typed field are shared with it, not copied)."""
        _check_sections_reached_by(cls)
        return cls.model_validate(document)

    def to_resolved_dict(self) -> dict[str, Any]:
        """Every field, defaults filled, as JSON-native values in declaration
        order: a config file that loads back to this config (through TOML
        whenever no value is `None`)."""
        return self.model_dump(mode="json")

    def to_user_dict(self) -> dict[str, Any]:
        """Only what the file set, in the same shape. A loaded config carries
        the tag of every kinds field it wrote, so this loads back;
        a variant built in code without its tag (`Config(loss=Huber(delta=2))`)
        exports without `kind`: pass the tag, or use `to_resolved_dict`."""
        return self.model_dump(mode="json", exclude_unset=True)
