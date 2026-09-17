"""The base class every v1 configuration schema derives from.

A configuration is a tree of fields. A field holds either one value (`seed`,
`cutoff`) or a named group of further fields; such a group is a *section*.
The root of the tree subclasses `ReforgeBaseConfig`, every section
subclasses `ConfigSection`:

    class RadialSection(ConfigSection):
        num_bessel: int = 8
        cutoff: float = 5.0

    class ModelSection(ConfigSection):
        num_interactions: int = 2
        radial: RadialSection = RadialSection()

    class TrainConfig(ReforgeBaseConfig):
        seed: int = 1
        model: ModelSection = ModelSection()

Here `model` and `radial` are sections. In a TOML file a section is a table
(`[model.radial]`), in YAML/JSON a nested mapping, and on the command line a
dotted prefix (`--model.radial.cutoff 5.0`). Values come from three layers,
lowest precedence first:

    schema defaults < one config file (.toml/.yaml/.yml/.json) < dotted CLI overrides

Nothing else feeds a config: no environment variables, no dotenv files, so a
run is reproducible from its file and its command line alone.

Unknown keys are hard errors. A CLI key is checked against the schema's
dotted paths before anything is built; a key in the file, or inside a
JSON-valued override, is caught by pydantic (`extra="forbid"` on every level
of the tree). Either way the message names the key by its dotted path and,
when there is one, the nearest valid neighbour.

Field types are restricted to what survives a JSON round trip unchanged, so
that the resolved export is a fixed point: `set` and `frozenset` fields are
rejected when the schema class is defined, because their element order is not
stable across interpreter runs. Use a list.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

import tomli
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["ConfigError", "ConfigSection", "ReforgeBaseConfig", "read_config_file"]

#: Config file extensions this module reads, keyed to their parsers.
_FILE_PARSERS = {
    ".toml": tomli.loads,
    ".yaml": yaml.safe_load,
    ".yml": yaml.safe_load,
    ".json": json.loads,
}


def _sections_in(
    annotation: Any, inside: bool = False
) -> Iterator[tuple[type[BaseModel], bool]]:
    """Every section class a field annotation can hold (`Radial`, `Radial | None`,
    `list[Radial]`), with whether it sits inside a dict/list/tuple, where an
    error location has a key or index before the section's own field names."""
    origin = get_origin(annotation)  # `list` for `list[X]`; None for a plain class
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield annotation, inside
    elif origin in (Union, UnionType):
        for arg in get_args(annotation):
            yield from _sections_in(arg, inside)
    elif origin in (dict, list, tuple):
        for arg in get_args(annotation):
            yield from _sections_in(arg, True)


def _section_of(annotation: Any) -> tuple[type[BaseModel] | None, bool]:
    """The one section a field can hold, and whether it is inside a collection."""
    found = dict(_sections_in(annotation))
    return next(iter(found.items())) if found else (None, False)


def _contains_a_set(annotation: Any) -> bool:
    """A bare `set`, a `set[X]`, or a set anywhere inside, e.g. `list[set[int]]`."""
    if annotation in (set, frozenset) or get_origin(annotation) in (set, frozenset):
        return True
    return any(_contains_a_set(arg) for arg in get_args(annotation))


def _check_schema(model: type[BaseModel]) -> None:
    """Fail at class definition for a field shape the contract cannot keep.

    Each rule protects one guarantee: no sets (order is not stable across
    runs, so the export would not be a fixed point); no aliases or computed
    fields (the export would not validate back); one section class per field
    (a value must not become whichever alternative happens to accept it, and
    the CLI needs one set of valid keys under a field); every section is a
    `ConfigSection` (a plain `BaseModel` ignores unknown keys, so a typo
    would vanish, and skips these checks).
    """

    def reject(name: str, reason: str) -> None:
        raise TypeError(f"{model.__name__}.{name} {reason}")

    for name in model.model_computed_fields:
        reject(name, "is a computed field; the export must validate back, so drop it")
    for name, field in model.model_fields.items():
        if field.alias or field.validation_alias or field.serialization_alias:
            reject(name, "has an alias; config keys are field names, so drop it")
        if _contains_a_set(field.annotation):
            reject(
                name,
                "is typed as a set; set order is not stable across runs. Use a list",
            )
        sections = {section for section, _ in _sections_in(field.annotation)}
        if len(sections) > 1:
            reject(
                name,
                "is a union of sections; give each alternative its own optional "
                "field, e.g. `huber: HuberLoss | None`",
            )
        for section in sections:
            if not issubclass(section, ConfigSection):
                reject(
                    name,
                    f"holds {section.__name__}, which is not a ConfigSection; "
                    f"subclass it",
                )


class ConfigError(ValueError):
    """A config file or override the schema rejects.

    Raised for a missing, unparsable or malformed file, an unknown key and an
    override the CLI parser cannot make sense of. The message names the file
    or the offending key by its dotted path, and suggests the nearest valid
    key when there is a close match.
    """


class ConfigSection(BaseModel):
    """A nested section of a configuration: a table in the file, a dotted
    prefix on the command line. Unknown keys are errors here too."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        _check_schema(cls)


class ReforgeBaseConfig(ConfigSection):
    """Root of a configuration tree. Subclass it; nest `ConfigSection`s in it.

    `load()` reads a file and applies overrides. Constructing the class
    directly behaves like a plain Pydantic model.
    """

    @classmethod
    def load(
        cls,
        config_file: str | Path | None = None,
        cli_overrides: Sequence[str] = (),
    ) -> Self:
        """Build the config from defaults, then the file, then the overrides.

        `cli_overrides` is the argument list after the program name, e.g.
        `["--model.num_interactions", "3", "--seed=7"]`: a dotted path names
        a field at any depth. A value starting with `[` or `{`, or the word
        `null`, is JSON, so a whole section, a list or a dict can be given;
        any other value is a string pydantic converts to the field's type.
        An override merges into the file, and into earlier overrides, like a
        section does: a dict-valued field gains or replaces entries, so an
        entry cannot be removed from the command line; a list-valued field
        is replaced whole.

        Raises `ConfigError` for an unknown key, an unreadable file or an
        unparsable override, and pydantic's `ValidationError` for a value of
        the wrong type.
        """
        values: dict[str, Any] = {}
        if config_file is not None:
            values = read_config_file(config_file)
        values = _apply_overrides(cls, values, cli_overrides)
        try:
            return cls.model_validate(values)
        except ValidationError as error:
            unknown = _unknown_key_messages(cls, error)
            if not unknown:
                raise
            raise ConfigError("\n".join(unknown)) from error

    def to_resolved_dict(self) -> dict[str, Any]:
        """Every field, defaults included, as JSON-native values, in schema
        order. Loading the result back and resolving again gives the same
        dict. (TOML has no null, so a `None` can only go out as YAML or JSON.)"""
        return self.model_dump(mode="json")

    def to_user_dict(self) -> dict[str, Any]:
        """Only the fields the file and the overrides set, as JSON-native
        values: what the user wrote, for the model metadata."""
        return self.model_dump(mode="json", exclude_unset=True)


def read_config_file(path: str | Path) -> dict[str, Any]:
    """Parse one config file, choosing the parser by extension.

    An empty TOML or YAML file is an empty config. Raises `ConfigError` for a missing
    file, an unknown extension, a file its parser rejects, or a file whose
    top level is not a table.
    """
    path = Path(path)
    parser = _FILE_PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ConfigError(
            f"cannot read config file {path}: unknown extension {path.suffix!r}; "
            f"expected one of {', '.join(_FILE_PARSERS)}"
        )
    try:
        values = parser(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(
            f"cannot read config file {path}: {error.strerror}"
        ) from error
    except (ValueError, yaml.YAMLError) as error:  # tomli/json errors are ValueErrors
        raise ConfigError(f"cannot parse config file {path}: {error}") from error
    if values is None:
        return {}
    # TOML always yields a table, but a YAML or JSON file can hold a list or a
    # scalar, which would crash the merge with the overrides instead of
    # naming the file.
    if not isinstance(values, dict):
        raise ConfigError(
            f"config file {path} must hold a table of keys at the top level, "
            f"not a {type(values).__name__}"
        )
    return values


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """`base` overlaid with `update`, recursing where both hold a dict."""
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_overrides(
    model: type[BaseModel], values: dict[str, Any], cli_overrides: Sequence[str]
) -> dict[str, Any]:
    """`values` with the `--a.b value` and `--a.b=value` pairs merged in, in order."""
    valid = list(_dotted_paths(model))
    valid_set = set(valid)
    unknown: list[str] = []
    tokens = iter(cli_overrides)

    for token in tokens:
        option = token.removeprefix("--")
        if "=" in option:
            name, value = option.split("=", 1)
        else:
            name, value = option, None
        if not token.startswith("--") or not name:
            raise ConfigError(
                f"unknown config option {token!r}; overrides are written --key value"
            )
        if value is None:
            value = next(tokens, None)
            if value is None:
                raise ConfigError(f"override --{name} is missing its value")

        if name not in valid_set:
            unknown.append(_unknown_key_message(name, valid))
            continue

        if value == "null" or value.startswith(("[", "{")):
            try:
                value = json.loads(value)
            except ValueError as error:
                raise ConfigError(
                    f"override --{name} is not valid JSON: {error}"
                ) from error

        # Nested under its path, an override merges by the file's rule.
        override: Any = value
        for section in reversed(name.split(".")):
            override = {section: override}
        values = _deep_update(values, override)

    if unknown:
        raise ConfigError("\n".join(unknown))

    return values


def _dotted_paths(model: type[BaseModel], prefix: str = "") -> Iterator[str]:
    """Every field of the tree as a dotted path, sections included.

    A section inside a dict or list is not descended into: the CLI addresses
    such a field only as a whole, with a JSON value.
    """
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        yield path
        section, inside_collection = _section_of(field.annotation)
        if section is not None and not inside_collection:
            yield from _dotted_paths(section, f"{path}.")


def _unknown_key_message(key: str, candidates: Sequence[str]) -> str:
    message = f"unknown config key {key!r}"
    closest = difflib.get_close_matches(key, candidates, n=1)
    if closest:
        message += f"; did you mean {closest[0]!r}?"
    return message


def _unknown_key_messages(model: type[BaseModel], error: ValidationError) -> list[str]:
    """Pydantic's unknown-key errors as messages that name the full dotted
    path and the closest valid key at that level.

    The error location is walked against the schema to find the section whose
    fields are the candidates: a field name moves into its section, a dict key
    or list index stays in it, and the tag pydantic inserts for a
    `Section | scalar` field (the class name) is dropped from the path.
    """
    messages = []
    for item in error.errors():
        # pydantic's error code for a key that matches no field (extra="forbid").
        # Every other code, e.g. a wrong type, is left for `load()` to re-raise.
        if item["type"] != "extra_forbidden":
            continue
        *location, key = (str(part) for part in item["loc"])
        section: type[BaseModel] | None = model
        inside_collection = False
        names = []
        for part in location:
            if section is not None and part == section.__name__:
                continue
            names.append(part)
            if inside_collection:
                inside_collection = False
            elif section is not None and part in section.model_fields:
                section, inside_collection = _section_of(
                    section.model_fields[part].annotation
                )
            else:
                section = None
        candidates = list(section.model_fields) if section is not None else []
        prefix = "".join(f"{name}." for name in names)
        messages.append(
            _unknown_key_message(f"{prefix}{key}", [f"{prefix}{c}" for c in candidates])
        )
    return messages
