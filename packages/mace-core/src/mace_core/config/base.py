"""The config base: one file, validated once by pydantic, exported as a dict.

Everything the schema rejects, unknown keys included, is pydantic's
`ValidationError`; `ConfigError` is only for a file that cannot be read or
parsed. Nothing here knows about a command line: that is `cli`.
"""

import json
import os
import tomllib
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any, Never, Self

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


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate configuration key {key!r}")
        result[key] = value
    return result


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        # Check explicit keys before expanding YAML merges. An explicit key may
        # override an anchor, but two declarations in the same map are ambiguous.
        keys = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in keys
                keys.add(key)
            except TypeError as error:
                raise ValueError("configuration keys must be scalar values") from error
            if duplicate:
                raise ValueError(
                    f"duplicate configuration key {key!r} at line "
                    f"{key_node.start_mark.line + 1}"
                )
        return super().construct_mapping(node, deep=deep)


def default_choice(value: Any, *, default: str, common: AbstractSet[str]) -> Any:
    """Shared settings alone use the default; variant settings require a kind."""
    if not isinstance(value, Mapping):
        return value
    if isinstance(value.get("kind"), Mapping):
        raise ValueError(
            "kind must be a string beside its settings, not a nested section"
        )
    if "kind" not in value and value.keys() <= common:
        return {"kind": default, **value}
    return value


def read_config_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse one TOML, YAML or JSON config file, chosen by its (case-folded)
    extension, into a dict; an empty or comment-only file is `{}`. Raises
    `ConfigError` for an unknown extension, an unreadable file, a parse failure
    or a top level that is not a mapping."""
    path = Path(path)
    parsers = {
        ".toml": tomllib.loads,
        ".yaml": lambda text: yaml.load(text, Loader=_UniqueKeyLoader),
        ".yml": lambda text: yaml.load(text, Loader=_UniqueKeyLoader),
        ".json": lambda text: json.loads(text, object_pairs_hook=_unique_pairs),
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

    Schema design rules are checked by the architecture tests. Runtime
    construction validates user values and freezes their containers.
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
        return cls.model_validate(document)

    def to_resolved_dict(self) -> dict[str, Any]:
        """A JSON-compatible dict that loads back to this config, defaults
        filled in."""
        return self.model_dump(mode="json")

    def to_user_dict(self) -> dict[str, Any]:
        """A JSON-compatible dict that loads back to this config, holding only
        the fields that were set."""
        return self.model_dump(mode="json", exclude_unset=True)
