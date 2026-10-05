"""The command-line half of a config: `--a.b value` tokens, and a parsed file
edited before validation.

`parse_overrides(tokens)` turns command-line tokens into a mapping of dotted
path to value; `apply_overrides(document, overrides)` writes such a mapping
into a copy of a parsed config file. A command line then builds its config as

    Config.from_dict(apply_overrides(read_config_file(path), parse_overrides(rest)))

and every error the schema reports is pydantic's, at the dotted location the
override named. Nothing in `base` knows about this module; a CLI that exposes
explicit flags instead can hand `apply_overrides` a mapping it built itself.
"""

import json
from collections.abc import Iterable, Mapping
from typing import Any

from mace_core.config.base import ConfigError

__all__ = ["apply_overrides", "parse_overrides"]


def parse_overrides(tokens: Iterable[str]) -> dict[str, Any]:
    """Turn command-line tokens into a dict of dotted path to value.

    Both `--a.b value` and `--a.b=value` give `{"a.b": value}`. A value that
    is `null` or starts with `[` or `{` is parsed as JSON; any other stays a
    string for pydantic to coerce. A repeated path keeps its last value. A
    token that fits none of this is a `ConfigError`."""
    if isinstance(tokens, str):
        raise TypeError(f"tokens is a string, {tokens!r}; pass a list of tokens")
    argv = list(tokens)
    overrides: dict[str, Any] = {}
    position = 0
    while position < len(argv):
        token = argv[position]
        position += 1
        dotted_path, has_inline_value, value = token[2:].partition("=")
        if not token.startswith("--") or not dotted_path:
            raise ConfigError(
                f"unknown config option '{token}'; options are --key.path value"
            )
        if not has_inline_value:
            if position == len(argv):
                raise ConfigError(f"override {token} is missing its value")
            value = argv[position]
            position += 1
        parsed: Any = value
        if value == "null" or value[:1] in ("[", "{"):
            try:
                parsed = json.loads(value)
            except ValueError as error:
                raise ConfigError(
                    f"override {token} is not valid JSON: {error}"
                ) from error
        overrides.pop(dotted_path, None)  # a repeated path applies where it is last
        overrides[dotted_path] = parsed
    return overrides


def apply_overrides(
    document: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a copy of a parsed config file with the overrides written in.

    Each value goes to its dotted path, in order. A mapping merges into a
    mapping already there, so `--model '{"depth": 3}'` keeps the file's other
    `model` keys; any other value replaces what was there."""
    copy = _copy_tree(document)
    for dotted_path, value in overrides.items():
        _set_at_path(copy, dotted_path.split("."), value)
    return copy


def _set_at_path(mapping: dict[str, Any], keys: list[str], value: Any) -> None:
    """Walk down the keys, creating dicts on the way (a parent that is not a
    dict is replaced by one), and set or merge the value at the last key."""
    *parent_keys, last_key = keys
    for key in parent_keys:
        if not isinstance(mapping.get(key), dict):
            mapping[key] = {}
        mapping = mapping[key]
    _set_or_merge(mapping, last_key, value)


def _set_or_merge(mapping: dict[str, Any], key: str, value: Any) -> None:
    """Set `mapping[key]`. A dict merges, key by key, into a dict already
    there; anything else replaces what was there."""
    if isinstance(value, Mapping) and isinstance(mapping.get(key), dict):
        for inner_key, item in value.items():
            _set_or_merge(mapping[key], inner_key, item)
    else:
        mapping[key] = _copy_tree(value)


def _copy_tree(value: Any) -> Any:
    """Copy the dicts and lists of a document, scalars as they are. Unlike
    `copy.deepcopy`, two keys sharing one object get two copies."""
    if isinstance(value, Mapping):
        return {key: _copy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    return value
