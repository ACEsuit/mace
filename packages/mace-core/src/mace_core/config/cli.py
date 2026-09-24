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
    """`--a.b.c value` or `--a.b.c=value`, in order, to `{"a.b.c": value}`. A
    value that is `null` or starts with `[` or `{` is parsed as JSON; any other
    value stays a string for pydantic to coerce. A repeated path keeps its last
    value, at its last position. A token that is not `--path`, a path without
    its value or with an empty key (`--a..b`), and a JSON value that does not
    parse are `ConfigError`s."""
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
        if "" in dotted_path.split("."):
            raise ConfigError(f"override {token} has an empty key in its path")
        if not has_inline_value:
            if position == len(argv):
                raise ConfigError(f"override {token} is missing its value")
            value = argv[position]
            position += 1
        parsed: Any = value
        if value == "null" or value[:1] in ("[", "{"):
            try:
                parsed = json.loads(value)
            except (ValueError, RecursionError) as error:
                raise ConfigError(
                    f"override {token} is not valid JSON: {error}"
                ) from error
        overrides.pop(dotted_path, None)  # a repeated path applies where it is last
        overrides[dotted_path] = parsed
    return overrides


def apply_overrides(
    document: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """A copy of the parsed file with each override written at its dotted path,
    in order. Mappings missing on the way are created and a parent that is not
    a mapping (a scalar, a list, null) is replaced. A mapping value merges key
    by key into a mapping already there, so `--model '{"depth": 3}'` keeps the
    file's other `model` keys; a list or a scalar replaces (a repeated mapping
    path from `parse_overrides` replaces too, since the mapping keeps one
    value per path). Values are written as given for pydantic to validate.
    Neither argument is written into: the copy takes dicts and lists apart, so
    two keys sharing one object (a YAML anchor) stop sharing; a value that
    contains itself is a `ConfigError`."""
    try:
        copy = _copy_tree(document)
        for dotted_path, value in overrides.items():
            _write(copy, dotted_path.split("."), value)
    except RecursionError:
        raise ConfigError(
            "the config contains a value that refers to itself or is nested too deeply"
        ) from None
    return copy


def _write(mapping: dict[str, Any], keys: list[str], value: Any) -> None:
    *parent_keys, last_key = keys
    for key in parent_keys:  # create mappings on the way; replace non-mappings
        if not isinstance(mapping.get(key), dict):
            mapping[key] = {}
        mapping = mapping[key]
    if isinstance(value, Mapping) and isinstance(mapping.get(last_key), dict):
        for key, item in value.items():  # a JSON key is one key, dots included
            _write(mapping[last_key], [key], item)
    else:
        mapping[last_key] = _copy_tree(value)


def _copy_tree(value: Any) -> Any:
    """Copy the dicts and lists of a document, scalars as they are. Unlike
    `copy.deepcopy`, two keys sharing one object get two copies."""
    if isinstance(value, Mapping):
        return {key: _copy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    return value
