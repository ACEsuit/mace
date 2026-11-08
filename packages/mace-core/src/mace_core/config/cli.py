"""Apply explicit configuration overrides before schema validation."""

from collections.abc import Mapping, MutableMapping
from typing import Any

from mace_core.config.base import ConfigError

__all__ = ["apply_overrides", "set_value"]


def set_value(document: MutableMapping[str, Any], path: str, value: Any) -> None:
    """Set or merge one dotted path; refuse to traverse a non-mapping section."""
    *sections, key = path.split(".")
    target = document
    walked = []
    for section in sections:
        walked.append(section)
        existing = target.setdefault(section, {})
        if not isinstance(existing, MutableMapping):
            raise ConfigError(
                f"cannot set {path}: {'.'.join(walked)} holds "
                f"{type(existing).__name__}, not a section"
            )
        target = existing
    _set_or_merge(target, key, value)


def apply_overrides(
    document: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy the document and apply dotted paths or nested mappings in order.

    Mapping values merge; other values replace. Containers are copied without
    preserving YAML aliases, so overriding one head does not change another.
    """
    result = _copy_tree(document)
    for path, value in overrides.items():
        set_value(result, path, value)
    return result


def _set_or_merge(mapping: MutableMapping[str, Any], key: str, value: Any) -> None:
    if isinstance(value, Mapping) and isinstance(mapping.get(key), MutableMapping):
        for child, item in value.items():
            _set_or_merge(mapping[key], child, item)
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
