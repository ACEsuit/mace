"""The config base: files and dotted command-line overrides into one validated
pydantic tree, and the tree back out as a dict.

`load` turns every file (any number, in order) and every override into leaf
updates `(path, value, source)`, walks each path once through the schema (the
one schema-dependent step: unknown keys and wrong shapes are rejected with full
dotted paths, a bare kind name at a kinds field becomes its `kind`, and the
kinds fields the path enters are recorded on the update), merges them
set-at-path into one plain dict, validates that dict once, and finally warns
for each override that lost its effect. Precedence is defaults < files in
order < overrides in order. Nothing else feeds a config: no environment, no
dotenv.

Wire form of a kinds field (`Annotated[A | B, Field(discriminator="kind")]`):
a mapping with a kind name holding that kind's settings (`huber: {delta: 0.1}`,
any number of them) and `kind: huber` selecting the one that runs; a bare
`huber` is `{kind: huber}`; a single kind key selects itself; `{}` keeps the
schema default. Two or more kind keys without a selection, a required kinds
field with nothing written, and the tag under a kind are the one kinds error,
rendered with the path, the fix and the source of each kind key. The exports
write the kind that ran as `{huber: {fields}}`, without the tag. `kind` is the
tag only under a kinds field: the same class as a plain section field keeps
`kind` as an ordinary key on input and export.

An override loses its effect in two ways, and only these warn (a file never
does): a later override writes at its path, above it, or below a value of its
that was not a mapping (`{}` never clears a mapping); or it wrote under a kind
that does not run at its kinds field.
"""

from __future__ import annotations

import copy
import difflib
import json
import sys
import warnings
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any, NamedTuple, TypeVar, get_args, get_origin

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
    model_validator,
)
from pydantic_core import (
    ErrorDetails,
    PydanticCustomError,
    PydanticUndefined,
)
from typing_extensions import Self

from mace_core.config._schema_rules import (
    check_fields,
    check_model_config,
    is_section,
    kinds_fields_of,
    kinds_of,
    members_of,
    reachable_sections,
    unwrap,
)

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_PARSERS = {
    ".toml": tomllib.loads,
    ".yaml": yaml.safe_load,
    ".yml": yaml.safe_load,
    ".json": json.loads,
}
#: The one error `_to_tagged_form` raises through pydantic; `load` renders it.
_KINDS_ERROR = "kinds"


class ConfigError(ValueError):
    """A file or command line the schema cannot take; the message names the
    key and the fix."""


class ConfigWarning(UserWarning):
    """An override that had no effect on the config that runs."""


class KindEntered(NamedTuple):
    """A kinds field an update's path passes through under one of its kind
    keys: the field's path, its kind names and the key entered."""

    field: tuple[Any, ...]
    kinds: tuple[str, ...]
    key: str


class Update(NamedTuple):
    """One leaf update; `source` is the file's path or the override as typed.
    The walk fills `under_kinds` with every kinds field the path enters under a
    kind key, and `selects` with the kinds field whose tag the path ends at."""

    path: tuple[Any, ...]
    value: Any
    source: str
    under_kinds: tuple[KindEntered, ...] = ()
    selects: tuple[Any, ...] | None = None


#: The node for the `kind` slot under a kinds field.
_TAG = object()


class _KindPosition(NamedTuple):
    """A variant class reached through its kinds field, where `kind` is the
    tag and not a key. The same class as a plain field is walked as a section."""

    variant: type[ConfigSection]


def _narrowed(node: Any) -> Any:
    """`Annotated` stripped and `X | None` stepped through to `X`."""
    members = [m for m in members_of(node) if m is not type(None)]
    return unwrap(members[0]) if len(members) == 1 else unwrap(node)


def _step(node: Any, key: Any) -> tuple[Any, list[Any] | None]:
    """One step of the walk: the child node under `key` (None when the key
    is not valid there) and the keys valid at `node` (None when any key is,
    `[]` when the node is written whole)."""
    node = unwrap(node)
    if isinstance(node, _KindPosition):
        fields = node.variant.model_fields
        keys = [name for name in fields if name != "kind"]
        return (fields[key].annotation if key in keys else None), keys
    if (kinds := kinds_of(node)) is not None:
        keys = ["kind", *kinds]
        if key == "kind":
            return _TAG, keys
        return (_KindPosition(kinds[key]) if key in kinds else None), keys
    if is_section(node):
        fields = node.model_fields
        return (fields[key].annotation if key in fields else None), list(fields)
    node = _narrowed(node)
    origin = get_origin(node)
    if origin is dict:
        return get_args(node)[1], None
    if node in (Any, object, dict):
        return Any, None
    if origin in (list, tuple) or node in (list, tuple):
        if not isinstance(key, int):
            return None, []
        args = get_args(node)
        if origin is tuple and Ellipsis not in args and args:
            return (args[key] if key < len(args) else Any), []
        return (args[0] if args else Any), []
    return None, []


def _dotted(path: tuple[Any, ...]) -> str:
    return ".".join(map(str, path))


def _as_shown(value: Any) -> str:
    """JSON where it can, `str` where it cannot (a TOML date, a YAML set)."""
    return json.dumps(value, default=str)


def _as_typed(value: Any) -> str:
    """A command-line value as typed (a str), anything else as JSON."""
    return value if isinstance(value, str) else _as_shown(value)


def _unknown_key(
    parent: Any, keys: list[Any] | None, path: tuple[Any, ...]
) -> ConfigError:
    """D1: the key by its dotted path, then the nearest neighbour, the kinds
    of a kinds field, or why the tag is not a key under a kind."""
    message = f"unknown config key '{_dotted(path)}'"
    above = _dotted(path[:-1])
    if keys == []:
        return ConfigError(
            f"{message}; {above} is written whole and takes no keys under it"
        )
    nearest = difflib.get_close_matches(
        str(path[-1]), [str(k) for k in keys or []], n=1
    )
    if nearest:
        message += f"; did you mean '{_dotted((*path[:-1], nearest[0]))}'?"
    parent = unwrap(parent)
    if (kinds := kinds_of(parent)) is not None:
        message += f"; the kinds of {above} are {', '.join(kinds)}"
    elif isinstance(parent, _KindPosition) and path[-1] == "kind":
        message += f"; the key {path[-2]} already names the kind"
    return ConfigError(message)


def _flatten(value: Any, path: tuple[Any, ...], source: str) -> list[Update]:
    """Leaf updates of a parsed document: a non-empty mapping recurses, anything
    else (a scalar, a whole list, an empty mapping) is a leaf. Each leaf is a
    copy, so a YAML anchor shared under two keys becomes two values (G4)."""
    if isinstance(value, dict) and value:
        return [
            u
            for key, item in value.items()
            for u in _flatten(item, (*path, key), source)
        ]
    return [Update(path, copy.deepcopy(value), source)]


def _resolve(
    root: type[ConfigSection], update: Update, problems: list[str]
) -> Update | None:
    """Walk one update's path through the schema. A bare kind name at a kinds
    field comes back as its `kind` update; a list value is checked item by item
    but stays whole. A `ConfigError` is recorded under the update's source and
    the update dropped, so that every problem of one load is reported together."""
    path, value, source = update.path, update.value, update.source
    under_kinds: list[KindEntered] = []
    try:
        parent, node = None, root
        for depth, key in enumerate(path):
            child, keys = _step(node, key)
            if child is None:
                raise _unknown_key(node, keys, path[: depth + 1])
            if isinstance(child, _KindPosition):
                kinds = tuple(kinds_of(unwrap(node)) or ())
                under_kinds.append(KindEntered(path[:depth], kinds, key))
            parent, node = node, child
        if node is _TAG:
            kinds = kinds_of(unwrap(parent)) or {}
            if not isinstance(value, str):
                raise ConfigError(
                    f"{_dotted(path)} must name a kind, one of {', '.join(kinds)}; "
                    f"got {_as_shown(value)}"
                )
            path, node = path[:-1], parent
        node = unwrap(node)
        if (kinds := kinds_of(node)) is not None:
            if isinstance(value, str):
                if value not in kinds:
                    raise _unknown_key(node, ["kind", *kinds], (*path, value))
                return Update((*path, "kind"), value, source, tuple(under_kinds), path)
            if value != {}:
                raise ConfigError(
                    f"{_dotted(path)} must be the name of a kind or a mapping under "
                    f"one, one of {', '.join(kinds)}; got {_as_shown(value)}"
                )
        elif isinstance(node, _KindPosition) or is_section(node):
            if not isinstance(value, dict):
                raise ConfigError(
                    f"{_dotted(path)} must be a mapping of its keys; "
                    f"got {_as_shown(value)}"
                )
        else:
            node = _narrowed(node)
            is_list = get_origin(node) in (list, tuple) or node in (list, tuple)
            if is_list and isinstance(value, list):
                for index, item in enumerate(value):
                    for item_update in _flatten(item, (*path, index), source):
                        _resolve(root, item_update, problems)
        return Update(path, value, source, tuple(under_kinds))
    except ConfigError as error:
        problems.append(f"{source}: {error}")
        return None


def _resolve_all(root: type[ConfigSection], updates: list[Update]) -> list[Update]:
    problems: list[str] = []
    resolved = [_resolve(root, update, problems) for update in updates]
    if problems:
        raise ConfigError("\n".join(problems))
    return [update for update in resolved if update is not None]


def _set_at_path(mapping: dict[Any, Any], path: tuple[Any, ...], value: Any) -> None:
    """Set `value` at `path`, creating mappings on the way; an empty mapping
    never clears a mapping already there, and is stored as a fresh one so that
    later writes under it leave the update's value alone."""
    if not path:
        return
    for key in path[:-1]:
        if not isinstance(mapping.get(key), dict):
            mapping[key] = {}
        mapping = mapping[key]
    if value != {} or not isinstance(mapping.get(path[-1]), dict):
        mapping[path[-1]] = {} if value == {} else value


def read_config_file(config_file: str | Path) -> dict[str, Any]:
    """Parse one file by its extension (`.toml`, `.yaml`, `.yml`, `.json`, in
    any case).

    A `ConfigError` names the file for an unknown extension, a file that cannot
    be read or parsed, and a top level that is not a table. An empty or
    comment-only file is `{}`. Non-string YAML keys (`1:`) stay as parsed;
    quote them where the schema wants strings.
    """
    path = Path(config_file)
    parser = _PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ConfigError(
            f"unknown extension '{path.suffix}' of config file {path}; use .toml, "
            f".yaml, .yml or .json"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"cannot read config file {path}: {error}") from error
    try:
        document = parser(text)
    except (ValueError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot parse config file {path}: {error}") from error
    if document is None:
        document = {}
    if not isinstance(document, dict):
        raise ConfigError(
            f"config file {path} must have a table of keys at the top level, not "
            f"{type(document).__name__}"
        )
    return document


def _parse_overrides(tokens: Iterable[str]) -> list[Update]:
    """`--a.b.c value` or `--a.b.c=value`, in order. A value that is `null` or
    starts with `[` or `{` is JSON (and flattened like a file); any other value
    stays a string for pydantic to coerce. Keys match exactly."""
    argv = list(tokens)
    updates = []
    position = 0
    while position < len(argv):
        token = argv[position]
        position += 1
        key, has_inline_value, value = token[2:].partition("=")
        if not token.startswith("--") or not key:
            raise ConfigError(
                f"unknown config option '{token}'; options are --key.path value"
            )
        if not has_inline_value:
            if position == len(argv):
                raise ConfigError(f"override {token} is missing its value")
            value = argv[position]
            position += 1
        source = token if has_inline_value else f"{token} {value}"
        parsed: Any = value
        if value == "null" or value[:1] in ("[", "{"):
            try:
                parsed = json.loads(value)
            except ValueError as error:
                raise ConfigError(
                    f"override {token} is not valid JSON: {error}"
                ) from error
        updates.extend(_flatten(parsed, tuple(key.split(".")), source))
    return updates


class ConfigSection(BaseModel):
    """A node of a config tree: unknown keys are errors, and the schema rules
    of `_schema_rules` are checked when a subclass is defined (or, behind a
    forward reference, on the first `load`).

    A kinds field takes the wire form `{kind: X, X: {...}, Y: {...}}`, `{X:
    {...}}`, `"X"` or `{}` and dumps as `{X: {fields}}`; internally pydantic
    sees the tagged form `{kind: X, ...X's fields}`, which `model_validate` and
    direct construction accept as well. An instance of a subclass of a variant
    is held as given and exported under the variant's kind.
    """

    # inf/nan are exported as floats (JSON constants Infinity, NaN) rather than
    # pydantic's default null, which would store a different value in the model
    # metadata; `metadata._Record` writes them the same way.
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="constants")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        check_model_config(cls)
        if cls.__pydantic_complete__:
            check_fields(cls)

    @model_validator(mode="before")
    @classmethod
    def _kinds_fields_to_tagged_form(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for name, kinds in kinds_fields_of(cls).items():
            if name in data:
                data = {**data, name: _to_tagged_form(cls, name, data[name], kinds)}
        return data

    # No return annotation: with one, the serialization JSON schema collapses.
    @model_serializer(mode="wrap")
    def _kinds_fields_under_their_name(self, handler: SerializerFunctionWrapHandler):
        dumped = handler(self)
        for name, kinds in kinds_fields_of(type(self)).items():
            if name in dumped:  # absent under exclude_unset
                variant = getattr(self, name)
                if not isinstance(variant, tuple(kinds.values())):
                    raise TypeError(
                        f"{type(self).__name__}.{name} holds {type(variant).__name__}, "
                        f"which is not one of its variants"
                    )
                settings = {k: v for k, v in dumped[name].items() if k != "kind"}
                dumped[name] = {variant.kind: settings}
        return dumped


def _selected_kind(value: dict[str, Any], kinds: Collection[str]) -> str | None:
    """The kind a kinds field's wire mapping runs: its `kind` scalar, else its
    sole kind key; None when nothing selects one (an empty mapping runs the
    schema default, several kind keys need a selection)."""
    selected = value.get("kind")
    if isinstance(selected, str):
        return selected
    written = [key for key in value if key in kinds]
    return written[0] if len(written) == 1 else None


def _kinds_error(
    field: str, kinds: list[str], under: str | None = None
) -> PydanticCustomError:
    """The kinds error: `field` needs a kind (one of `kinds`), or the tag was
    written under the kind `under`. `load` renders it with the path, the fix
    and the sources (`_kind_error_message`)."""
    choices = " or ".join(f"kind: {kind}" for kind in kinds)
    template = (
        "{field} needs a kind; write {choices}"
        if under is None
        else "{field}.{under}.kind is not a key; {under} already names the kind"
    )
    context = {"field": field, "kinds": kinds, "under": under, "choices": choices}
    return PydanticCustomError(_KINDS_ERROR, template, context)


def _to_tagged_form(
    cls: type[ConfigSection], name: str, value: Any, kinds: dict[str, type[Any]]
) -> Any:
    """Wire form of one kinds field to pydantic's tagged form. Shapes the
    schema cannot take are returned unchanged for pydantic to report; the
    shapes it would misreport raise the kinds error, among them a `kind` inside
    the selected kind's settings, which is not a key there."""
    if isinstance(value, str):
        return {"kind": value}
    if not isinstance(value, dict) or not isinstance(value.get("kind", ""), str):
        return value
    selected = _selected_kind(value, kinds)
    if selected is None:
        written = [key for key in value if key in kinds]
        if written:
            raise _kinds_error(name, written)
        if value:
            return value
        default = cls.model_fields[name].get_default(call_default_factory=True)
        if default is PydanticUndefined:
            raise _kinds_error(name, list(kinds))
        if type(default) not in kinds.values():
            example = next(iter(kinds.values())).__name__
            raise TypeError(
                f"{cls.__name__}.{name} has a default factory whose result is not "
                f"one of its variants; return e.g. {example}()"
            )
        return {"kind": default.kind, **default.model_dump(exclude_unset=True)}
    settings = value.get(selected, {})
    if not isinstance(settings, dict):
        return value
    if "kind" in settings:
        raise _kinds_error(name, list(kinds), under=selected)
    rest = {k: v for k, v in value.items() if k not in kinds and k != "kind"}
    return {**rest, **settings, "kind": selected}


_ConfigT = TypeVar("_ConfigT", bound="ReforgeBaseConfig")


class ReforgeBaseConfig(ConfigSection):
    """The root of a config tree: `load` builds it from files and the command
    line; the two exports write it back as JSON-native dicts."""

    @classmethod
    def load(
        cls,
        config_files: str | Path | Iterable[str | Path] | None = (),
        cli_overrides: Iterable[str] = (),
    ) -> Self:
        """Build the config from `config_files` in order, then `cli_overrides`
        in order (`sys.argv[1:]`-style tokens); a lone path is one file, None
        is no file.

        Raises `ConfigError` for a file or override the schema cannot take
        (every unknown key of the load together, each under its file or
        override), pydantic's `ValidationError` for a value of the wrong type,
        and warns `ConfigWarning` for each override that had no effect on the
        config that runs.
        """
        if isinstance(cli_overrides, str):
            raise TypeError(
                "cli_overrides is a string; pass the tokens as a list, "
                "like sys.argv[1:]"
            )
        for section in reachable_sections(cls):
            check_fields(section)
        if config_files is None:
            config_files = ()
        files = (
            [config_files]
            if isinstance(config_files, (str, Path))
            else list(config_files)
        )
        from_files = [
            update
            for file in files
            for update in _flatten(read_config_file(file), (), str(Path(file)))
        ]
        updates = _resolve_all(cls, [*from_files, *_parse_overrides(cli_overrides)])
        merged: dict[str, Any] = {}
        for update in updates:
            _set_at_path(merged, update.path, update.value)
        config = _validated(cls, merged, updates)
        overrides = updates[len(from_files) :]
        messages = [
            _override_without_effect(update, overrides[position + 1 :], merged)
            for position, update in enumerate(overrides)
        ]
        for message in dict.fromkeys(m for m in messages if m is not None):
            warnings.warn(message, ConfigWarning, stacklevel=2)
        return config

    def to_resolved_dict(self) -> dict[str, Any]:
        """Every field with defaults filled, JSON-native, in declaration order;
        loading it back gives an equal config and the same dict. A kinds field
        holding a subclass of its variant is written as the variant, so a
        field the subclass added is not written."""
        return self.model_dump(mode="json")

    def to_user_dict(self) -> dict[str, Any]:
        """Only what the files and the overrides set, in the same shape; the
        kind that ran is recorded even when it was chosen by default."""
        return self.model_dump(mode="json", exclude_unset=True)


def _validated(
    cls: type[_ConfigT], merged: dict[str, Any], updates: list[Update]
) -> _ConfigT:
    """Validate once; a kinds error comes out as one `ConfigError`, and a load
    without a kinds error passes pydantic's `ValidationError` through (D3). A
    kinds error ends the validation of its class, so the other errors of that
    class are reported on the next load."""
    try:
        return cls.model_validate(merged)
    except ValidationError as error:
        kind_errors = [e for e in error.errors() if e["type"] == _KINDS_ERROR]
        if not kind_errors:
            raise
        lines = [_kind_error_message(e, updates) for e in kind_errors]
        raise ConfigError("\n".join(lines)) from error


def _wrote(update: Update, target: tuple[Any, ...]) -> bool:
    """Whether the update wrote at or under `target`, or a whole list holding it."""
    depth = len(update.path)
    if depth >= len(target):
        return update.path[: len(target)] == target
    return update.path == target[:depth] and isinstance(update.value, list)


def _kind_error_message(error: ErrorDetails, updates: list[Update]) -> str:
    """Pydantic's location is the section that raised, and the field is in the
    context. The dotted fix is left out inside a list item, where the walk
    would reject it; each kind that was written is named with the source that
    first wrote it. The tag under a kind is rejected by the walk before
    validation; should it arrive, pydantic's own sentence is kept."""
    ctx = error.get("ctx") or {}
    if ctx.get("under") is not None:
        return error["msg"]
    path = (*error["loc"], ctx["field"])
    kinds = ctx["kinds"]
    message = f"{_dotted(path)} needs a kind; write {ctx['choices']} in a file"
    if not any(isinstance(part, int) for part in path):
        message += f", or pass --{_dotted(path)} {kinds[0]}"
    writers = []
    for kind in kinds:
        writer = next((u for u in updates if _wrote(u, (*path, kind))), None)
        if writer is not None:
            writers.append(f"{kind} from {writer.source}")
    if writers:
        message += f"; {', '.join(writers)}"
    return message


def _override_without_effect(
    update: Update, later: list[Update], merged: dict[str, Any]
) -> str | None:
    """C11, the two ways an override loses its effect: a later override writes
    at its path, above it, or below a value of its that was not a mapping (`{}`
    never clears a mapping); or it wrote under a kind that does not run at its
    kinds field, decided from the merged dict as `_to_tagged_form` decides it.
    Never raises."""
    for other in reversed(later):
        depth = min(len(update.path), len(other.path))
        if other.value == {} or update.path[:depth] != other.path[:depth]:
            continue
        if len(other.path) > len(update.path) and update.value == {}:
            continue
        if other.selects is not None:
            field, running = _dotted(other.selects), _as_typed(other.value)
            return f"{update.source} is overridden: {field} runs {running}"
        at, value = _dotted(other.path), _as_typed(other.value)
        return f"{update.source} is overridden: {at} is {value}"
    for field, kinds, key in update.under_kinds:
        mapping: Any = merged
        for step in field:
            mapping = mapping.get(step) if isinstance(mapping, dict) else None
        running = _selected_kind(mapping, kinds) if isinstance(mapping, dict) else None
        if running is not None and running != key:
            at = _dotted(field)
            return f"{update.source}: {at}.{key} has no effect, {at} runs {running}"
    return None
