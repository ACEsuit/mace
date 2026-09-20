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

A field may hold a section of one of several *kinds*: a discriminated union
whose tag field names the kind.

    class HuberLoss(ConfigSection):
        kind: Literal["huber"] = "huber"
        delta: float = 0.01

    Loss = Annotated[WeightedLoss | HuberLoss, Field(discriminator="kind")]

    class TrainConfig(ReforgeBaseConfig):
        loss: Loss = WeightedLoss()

Code sees the union: `config.loss` is a `WeightedLoss` or a `HuberLoss`. A
file and the command line never write the tag; they write the kind as the
key the section sits under, or as a bare name, which is that kind with
nothing set under it:

    loss: huber                     --loss huber
    loss: {huber: {delta: 0.1}}     --loss.huber.delta 0.1

The file and each override merge in order. A bare name on top of a section
of the same kind keeps that section's keys; a different kind replaces it:
`--loss weighted` on top of a file with a huber section runs the weighted
loss and warns (`ConfigWarning`) that the huber section is ignored. Two
kinds written in one place, one file or one override, is an error, and so
is `null` at a kinds field or under a kind: a kinds field always holds a
kind. When none of the kinds is a valid choice, that is a kind too, an
empty variant such as `class NoLoss(ConfigSection): kind: Literal["none"]`,
written `loss: none`. A kinds field with no kind written takes the kind of
its default. The resolved and user dicts are written the same way, kind as
key.

The tagged dict, `{kind: huber, delta: 0.1}`, is pydantic's internal form:
what `model_validate` takes and what `model_json_schema` describes. It is
not a file format; `load()` refuses it. Code builds sections as instances
(`HuberLoss(delta=0.1)`) and never meets either dict form.

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
import warnings
from collections.abc import Callable, Iterator, Sequence
from itertools import cycle
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, get_args, get_origin

import tomli
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
    model_validator,
)
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = [
    "ConfigError",
    "ConfigSection",
    "ConfigWarning",
    "ReforgeBaseConfig",
    "read_config_file",
]

#: Config file extensions this module reads, keyed to their parsers.
_FILE_PARSERS = {
    ".toml": tomli.loads,
    ".yaml": yaml.safe_load,
    ".yml": yaml.safe_load,
    ".json": json.loads,
}

#: A dotted path as its parts; a list index is a part too.
_Path = tuple[str, ...]

#: Where a value came from: the layer's index and its name ("the config file"
#: or the override as typed). Comparing two origins compares their order.
_Origin = tuple[int, str]

#: What runs at a kinds field during a walk: gets the field's value (any
#: shape), the field and its path, returns the value to go on with.
_KindsAction = Callable[[Any, FieldInfo, _Path], Any]


# ---------------------------------------------------------------------------
# Schema introspection


def _sections_in(
    annotation: Any, inside: bool = False
) -> Iterator[tuple[type[BaseModel], bool]]:
    """Every section class a field annotation can hold (`Radial`, `Radial | None`,
    `list[Radial]`, `Annotated[A | B, ...]`), with whether it sits inside a
    dict/list/tuple, where an error location has a key or index before the
    section's own field names."""
    origin = get_origin(annotation)  # `list` for `list[X]`; None for a plain class
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield annotation, inside
    elif origin is Annotated:
        yield from _sections_in(get_args(annotation)[0], inside)
    elif origin in (Union, UnionType):
        for arg in get_args(annotation):
            yield from _sections_in(arg, inside)
    elif origin in (dict, list, tuple):
        for arg in get_args(annotation):
            yield from _sections_in(arg, True)


def _arms(annotation: Any) -> Iterator[Any]:
    """The alternatives of a union, through `Annotated`; else the annotation."""
    origin = get_origin(annotation)
    if origin is Annotated:
        yield from _arms(get_args(annotation)[0])
    elif origin in (Union, UnionType):
        for arg in get_args(annotation):
            yield from _arms(arg)
    else:
        yield annotation


def _section_of(annotation: Any) -> tuple[type[BaseModel] | None, bool]:
    """The one section a field can hold, and whether it is inside a collection."""
    found = dict(_sections_in(annotation))
    return next(iter(found.items())) if found else (None, False)


def _tag_of(field: FieldInfo) -> str | None:
    """The tag field name of a kinds field, else None. Also found through an
    outer union, `Annotated[...] | None`, which keeps it in the `Annotated`
    metadata, so that `_check_schema` can reject that spelling by name."""
    found: Any = field.discriminator
    if found is None and get_origin(field.annotation) in (Union, UnionType):
        for arg in get_args(field.annotation):
            if get_origin(arg) is Annotated:
                for meta in get_args(arg)[1:]:
                    if isinstance(meta, FieldInfo) and meta.discriminator is not None:
                        found = meta.discriminator
    return found if isinstance(found, str) else None


def _tag_values(section: type[BaseModel], tag: str) -> tuple[Any, ...]:
    """The `Literal` values of a variant's tag field; empty if not a Literal."""
    tag_field = section.model_fields.get(tag)
    if tag_field is None or get_origin(tag_field.annotation) is not Literal:
        return ()
    return get_args(tag_field.annotation)


def _kinds_of(field: FieldInfo) -> dict[str, type[BaseModel]]:
    """Kind name -> section class of a kinds field, in declaration order."""
    tag = _tag_of(field)
    if tag is None:
        return {}
    return {
        _tag_values(section, tag)[0]: section
        for section, _ in _sections_in(field.annotation)
    }


def _default_kind(field: FieldInfo) -> str | None:
    """The kind of the field's default section; None for a required field."""
    tag = _tag_of(field)
    default = field.get_default(call_default_factory=True)
    if tag is None or not isinstance(default, BaseModel):
        return None
    return getattr(default, tag)


def _kinds_text(field: FieldInfo) -> str:
    return ", ".join(_kinds_of(field))


def _admits_none(annotation: Any) -> bool:
    """`X | None`, `Optional[X]`, `Any`, `object`, also under `Annotated`."""
    return any(arm in (type(None), Any, object) for arm in _arms(annotation))


def _contains_a_set(annotation: Any) -> bool:
    """A bare `set`, a `set[X]`, or a set anywhere inside, e.g. `list[set[int]]`."""
    if annotation in (set, frozenset) or get_origin(annotation) in (set, frozenset):
        return True
    return any(_contains_a_set(arg) for arg in get_args(annotation))


_NONE_IS_A_KIND = (
    "default to a variant, or for none of the kinds add an empty variant, "
    'kind: Literal["none"], and default to that'
)


def _check_schema(model: type[BaseModel]) -> None:
    """Fail at class definition for a field shape the contract cannot keep.

    Each rule protects one guarantee: no sets (order is not stable across
    runs, so the export would not be a fixed point); no aliases or computed
    fields (the export would not validate back); no `None` default on a type
    that does not admit `None` (pydantic does not validate defaults, so the
    export would not validate back; a default factory is not run here, so
    what it returns is not checked); several sections under one field only
    as kinds, i.e. a discriminated union, and not inside a collection (a
    value must not become whichever alternative happens to accept it, and
    the CLI addresses a collection only as a whole); nothing beside the
    variants of a kinds field, not even `None` (a kinds field always holds a
    kind; "none of them" is an empty variant, so it can be written, named
    among the kinds and warned about like any other); a tag that is one
    string other than the tag's own name (it is the key the kind is written
    under); a default that is a variant written as an instance, not a
    factory (its kind is the default kind, read without running anything);
    every section is a `ConfigSection` (a plain `BaseModel` ignores unknown
    keys, so a typo would vanish, and skips these checks).
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
        sections = dict(_sections_in(field.annotation))
        tag = _tag_of(field)
        # `default` is undefined, not None, for a required field or a factory;
        # a factory is not run here, so what it returns is not checked
        if field.default is None and tag is not None:
            reject(name, f"defaults to None, which is not a kind; {_NONE_IS_A_KIND}")
        if field.default is None and not _admits_none(field.annotation):
            reject(
                name, "defaults to None but its type does not admit None; add | None"
            )
        if len(sections) > 1 and any(sections.values()):
            reject(
                name,
                "is a union of sections inside a dict, list or tuple; the CLI "
                "addresses such a field only as a whole. Put the union in a "
                "field of the section that is the element",
            )
        if len(sections) > 1 and tag is None:
            reject(
                name,
                "is a union of sections without a discriminator; spell it "
                'Annotated[A | B, Field(discriminator="kind")] with a '
                '`kind: Literal["a"]` field in each',
            )
        for section in sections:
            if not issubclass(section, ConfigSection):
                reject(
                    name,
                    f"holds {section.__name__}, which is not a ConfigSection; "
                    f"subclass it",
                )
        if tag is None:
            continue
        others = [arm for arm in _arms(field.annotation) if arm not in sections]
        if type(None) in others:
            reject(name, f"admits None, which is not a kind; {_NONE_IS_A_KIND}")
        if others:
            reject(
                name,
                f"mixes its kinds with {getattr(others[0], '__name__', others[0])}; "
                f"a kinds field holds its variants only",
            )
        for section in sections:
            values = _tag_values(section, tag)
            if len(values) != 1 or not isinstance(values[0], str):
                reject(
                    name,
                    f"has variant {section.__name__} whose {tag} must be a Literal "
                    f"of exactly one string; a config names the kind by it",
                )
            if values[0] == tag:
                reject(
                    name,
                    f"has variant {section.__name__} whose kind is named {tag!r} like "
                    f"the tag; a config could not tell the two apart. Rename it",
                )
        example = f"{next(iter(sections)).__name__}()"
        if field.default_factory is not None:
            reject(
                name,
                f"has a default_factory; write the default as an instance, e.g. "
                f"{example} (pydantic copies it per instance)",
            )
        if not field.is_required() and not isinstance(field.default, tuple(sections)):
            reject(
                name,
                f"has a default that is not one of its variants; write e.g. {example}",
            )


class ConfigError(ValueError):
    """A config file or override the schema rejects.

    Raised for a missing, unparsable or malformed file, an unknown key or
    kind, an override the CLI parser cannot make sense of, two kinds of one
    section written in one place, and `null` at or under a kind. The
    message names the file or the
    offending key by its dotted path, and suggests the nearest valid key
    when there is a close match.
    """


class ConfigWarning(UserWarning):
    """A section of one kind is ignored because a later layer selected another
    kind. `warnings.simplefilter("error", ConfigWarning)` makes it an error."""


class ConfigSection(BaseModel):
    """A nested section of a configuration: a table in the file, a dotted
    prefix on the command line. Unknown keys are errors here too."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        _check_schema(cls)

    @model_validator(mode="before")
    @classmethod
    def _kinds_from_keys(cls, values: Any) -> Any:
        """`{huber: {delta: 0.1}}` under a kinds field becomes the
        `{kind: huber, delta: 0.1}` pydantic's discriminator reads."""
        if not isinstance(values, dict):
            return values
        values = dict(values)
        for name, field in cls.model_fields.items():
            tag, value = _tag_of(field), values.get(name)
            if tag is not None and isinstance(value, dict) and len(value) == 1:
                ((kind, inner),) = value.items()
                if isinstance(inner, dict) and tag not in value:
                    values[name] = {**inner, tag: kind}
        return values

    @model_serializer(mode="wrap")
    def _kinds_as_keys(self, handler: SerializerFunctionWrapHandler):
        """The dump with every kinds field written kind-as-key. The kind is read
        from the instance: an unset default tag is absent from a user dict.
        No return annotation: pydantic would take it as the JSON schema."""
        dumped = handler(self)
        for name, field in type(self).model_fields.items():
            tag = _tag_of(field)
            if tag is not None and isinstance(dumped.get(name), dict):
                inner = {key: v for key, v in dumped[name].items() if key != tag}
                dumped[name] = {getattr(getattr(self, name), tag): inner}
        return dumped


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
        is replaced whole. Under a kinds field the kind written last wins
        and the others are dropped with a `ConfigWarning`.

        Raises `ConfigError` for an unknown key or kind, an unreadable file
        or an unparsable override, and pydantic's `ValidationError` for a
        value of the wrong type.
        """
        values: dict[str, Any] = {}
        if config_file is not None:
            values = read_config_file(config_file)
        layers = [(values, "the config file"), *_parse_overrides(cls, cli_overrides)]
        merged: dict[str, Any] = {}
        origins: dict[_Path, _Origin] = {}
        for index, (layer, source) in enumerate(layers):
            layer = _at_kinds(layer, cls, _name_as_mapping)
            merged = _deep_update(merged, layer)
            _record_origins(origins, layer, (index, source))
        merged = _at_kinds(merged, cls, _KindSelector(origins))
        try:
            return cls.model_validate(merged)
        except ValidationError as error:
            unknown = _unknown_key_messages(cls, error)
            if not unknown:
                raise
            raise ConfigError("\n".join(unknown)) from error

    def to_resolved_dict(self) -> dict[str, Any]:
        """Every field, defaults included, as JSON-native values, in schema
        order, a kinds field as `{kind: {...}}`. Loading the result back and
        resolving again gives the same dict. (TOML has no null, so a `None`
        can only go out as YAML or JSON.)"""
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


# ---------------------------------------------------------------------------
# Merging


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """`base` overlaid with `update`, recursing where both hold a dict."""
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _record_origins(
    origins: dict[_Path, _Origin], values: Any, origin: _Origin, path: _Path = ()
) -> None:
    """Note `origin` for every path a layer writes, sections included."""
    items: Any = ()
    if isinstance(values, dict):
        items = values.items()
    elif isinstance(values, list):
        items = enumerate(values)
    for key, value in items:
        origins[(*path, str(key))] = origin
        _record_origins(origins, value, origin, (*path, str(key)))


def _parse_overrides(
    model: type[BaseModel], cli_overrides: Sequence[str]
) -> list[tuple[dict[str, Any], str]]:
    """Each `--a.b value` or `--a.b=value` pair as a mapping nested under its
    path, with the override as typed, in order."""
    valid = list(_dotted_paths(model))
    valid_set = set(valid)
    unknown: list[str] = []
    tokens = iter(cli_overrides)
    layers = []

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
        source = token
        if value is None:
            value = next(tokens, None)
            if value is None:
                raise ConfigError(f"override --{name} is missing its value")
            source = f"{token} {value}"

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

        override: Any = value
        for section in reversed(name.split(".")):
            override = {section: override}
        layers.append((override, source))

    if unknown:
        raise ConfigError("\n".join(unknown))

    return layers


def _dotted_paths(model: type[BaseModel], prefix: str = "") -> Iterator[str]:
    """Every field of the tree as a dotted path, sections included. A kinds
    field lists each kind as a key with the kind's fields under it, without
    the tag field: the key names the kind.

    A section inside a dict or list is not descended into: the CLI addresses
    such a field only as a whole, with a JSON value.
    """
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        yield path
        kinds = _kinds_of(field)
        for kind, variant in kinds.items():
            yield f"{path}.{kind}"
            for sub_path in _dotted_paths(variant, f"{path}.{kind}."):
                if sub_path != f"{path}.{kind}.{_tag_of(field)}":
                    yield sub_path
        section, inside_collection = _section_of(field.annotation)
        if section is not None and not inside_collection and not kinds:
            yield from _dotted_paths(section, f"{path}.")


# ---------------------------------------------------------------------------
# Kinds: a walk over the values against the schema, with an action at every
# kinds field. The action sees and returns the kind-as-key form.


def _at_kinds(
    values: Any, section: type[BaseModel] | None, action: _KindsAction, path: _Path = ()
) -> Any:
    """`values`, a mapping for `section`, with `action` applied at each kinds
    field and every section under it walked in turn."""
    if section is None or not isinstance(values, dict):
        return values
    out = dict(values)
    for key, value in values.items():
        field = section.model_fields.get(key)
        if field is None:
            continue
        kinds = _kinds_of(field)
        if not kinds:
            out[key] = _under(value, field.annotation, action, (*path, key))
            continue
        value = action(value, field, (*path, key))
        if isinstance(value, dict):
            value = {
                kind: _at_kinds(inner, kinds.get(kind), action, (*path, key, kind))
                for kind, inner in value.items()
            }
        out[key] = value
    return out


def _under(value: Any, annotation: Any, action: _KindsAction, path: _Path) -> Any:
    """`value` with `_at_kinds` applied to every section the annotation reaches
    through unions, dicts, lists and tuples. A value whose shape the annotation
    does not describe is returned as is, for pydantic to report."""
    origin = get_origin(annotation)
    if origin is None:
        return _at_kinds(value, _section_of(annotation)[0], action, path)
    if origin is Annotated:
        return _under(value, get_args(annotation)[0], action, path)
    if origin in (Union, UnionType):  # at most one arm takes a dict or a list
        for arm in get_args(annotation):
            value = _under(value, arm, action, path)
        return value
    if origin is dict and isinstance(value, dict):
        value_type = get_args(annotation)[1]
        return {
            k: _under(v, value_type, action, (*path, str(k))) for k, v in value.items()
        }
    if origin in (list, tuple) and isinstance(value, list):
        item_types = [a for a in get_args(annotation) if a is not Ellipsis]
        return [
            _under(v, t, action, (*path, str(i)))
            for i, (v, t) in enumerate(zip(value, cycle(item_types), strict=False))
        ]
    return value


def _shown(value: Any) -> str:
    """A value as the user could have written it; a date or a YAML set as text."""
    return json.dumps(value, default=str)


def _name_as_mapping(value: Any, field: FieldInfo, path: _Path) -> Any:
    """A bare kind name is that kind with its defaults, `{huber: {}}`, so that
    it merges into an earlier section of the same kind instead of replacing it."""
    return {value: {}} if isinstance(value, str) else value


class _KindSelector:
    """At a kinds field after the merge: keep the kind written last, drop the
    others with a warning, fill in the default kind, and check the shape."""

    def __init__(self, origins: dict[_Path, _Origin]) -> None:
        self.origins = origins

    def __call__(self, value: Any, field: FieldInfo, path: _Path) -> Any:
        dotted = ".".join(path)
        tag, kinds, named = _tag_of(field), _kinds_of(field), _kinds_text(field)
        if value is None:
            raise ConfigError(
                f"{dotted} does not take null; write a kind, one of {named}"
            )
        if not isinstance(value, dict):
            raise ConfigError(
                f"{dotted} must be the name of a kind or a mapping under one, "
                f"one of {named}; got {_shown(value)}"
            )
        if tag in value:
            raise ConfigError(
                f"{dotted}.{tag} is not a key; write the kind as the key the "
                f"section sits under, {dotted}: {{{_shown(value[tag])}: {{...}}}}"
            )
        nulled = [k for k, inner in value.items() if inner is None and k in kinds]
        if nulled:
            raise ConfigError(
                f"{dotted}.{nulled[0]} does not take null; set the keys wanted "
                f"under it, or write another kind"
            )
        if not value:
            default = _default_kind(field)
            if default is None:
                raise ConfigError(f"{dotted} needs a kind; one of {named}")
            return {default: {}}
        present = value
        origin_of = {k: self.origins[(*path, str(k))] for k in present}
        by_origin = sorted(present, key=origin_of.__getitem__)
        kind = by_origin[-1]
        index, source = origin_of[kind]
        tied = [k for k in by_origin if origin_of[k][0] == index]
        if len(tied) > 1:
            raise ConfigError(
                f"{dotted} is given as several kinds ({', '.join(tied)}) in "
                f"{source}; keep one"
            )
        if kind not in kinds:
            valid = [f"{dotted}.{k}" for k in kinds]
            raise ConfigError(
                _unknown_key_message(f"{dotted}.{kind}", valid)
                + f"; the kinds of {dotted} are {named}"
            )
        for loser in by_origin[:-1]:
            warnings.warn(
                f"{dotted}.{loser} from {origin_of[loser][1]} is "
                f"ignored: {source} selects {dotted}.{kind}",
                ConfigWarning,
                stacklevel=2,
            )
        inner = present[kind]
        if not isinstance(inner, dict):
            raise ConfigError(
                f"{dotted}.{kind} must be a mapping of the kind's keys; "
                f"got {_shown(inner)}"
            )
        if tag in inner:
            raise ConfigError(
                f"{dotted}.{kind}.{tag} is not a key; the kind is given by the "
                f"key {kind!r}"
            )
        return {kind: inner}


# ---------------------------------------------------------------------------
# Error messages


def _unknown_key_message(key: str, candidates: Sequence[str]) -> str:
    message = f"unknown config key {key!r}"
    closest = difflib.get_close_matches(key, candidates, n=1)
    if closest:
        message += f"; did you mean {closest[0]!r}?"
    return message


def _locate(
    model: type[BaseModel], location: Sequence[Any]
) -> tuple[list[str], list[str]]:
    """Pydantic's error location as the names of a dotted path, and the keys
    valid where it ends. A field name moves into its section; a kind moves
    into its variant, whose tag field is not offered since the key names the
    kind; a dict key or list index stays in the section; the class name
    pydantic inserts under a `Section | scalar` field is dropped."""
    section: type[BaseModel] | None = model
    kinds: dict[str, type[BaseModel]] | None = None
    hidden: str | None = None
    inside_collection = False
    names = []
    for part in map(str, location):
        if kinds is not None:
            names.append(part)
            section, kinds = kinds.get(part), None
            continue
        if inside_collection:
            names.append(part)
            inside_collection = False
            continue
        if section is not None and part == section.__name__:
            continue
        names.append(part)
        if section is not None and part in section.model_fields:
            field = section.model_fields[part]
            hidden = _tag_of(field)
            if hidden is not None:
                kinds = _kinds_of(field)
            else:
                section, inside_collection = _section_of(field.annotation)
        else:
            section = None
    candidates = list(section.model_fields) if section is not None else []
    return names, [c for c in candidates if c != hidden]


def _unknown_key_messages(model: type[BaseModel], error: ValidationError) -> list[str]:
    """Pydantic's unknown-key errors as messages that name the full dotted
    path and the closest valid key at that level."""
    messages = []
    for item in error.errors():
        # pydantic's error code for a key that matches no field (extra="forbid").
        # Every other code, e.g. a wrong type, is left for `load()` to re-raise.
        if item["type"] != "extra_forbidden":
            continue
        *location, key = item["loc"]
        names, candidates = _locate(model, location)
        prefix = "".join(f"{name}." for name in names)
        messages.append(
            _unknown_key_message(f"{prefix}{key}", [f"{prefix}{c}" for c in candidates])
        )
    return messages
