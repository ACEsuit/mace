"""The schema rules a config class must keep, checked when the class is defined,
and the annotation introspection the loader shares with them.

A *section* is a `ConfigSection` subclass; a *kinds field* holds a union of
two or more sections, declared `Annotated[A | B, Field(discriminator="kind")]`
with `kind: Literal["a"]` in each variant. Every rule raises `TypeError`
`<Class>.<field> <reason>` naming the fix.

`base` is bound as a module and dereferenced at call time only: `base.py`
imports this module at its top, so a `from mace_core.config.base import ...`
here would break the package import whichever module is imported first.
"""

from __future__ import annotations

import types
from typing import Annotated, Any, Literal, TypeGuard, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from mace_core.config import base as _base

_NO_KIND_FIX = (
    "default to a variant, or for none of the kinds declare an empty variant "
    "with kind: Literal['none']"
)


def is_section(node: Any) -> TypeGuard[type[_base.ConfigSection]]:
    """A section class. A parametrised generic such as `list[int]` passes
    `isinstance(node, type)` on 3.10, hence the origin check first."""
    return (
        get_origin(node) is None
        and isinstance(node, type)
        and issubclass(node, _base.ConfigSection)
    )


def unwrap(node: Any) -> Any:
    """`Annotated[X, ...]` as `X`; anything else unchanged."""
    return get_args(node)[0] if get_origin(node) is Annotated else node


def members_of(node: Any) -> tuple[Any, ...]:
    """The members of a union, nested unions and `Annotated` flattened; a
    non-union is its own single member."""
    node = unwrap(node)
    if get_origin(node) in (Union, types.UnionType):
        return tuple(member for arg in get_args(node) for member in members_of(arg))
    return (node,)


def tag_of(variant: type[BaseModel]) -> str | None:
    """The one string of a variant's `kind: Literal[...]`; None for any other
    shape of tag."""
    field = variant.model_fields.get("kind")
    if field is None or get_origin(field.annotation) is not Literal:
        return None
    values = get_args(field.annotation)
    return values[0] if len(values) == 1 and isinstance(values[0], str) else None


def kinds_of(node: Any) -> dict[str, type[_base.ConfigSection]] | None:
    """`{tag: variant}` for a union of two or more sections, in declaration
    order; None for anything else. A variant without a proper tag is keyed by
    its class name so that `check_kinds_field` can still name it."""
    members = members_of(node)
    if len(members) < 2 or not all(is_section(member) for member in members):
        return None
    return {tag_of(member) or member.__name__: member for member in members}


def kinds_fields_of(cls: type[BaseModel]) -> dict[str, dict[str, type[Any]]]:
    """The kinds fields of a class, each with its `{tag: variant}`."""
    fields = {}
    for name, field in cls.model_fields.items():
        kinds = kinds_of(field.annotation)
        if kinds is not None:
            fields[name] = kinds
    return fields


def reachable_sections(cls: type[BaseModel]) -> list[type[BaseModel]]:
    """Every section reachable from `cls` through its fields, `cls` included.
    A class left incomplete by a forward reference is rebuilt on the way, which
    raises if the name never resolves (F7)."""
    found: list[type[BaseModel]] = []
    pending = [cls]
    while pending:
        section = pending.pop()
        if section in found:
            continue
        if not section.__pydantic_complete__:
            section.model_rebuild()
        found.append(section)
        for field in section.model_fields.values():
            pending.extend(sections_in(field.annotation))
    return found


def sections_in(annotation: Any) -> list[type[BaseModel]]:
    """The sections inside an annotation: union members and the parameters
    of lists, dicts and tuples, at any depth."""
    node = unwrap(annotation)
    if is_section(node):
        return [node]
    return [section for arg in get_args(node) for section in sections_in(arg)]


def admits_none(annotation: Any) -> bool:
    return any(m in (Any, object, type(None)) for m in members_of(annotation))


def check_model_config(cls: type[BaseModel]) -> None:
    """G5: `extra="forbid"` cannot be reopened by a subclass."""
    extra = cls.model_config.get("extra")
    if extra != "forbid":
        raise TypeError(
            f"{cls.__name__} sets extra={extra!r}; a section keeps extra='forbid' "
            f"so that an unknown key is an error"
        )


def check_fields(cls: type[BaseModel]) -> None:
    """F1-F6 over every field of a complete class."""
    for name in cls.model_computed_fields:
        raise TypeError(
            f"{cls.__name__}.{name} is a computed field; the resolved export "
            f"could not be loaded back"
        )
    for name, field in cls.model_fields.items():
        where = f"{cls.__name__}.{name}"
        if field.alias or field.validation_alias or field.serialization_alias:
            raise TypeError(
                f"{where} has an alias; the resolved export could not be loaded back"
            )
        if field.exclude:
            raise TypeError(
                f"{where} is excluded from dumps; the resolved export could not be "
                f"loaded back"
            )
        check_annotation(where, field.annotation, field.discriminator is not None)
        kinds = kinds_of(field.annotation)
        if kinds is not None:
            check_kinds_field(where, kinds, field.default)
        elif field.default is None and not admits_none(field.annotation):
            raise TypeError(f"{where} defaults to None, which its type does not admit")


def _is_lenient_model(node: Any) -> bool:
    return (
        get_origin(node) is None
        and isinstance(node, type)
        and issubclass(node, BaseModel)
        and not is_section(node)
    )


def check_annotation(
    where: str, annotation: Any, discriminated: bool, inside: bool = False
) -> None:
    """F1 (no sets), F4 (sections only) and F5 (union shapes) at every depth;
    `inside` is true under a dict, list or tuple, where a kinds union is not
    allowed."""
    node = unwrap(annotation)
    if get_origin(node) is Literal:
        return
    if node in (set, frozenset) or get_origin(node) in (set, frozenset):
        raise TypeError(
            f"{where} is typed as a set, whose order changes between runs. Use a list"
        )
    members = members_of(node)
    for member in members:
        if _is_lenient_model(member):
            raise TypeError(
                f"{where} holds {member.__name__}, which is not a ConfigSection"
            )
    sections = [member for member in members if is_section(member)]
    if sections:
        if type(None) in members:
            raise TypeError(f"{where} admits None, which is not a kind; {_NO_KIND_FIX}")
        others = [member for member in members if not is_section(member)]
        if others:
            other = get_origin(others[0]) or others[0]
            raise TypeError(
                f"{where} mixes its kinds with {other.__name__}; a kinds field holds "
                f"its variants only"
            )
        if len(sections) > 1 and inside:
            raise TypeError(
                f"{where} is a union of sections inside a dict, list or tuple; "
                f"declare a section holding the kinds field there"
            )
        if len(sections) > 1 and not discriminated:
            raise TypeError(
                f"{where} is a union of sections without a discriminator; declare "
                f"kinds as Annotated[A | B, Field(discriminator='kind')]"
            )
        return
    if len(members) > 1:
        for member in members:
            check_annotation(where, member, False, inside)
    else:
        for arg in get_args(node):
            check_annotation(where, arg, False, True)


def check_kinds_field(where: str, kinds: dict[str, type[Any]], default: Any) -> None:
    """F6: one-string tags, no variant named like the tag, no variant field
    named like a kind (the flat form could not tell it from a kind's settings),
    a variant as the default. A default factory is neither run nor checked here
    (F3); its result is checked when an empty mapping selects it."""
    for tag, variant in kinds.items():
        if tag_of(variant) is None:
            raise TypeError(
                f"{where} has variant {variant.__name__} whose kind must be a "
                f"Literal of exactly one string"
            )
        if tag == "kind":
            raise TypeError(
                f"{where} has variant {variant.__name__} whose kind is named 'kind' "
                f"like the tag; rename it"
            )
        for name in variant.model_fields:
            if name != "kind" and name in kinds:
                raise TypeError(
                    f"{where} has variant {variant.__name__} with a field named "
                    f"like the kind {name}; rename the field"
                )
    if default is None:
        raise TypeError(
            f"{where} defaults to None, which is not a kind; {_NO_KIND_FIX}"
        )
    if default is not PydanticUndefined and type(default) not in kinds.values():
        example = next(iter(kinds.values())).__name__
        raise TypeError(
            f"{where} has a default that is not one of its variants; write e.g. "
            f"{example}()"
        )
