"""What a run batches, declared once: the :class:`GraphLayout`.

The graph is the structural core of :mod:`mace_core.graph.schema` plus what
the layout declares, and nothing else: the inputs and embedding features the
model reads, and the fields its batch requirements build. The targets beside
it are the observables the trainer declares, each with a label, a presence
flag and a loss weight. A model that reads no magnetic moment gets none in its
batch.

A field's level comes from ``per_atom`` and its trailing shape from the irreps,
so a property declared in config becomes a batched, padded field with no new
code. A default convention name (``forces``, ``stress``) keeps the shape its
row of :class:`~mace_core.elements.DefaultKeys` gives, and its irreps must fill
that shape: three components for a vector, nine for a 3x3 tensor, or ``0e+2e``
for a symmetric one.

The layout is resolved once per run and is plain data, so it can be recorded
with the resolved configuration.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from mace_core.config.base import ConfigSection
from mace_core.data.configuration import DEFAULT_HEAD
from mace_core.elements.default_keys import STORAGES, DefaultKeys, Storage
from mace_core.elements.number_table import AtomicNumberTable
from mace_core.graph.schema import (
    ANY_TRAILING,
    FLOAT,
    GRAPH_CORE,
    GRAPH_LEVEL,
    NODE_LEVEL,
    PRESENCE_SUFFIX,
    WEIGHT_SUFFIX,
    FieldSpec,
    presence_key,
    weight_key,
)
from mace_core.observables import InputSpec, ObservableSpec
from mace_core.observables.defaults import DEFAULT_CATALOGUE
from mace_core.observables.grammar import irreps_dimension, parse_irreps

__all__ = [
    "ENGINE_INPUTS",
    "FloatDtype",
    "GraphLayout",
    "feature_field",
    "graph_layout",
    "input_field",
    "target_field",
]

#: A run's floating dtype, by name.
FloatDtype = Literal["float16", "bfloat16", "float32", "float64"]

#: Inputs the derivative engine creates around the model call; never batched.
ENGINE_INPUTS: frozenset[str] = frozenset(
    spec.name for spec in DEFAULT_CATALOGUE.inputs
)

#: The structure-level loss weight every target set carries.
STRUCTURE_WEIGHT = "weight"

#: What a structure without a value for an input carries, where it is not 0.
#: A total spin is a multiplicity, and 1 is the closed shell.
_ABSENT_VALUES: dict[str, float] = {DefaultKeys.TOTAL_SPIN.convention_name: 1.0}

#: Weights with one entry per component of their label, as develop stores them.
_WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    member.convention_name: member.value_shape
    for member in (DefaultKeys.DIPOLE, DefaultKeys.POLARIZABILITY)
}

_CONVENTION_SHAPES: dict[str, tuple[int, ...]] = {
    member.convention_name: member.value_shape for member in DefaultKeys
}

#: The batched shape of each default convention name: its level from where
#: the data stores it, then the shape of one value.
_CONVENTION_FIELDS: dict[str, tuple[int | str, ...]] = {
    member.convention_name: (
        NODE_LEVEL if member.storage == "atom" else GRAPH_LEVEL,
        *member.value_shape,
    )
    for member in DefaultKeys
}


def _is_symmetric_rank_two(irreps: str, observable: str) -> bool:
    terms = parse_irreps(irreps, observable=observable)
    return sorted((t.degree, t.multiplicity, t.parity) for t in terms) == [
        (0, 1, "e"),
        (2, 1, "e"),
    ]


def _value_shape(name: str, irreps: str) -> tuple[int, ...]:
    dimension = irreps_dimension(irreps, observable=name)
    if name in _CONVENTION_SHAPES:
        shape = _CONVENTION_SHAPES[name]
        fits = dimension == math.prod(shape) or (
            shape == (3, 3) and _is_symmetric_rank_two(irreps, name)
        )
        if not fits:
            raise ValueError(
                f"{name!r} is declared with irreps {irreps!r}, which span "
                f"{dimension} components; its stored shape is {shape}. Declare "
                f"irreps that fill it."
            )
        return shape
    terms = parse_irreps(irreps, observable=name)
    scalar = len(terms) == 1 and terms[0].degree == 0 and terms[0].multiplicity == 1
    return () if scalar else (dimension,)


def _declared(per_atom: bool, value_shape: tuple[int, ...], absent: float) -> FieldSpec:
    level = NODE_LEVEL if per_atom else GRAPH_LEVEL
    return FieldSpec(shape=(level, *value_shape), dtype=FLOAT, absent_value=absent)


def input_field(spec: InputSpec) -> FieldSpec:
    """The batched field of a declared model input.

    Raises:
        ValueError: for ``pos`` or ``strain``, which the derivative engine
            creates rather than reads from a batch, or for irreps that do not
            fill a default name's shape.
    """
    if spec.name in ENGINE_INPUTS:
        raise ValueError(
            f"input {spec.name!r} is created by the derivative engine, not "
            f"batched. Leave it out of the layout."
        )
    shape = _value_shape(spec.name, spec.irreps)
    return _declared(spec.per_atom, shape, _ABSENT_VALUES.get(spec.name, 0.0))


def target_field(spec: ObservableSpec) -> FieldSpec:
    """The batched label of a declared observable.

    Raises:
        ValueError: for irreps that do not fill a default name's shape.
    """
    return _declared(spec.per_atom, _value_shape(spec.name, spec.irreps), 0.0)


def feature_field(storage: Storage) -> FieldSpec:
    """The batched field of an embedding feature, stored per ``atom`` or per
    ``graph``. Its trailing sizes are the data's own; it is floating.

    Raises:
        ValueError: for a storage other than ``atom`` or ``graph``.
    """
    if storage not in STORAGES:
        raise ValueError(
            f"a feature is declared per {storage!r}; use 'atom' or 'graph'."
        )
    level = NODE_LEVEL if storage == "atom" else GRAPH_LEVEL
    return FieldSpec(shape=(level, ANY_TRAILING), dtype=FLOAT)


def _check_name(name: str, kind: str) -> None:
    if not name or not name.strip():
        raise ValueError(f"a declared {kind} has an empty name.")
    if name in GRAPH_CORE:
        raise ValueError(
            f"{kind} {name!r} is named like a key of the graph core. Give it a "
            f"name of its own."
        )
    if name.endswith(PRESENCE_SUFFIX) or name.endswith(WEIGHT_SUFFIX):
        raise ValueError(
            f"{kind} {name!r} ends like a presence flag or a loss weight. Give "
            f"it a name of its own."
        )


def _check_declared(name: str, spec: FieldSpec, kind: str) -> None:
    """A declared input or target is floating, per atom or per structure, and
    a default name keeps the level and shape the data stores it with."""
    if spec.dtype != FLOAT or spec.shape[0] not in (NODE_LEVEL, GRAPH_LEVEL):
        raise ValueError(
            f"{kind} {name!r} is declared as {spec.dtype} of shape "
            f"{spec.shape}; a declared {kind} is floating, per atom or per "
            f"structure."
        )
    stored = _CONVENTION_FIELDS.get(name)
    if stored is not None and spec.shape[-1] != ANY_TRAILING and spec.shape != stored:
        raise ValueError(
            f"{kind} {name!r} is declared with shape {spec.shape}; the data "
            f"stores it as {stored}. Declare it with per_atom="
            f"{stored[0] == NODE_LEVEL} and irreps that fill it."
        )


class GraphLayout(ConfigSection):
    """What a run batches, as plain data that round-trips through JSON.

    Frozen: its fields cannot be reassigned, and a change is a new layout. Every
    construction path, :func:`graph_layout` or a record read back, is checked
    the same way.

    Attributes:
        atomic_numbers: The model's element table, in its order.
        heads: The model's head names, in order; a structure's ``head`` is
            the index of its head here.
        cutoff: The neighbour cutoff, Angstrom. A list searched with another
            cutoff is refused.
        float_dtype: The floating dtype every ``"float"`` field carries.
        inputs: The model's declared inputs and embedding features, by name.
        targets: The trainer's declared labels, by name. Each carries a
            presence flag and a loss weight beside it.
        requirement_fields: The fields the model's batch requirements build
            on the collated batch, by name. A batch carries all of them or
            none, the latter until they are built.
    """

    atomic_numbers: tuple[int, ...]
    heads: tuple[str, ...]
    cutoff: float = Field(gt=0, allow_inf_nan=False, strict=True)
    float_dtype: FloatDtype = "float64"
    inputs: dict[str, FieldSpec] = Field(default_factory=dict)
    targets: dict[str, FieldSpec] = Field(default_factory=dict)
    requirement_fields: dict[str, FieldSpec] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _element_table(cls, data: object) -> object:
        if isinstance(data, Mapping) and "atomic_numbers" in data:
            # AtomicNumberTable refuses a float, which int() would truncate
            # into a different element, and a repeated element.
            try:
                table = AtomicNumberTable(data["atomic_numbers"])
            except TypeError as error:
                raise ValueError(f"the element table: {error}") from None
            data = {**data, "atomic_numbers": table.zs}
        return data

    @model_validator(mode="after")
    def _validate(self) -> GraphLayout:
        if not self.atomic_numbers or any(
            not 1 <= z <= 118 for z in self.atomic_numbers
        ):
            raise ValueError(
                f"the element table {list(self.atomic_numbers)} is empty or "
                f"holds a number outside 1 to 118. Padding atoms carry 0, which "
                f"is never an element."
            )
        if (
            not self.heads
            or len(set(self.heads)) != len(self.heads)
            or not all(head.strip() for head in self.heads)
        ):
            raise ValueError(
                f"the head list {list(self.heads)} is empty, repeats a head or "
                f"holds an empty name; each head has one name and one index."
            )
        for kind, fields in (
            ("input", self.inputs),
            ("target", self.targets),
            ("batch-requirement field", self.requirement_fields),
        ):
            for name in fields:
                _check_name(name, kind)
        for name, spec in self.inputs.items():
            if name in ENGINE_INPUTS:
                raise ValueError(
                    f"input {name!r} is created by the derivative engine, not "
                    f"batched. Leave it out of the layout."
                )
            _check_declared(name, spec, "input")
        for name, spec in self.targets.items():
            if spec.shape[-1] == ANY_TRAILING:
                raise ValueError(
                    f"target {name!r} has no fixed shape; a label is declared "
                    f"with irreps."
                )
            _check_declared(name, spec, "target")
        if STRUCTURE_WEIGHT in self.targets:
            raise ValueError(
                f"target {STRUCTURE_WEIGHT!r} is named like the loss weight of a "
                f"whole structure, which the targets already carry. Give it a "
                f"name of its own."
            )
        for first, second in (
            (self.inputs, self.targets),
            (self.inputs, self.requirement_fields),
            (self.targets, self.requirement_fields),
        ):
            shared = sorted(set(first) & set(second))
            if shared:
                raise ValueError(
                    f"{shared} are declared twice, as two of an input, a target "
                    f"and a batch-requirement field. Each name is one field."
                )
        return self

    def graph_fields(self) -> dict[str, FieldSpec]:
        """Every key the graph may carry: core, inputs and requirement fields."""
        return {**GRAPH_CORE, **self.inputs, **self.requirement_fields}

    def target_fields(self) -> dict[str, FieldSpec]:
        """Every key of the targets: each label, its presence flag and its
        weight, and the structure weight."""
        fields = {STRUCTURE_WEIGHT: FieldSpec(shape=(GRAPH_LEVEL,), dtype=FLOAT)}
        for name, spec in self.targets.items():
            fields[name] = spec
            fields[presence_key(name)] = FieldSpec(shape=spec.shape[:1], dtype="bool")
            fields[weight_key(name)] = FieldSpec(
                shape=(GRAPH_LEVEL, *_WEIGHT_SHAPES.get(name, ())), dtype=FLOAT
            )
        return fields


def _unique(names: Iterable[str], kind: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(
                f"{kind} {name!r} is declared twice. Declare each name once, "
                f"so no declaration depends on the order it was given in."
            )
        seen.add(name)


def graph_layout(
    atomic_numbers: Iterable[int],
    *,
    cutoff: float,
    heads: Sequence[str] = (DEFAULT_HEAD,),
    float_dtype: FloatDtype = "float64",
    inputs: Iterable[InputSpec] = (),
    features: Mapping[str, Storage] | None = None,
    targets: Iterable[ObservableSpec] = (),
    requirement_fields: Mapping[str, FieldSpec] | None = None,
) -> GraphLayout:
    """The layout of a run, from what the model and the trainer declare.

    Args:
        atomic_numbers: The model's element table.
        cutoff: The neighbour cutoff, Angstrom.
        heads: The model's head names, in order.
        float_dtype: The run's floating dtype.
        inputs: The inputs the model reads.
        features: Embedding features, by name, stored per ``atom`` or ``graph``.
        targets: The observables the trainer fits.
        requirement_fields: The fields the model's batch requirements build.

    Raises:
        ValueError: if a name is declared twice, a feature has the name of an
            input, or the layout is invalid.
    """
    inputs, targets = tuple(inputs), tuple(targets)
    features = dict(features or {})
    _unique([spec.name for spec in inputs], "input")
    _unique([spec.name for spec in targets], "target")
    declared_inputs = {spec.name: input_field(spec) for spec in inputs}
    for name, storage in features.items():
        if name in declared_inputs:
            raise ValueError(
                f"feature {name!r} has the name of a declared input. A feature "
                f"cannot stand in for an input; give it a name of its own."
            )
        declared_inputs[name] = feature_field(storage)
    return GraphLayout(
        atomic_numbers=tuple(atomic_numbers),
        heads=tuple(heads),
        cutoff=cutoff,
        float_dtype=float_dtype,
        inputs=declared_inputs,
        targets={spec.name: target_field(spec) for spec in targets},
        requirement_fields=dict(requirement_fields or {}),
    )
