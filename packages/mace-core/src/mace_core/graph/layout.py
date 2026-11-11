"""What a run batches, declared once: the :class:`GraphLayout`.

The graph is the structural core of :mod:`mace_core.graph.schema` plus what
the layout declares. The model declares the inputs it reads (``InputSpec``s and
embedding features) and its batch requirements; the trainer declares the
targets, each an observable with a label, a presence flag and a loss weight.
The layout also carries the element table, the head list and the cutoff a
neighbour list must have been searched with.

A field's level comes from ``per_atom`` and its trailing shape from the irreps,
so a property declared in config becomes a batched, padded field with no new
code. The names of the twelve default keys keep the shapes the data contract
gives them (``forces`` is ``(3,)`` per atom, ``stress`` ``(3, 3)``), whatever
their irreps spell.

The layout is resolved once per run, passed to
:func:`~mace_core.graph.build.graph_sample`, collation, padding and
:func:`~mace_core.graph.validation.check_layout`, and recorded in
``config.resolved``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from mace_core.data.configuration import DEFAULT_HEAD
from mace_core.elements.default_keys import STORAGES, DefaultKeys, Storage
from mace_core.graph.schema import (
    ANY_TRAILING,
    FLOAT,
    GRAPH_CORE,
    GRAPH_LEVEL,
    NODE_LEVEL,
    NOT_KEYS,
    PRESENCE_SUFFIX,
    WEIGHT_SUFFIX,
    FieldSpec,
    presence_key,
    weight_key,
)
from mace_core.observables import InputSpec, ObservableSpec
from mace_core.observables.grammar import irreps_dimension, parse_irreps

__all__ = [
    "DEFAULT_INPUTS",
    "DEFAULT_TARGETS",
    "ENGINE_INPUTS",
    "GraphLayout",
    "default_layout",
    "feature_field",
    "input_field",
    "target_field",
]

#: Inputs the derivative engine creates itself; never batched.
ENGINE_INPUTS: frozenset[str] = frozenset({"pos", "strain"})

#: Weights whose shape is not one value per structure, as develop stores them.
_WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    "dipole": (3,),
    "polarizability": (3, 3),
}

_CONVENTION_SHAPES: dict[str, tuple[int, ...]] = {
    member.convention_name: member.value_shape for member in DefaultKeys
}


def _declared(
    name: str,
    per_atom: bool,
    value_shape: tuple[int, ...],
    doc: str,
    absent_value: float = 0.0,
) -> FieldSpec:
    level = NODE_LEVEL if per_atom else GRAPH_LEVEL
    return FieldSpec((level, *value_shape), FLOAT, True, "data", doc, absent_value)


def _value_shape(name: str, irreps: str, *, observable: str) -> tuple[int, ...]:
    if name in _CONVENTION_SHAPES:
        return _CONVENTION_SHAPES[name]
    terms = parse_irreps(irreps, observable=observable)
    dimension = irreps_dimension(irreps, observable=observable)
    scalar = len(terms) == 1 and terms[0].degree == 0 and terms[0].multiplicity == 1
    return () if scalar else (dimension,)


def input_field(spec: InputSpec, *, absent_value: float = 0.0) -> FieldSpec:
    """The batched field of a declared model input.

    Raises:
        ValueError: for ``pos`` or ``strain``, which the derivative engine
            creates rather than reads from a batch.
    """
    if spec.name in ENGINE_INPUTS:
        raise ValueError(
            f"input {spec.name!r} is created by the derivative engine, not "
            f"batched. Leave it out of the layout."
        )
    shape = _value_shape(spec.name, spec.irreps, observable=spec.name)
    doc = f"Declared input, {spec.units}."
    return _declared(spec.name, spec.per_atom, shape, doc, absent_value)


def target_field(spec: ObservableSpec) -> FieldSpec:
    """The batched label of a declared observable."""
    shape = _value_shape(spec.name, spec.irreps, observable=spec.name)
    return _declared(spec.name, spec.per_atom, shape, f"Label, {spec.units}.")


def feature_field(name: str, storage: Storage) -> FieldSpec:
    """The batched field of an embedding feature, stored per ``atom`` or per
    ``graph``. Its trailing sizes are the data's own; it is floating.

    Raises:
        ValueError: for a storage other than ``atom`` or ``graph``, so a
            misspelling such as ``"atoms"`` is refused rather than read as
            per structure.
    """
    if storage not in STORAGES:
        raise ValueError(
            f"feature {name!r} is declared per {storage!r}; use 'atom' or 'graph'."
        )
    level = NODE_LEVEL if storage == "atom" else GRAPH_LEVEL
    return FieldSpec(
        (level, ANY_TRAILING), FLOAT, True, "data", f"Embedding feature per {storage}."
    )


#: The inputs develop batches for every model, by name.
DEFAULT_INPUTS: Mapping[str, FieldSpec] = MappingProxyType(
    {
        "magmom": _declared("magmom", True, (3,), "Magnetic moments, μB."),
        "total_charge": _declared("total_charge", False, (), "Total charge, e."),
        "total_spin": _declared(
            "total_spin",
            False,
            (),
            "Total spin multiplicity; 1 when absent, and 1 in padding.",
            absent_value=1.0,
        ),
        "elec_temp": _declared("elec_temp", False, (), "Electronic temperature, K."),
        "fermi_level": _declared("fermi_level", False, (), "Fermi level, eV."),
        "external_field": _declared(
            "external_field", False, (3,), "External electric field, V/Å."
        ),
    }
)

#: The labels develop batches for every training run, by name.
DEFAULT_TARGETS: Mapping[str, FieldSpec] = MappingProxyType(
    {
        "energy": _declared("energy", False, (), "Total energy, eV."),
        "forces": _declared("forces", True, (3,), "Forces, eV/Å."),
        "magforces": _declared("magforces", True, (3,), "Magnetic forces."),
        "stress": _declared("stress", False, (3, 3), "Stress, eV/Å³."),
        "virials": _declared("virials", False, (3, 3), "Virials, eV."),
        "dipole": _declared("dipole", False, (3,), "Total dipole, e·Å."),
        "charges": _declared("charges", True, (), "Per-atom charges, e."),
        "polarizability": _declared(
            "polarizability", False, (3, 3), "Polarizability tensor."
        ),
    }
)


def _check_name(name: str, kind: str) -> None:
    if not name or not name.strip():
        raise ValueError(f"a declared {kind} has an empty name.")
    if name in GRAPH_CORE or name in NOT_KEYS:
        raise ValueError(
            f"{kind} {name!r} is named like a key of the graph core or a name "
            f"the graph never carries. Give it a name of its own."
        )
    if name.endswith(PRESENCE_SUFFIX) or name.endswith(WEIGHT_SUFFIX):
        raise ValueError(
            f"{kind} {name!r} ends like a presence flag or a loss weight. Give "
            f"it a name of its own."
        )


@dataclass(frozen=True, eq=False)
class GraphLayout:
    """What a run batches. Immutable; equal layouts hash equal.

    Attributes:
        atomic_numbers: The model's element table, in its order.
        heads: The model's head names, in order; a structure's ``head`` is
            the index of its head here.
        cutoff: The neighbour cutoff, Angstrom. A list searched with another
            cutoff is refused.
        inputs: The model's declared inputs and embedding features, by name.
        targets: The trainer's declared labels, by name. Each carries a
            presence flag and a loss weight beside it.
        requirements: The model's batch requirements. A requirement that
            exposes ``fields() -> Mapping[str, FieldSpec]`` has its fields
            checked with the rest of the batch; DATA-4 defines the type.
    """

    atomic_numbers: tuple[int, ...]
    heads: tuple[str, ...]
    cutoff: float
    inputs: Mapping[str, FieldSpec] = field(default_factory=dict)
    targets: Mapping[str, FieldSpec] = field(default_factory=dict)
    requirements: frozenset[Any] = frozenset()

    def __post_init__(self) -> None:
        numbers = tuple(int(z) for z in self.atomic_numbers)
        if not numbers or len(set(numbers)) != len(numbers):
            raise ValueError(
                f"the element table {list(numbers)} is empty or repeats an "
                f"element; each element has one index."
            )
        if any(not 1 <= z <= 118 for z in numbers):
            raise ValueError(
                f"the element table {list(numbers)} holds a number outside 1 to "
                f"118. Padding atoms carry 0, which is never an element."
            )
        heads = tuple(self.heads)
        repeated = sorted({head for head in heads if heads.count(head) > 1})
        if not heads or repeated:
            raise ValueError(
                f"the head list {list(heads)} is empty or repeats {repeated}; "
                f"each head has one index."
            )
        cutoff = float(self.cutoff)
        if not (math.isfinite(cutoff) and cutoff > 0):
            raise ValueError(
                f"cutoff is {self.cutoff!r}. Expected a positive, finite distance "
                f"in Angstrom."
            )
        for name in self.inputs:
            _check_name(name, "input")
        for name in self.targets:
            _check_name(name, "target")
        shared = sorted(set(self.inputs) & set(self.targets))
        if shared:
            raise ValueError(
                f"{shared} are declared both as an input and as a target. A "
                f"quantity the model reads cannot be the label it is trained on."
            )
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "heads", heads)
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))
        object.__setattr__(self, "requirements", frozenset(self.requirements))
        self.requirement_fields()

    def _key(self) -> tuple[Any, ...]:
        return (
            self.atomic_numbers,
            self.heads,
            self.cutoff,
            tuple(self.inputs.items()),
            tuple(self.targets.items()),
            self.requirements,
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GraphLayout) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def element_index(self, atomic_number: int) -> int:
        """The index of ``atomic_number`` in the element table.

        Raises:
            ValueError: for an element the table does not hold, which means a
                structure the model was not fitted for.
        """
        try:
            return self.atomic_numbers.index(int(atomic_number))
        except ValueError:
            raise ValueError(
                f"element {int(atomic_number)} is not in the model's element "
                f"table {list(self.atomic_numbers)}."
            ) from None

    def requirement_fields(self) -> Mapping[str, FieldSpec]:
        """The fields the batch requirements declare, by name.

        Raises:
            ValueError: if two requirements, or a requirement and the core or
                an input, declare the same name.
        """
        fields: dict[str, FieldSpec] = {}
        for requirement in sorted(self.requirements, key=repr):
            declared = getattr(requirement, "fields", None)
            if declared is None:
                continue
            for name, spec in declared().items():
                if name in fields or name in GRAPH_CORE or name in self.inputs:
                    raise ValueError(
                        f"batch requirement {requirement!r} declares {name!r}, "
                        f"which another field of the batch already has."
                    )
                fields[name] = spec
        return MappingProxyType(fields)

    def graph_fields(self) -> Mapping[str, FieldSpec]:
        """Every key the graph may carry: core, inputs and requirement fields."""
        return MappingProxyType(
            {**GRAPH_CORE, **self.inputs, **self.requirement_fields()}
        )

    def target_fields(self) -> Mapping[str, FieldSpec]:
        """Every key of the targets: each label, its presence flag and its
        weight, and the structure weight ``weight``."""
        fields: dict[str, FieldSpec] = {
            "weight": FieldSpec(
                (GRAPH_LEVEL,),
                FLOAT,
                True,
                "data",
                "Loss weight per structure; 1 when the file gives none.",
            )
        }
        for name, spec in self.targets.items():
            fields[name] = spec
            fields[presence_key(name)] = FieldSpec(
                (spec.shape[0],),
                "bool",
                True,
                "derived",
                f"Whether {name} is labelled, from Configuration.is_labelled. A "
                f"zero weight does not make a present label absent.",
            )
            fields[weight_key(name)] = FieldSpec(
                (GRAPH_LEVEL, *_WEIGHT_SHAPES.get(name, ())),
                FLOAT,
                True,
                "data",
                f"Loss weight of {name}; 0 where it is absent.",
            )
        return MappingProxyType(fields)


def default_layout(
    atomic_numbers: Iterable[int],
    *,
    cutoff: float,
    heads: Sequence[str] = (DEFAULT_HEAD,),
    features: Mapping[str, Storage] | None = None,
    inputs: Iterable[InputSpec] = (),
    targets: Iterable[ObservableSpec] = (),
    requirements: Iterable[Any] = (),
) -> GraphLayout:
    """The layout develop batches today, plus what a run declares beyond it.

    Args:
        atomic_numbers: The model's element table.
        cutoff: The neighbour cutoff, Angstrom.
        heads: The model's head names, in order.
        features: Embedding features, by name, stored per ``atom`` or ``graph``.
        inputs: Inputs declared beyond the defaults.
        targets: Observables with labels declared beyond the defaults.
        requirements: The model's batch requirements.
    """
    declared_inputs = dict(DEFAULT_INPUTS)
    for spec in inputs:
        declared_inputs[spec.name] = input_field(spec)
    for name, storage in (features or {}).items():
        if name in declared_inputs:
            raise ValueError(
                f"feature {name!r} has the name of a declared input. A feature "
                f"cannot stand in for an input; give it a name of its own."
            )
        declared_inputs[name] = feature_field(name, storage)
    declared_targets = dict(DEFAULT_TARGETS)
    for spec in targets:
        declared_targets[spec.name] = target_field(spec)
    return GraphLayout(
        atomic_numbers=tuple(atomic_numbers),
        heads=tuple(heads),
        cutoff=cutoff,
        inputs=declared_inputs,
        targets=declared_targets,
        requirements=frozenset(requirements),
    )
