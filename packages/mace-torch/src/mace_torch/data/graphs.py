"""From a parsed structure to the two things a training step needs.

A :class:`~mace_core.data.configuration.Configuration` is what a file says. A
graph is what the model reads, and a target set is what the loss compares
against. They are built together here, in one pass over the structure, because
they have to agree: a graph whose nodes were reordered against its forces is a
run that trains on noise and reports nothing.

**The two are kept apart.** The graph is exactly the schema in
:mod:`mace_core.graph`, which is the model's input and nothing else; the
reference values live beside it. The frozen tree puts both in one object and
then has to remember which keys the model may read, which is how a label
reaches a forward pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.neighbors import get_neighborhood
from mace_core.observables import RequestedOutputs

__all__ = [
    "MissingTargetError",
    "TargetSpec",
    "graph_from_configuration",
    "target_specs",
    "targets_from_configuration",
]


class MissingTargetError(KeyError):
    """A declared observable the structure carries no value for."""


@dataclass(frozen=True)
class TargetSpec:
    """One reference quantity the loss compares a model output against.

    Attributes:
        name: The key, in the convention names the data layer resolved to.
        per_atom: Whether the value has one row per atom. It decides how the
            value is joined into a batch, and getting it wrong is silent: a
            per-graph quantity concatenated per node gives an array of the
            right dtype and the wrong length.
    """

    name: str
    per_atom: bool


def target_specs(requested: RequestedOutputs) -> tuple[TargetSpec, ...]:
    """What a run that asked for these outputs trains against.

    Driven by what the configuration requested, not by what the catalogue
    declares. ``energy`` carries a stress derivative whether or not anyone
    asked for one, and reading the declaration instead would demand a stress
    from every dataset.
    """
    specs = [
        TargetSpec(observable.name, observable.per_atom)
        for observable in requested.observables
    ]
    for observable in requested.observables:
        for request in observable.derivatives:
            name = observable.derivative_name(request.wrt)
            if name in requested.derivatives:
                specs.append(TargetSpec(name, request.wrt == "pos"))
    return tuple(specs)


def graph_from_configuration(
    configuration: Configuration,
    *,
    cutoff: float,
    z_table: AtomicNumberTable,
    head: int = 0,
    weight: float = 1.0,
) -> dict[str, np.ndarray]:
    """One structure's contribution to a batch, as the schema declares it.

    Args:
        configuration: The parsed structure.
        cutoff: The neighbour radius, in Angstrom.
        z_table: The element table. Present so that an element the model was
            not built for is refused here rather than indexing out of the
            one-hot at the first forward.
        head: Which head this structure belongs to, as a position in the
            model's head list.
        weight: Its weight in the loss.

    Raises:
        ValueError: If the structure holds an element the table does not.
    """
    numbers = np.asarray(configuration.atomic_numbers, dtype=np.int64)
    unknown = sorted({int(z) for z in numbers} - set(z_table.zs))
    if unknown:
        raise ValueError(
            f"the structure holds element(s) {unknown} and the element table "
            f"is {list(z_table.zs)}. A model cannot embed an element it was "
            f"not built for."
        )
    positions = np.asarray(configuration.positions, dtype=float)
    pbc = (
        (False, False, False) if configuration.pbc is None else tuple(configuration.pbc)
    )
    neighborhood = get_neighborhood(positions, cutoff, pbc, configuration.cell)
    # The cell the neighbour search returns, never the one the structure came
    # with. It is the physical cell in every regime but two, and both are
    # deliberate: a padding box for an aperiodic structure, whose stress the
    # model masks, and an inflated row for a non-periodic axis whose physical
    # row is all zeros, which would otherwise make the volume zero and the
    # stress of a slab with no vacuum a division by zero.
    cell = neighborhood.cell
    # The per-graph fields carry no leading axis: the collation stacks them,
    # and `batch` and `ptr` are built there rather than here. So this is a
    # structure's contribution to a batch and not a batch of one, which is why
    # it does not satisfy `validate_graph` on its own.
    return {
        "positions": positions,
        "atomic_numbers": numbers,
        "edge_index": np.asarray(neighborhood.edge_index, dtype=np.int64),
        "shifts": np.asarray(neighborhood.shifts, dtype=float),
        "unit_shifts": np.asarray(neighborhood.unit_shifts, dtype=float),
        "cell": np.asarray(cell, dtype=float).reshape(3, 3),
        "pbc": np.asarray(pbc, dtype=bool).reshape(3),
        "weight": np.asarray(weight, dtype=float),
        "head": np.asarray(head, dtype=np.int64),
    }


def targets_from_configuration(
    configuration: Configuration, specs: Sequence[TargetSpec]
) -> dict[str, np.ndarray]:
    """The reference values, shaped so a batch is a concatenation.

    A per-graph value gets a leading axis of one, so every target joins along
    axis zero whether it is one number per structure or three per atom. Without
    it the two kinds need two code paths in the collation, and the one that is
    wrong stays quiet.

    Raises:
        MissingTargetError: Naming the key and what declared it. Training a
            head against a property the file does not carry is the failure this
            replaces, and on the frozen tree it shows up as a loss term that is
            always zero.
    """
    values: dict[str, np.ndarray] = {}
    properties: Mapping[str, object] = configuration.properties
    for spec in specs:
        value = properties.get(spec.name)
        if value is None:
            raise MissingTargetError(
                f"{spec.name!r} is declared and this structure carries no such "
                f"property. It has {sorted(properties)}. Either declare the "
                f"observable away or supply the value."
            )
        array = np.asarray(value, dtype=float)
        values[spec.name] = array if spec.per_atom else array.reshape(1, *array.shape)
    return values
