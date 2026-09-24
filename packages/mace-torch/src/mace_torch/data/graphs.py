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
from mace_core.graph import GRAPH_INPUT_DEFAULTS, NODE_INPUT_DEFAULTS
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
    """A declared observable no structure in the dataset carries a value for."""


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
                specs.append(
                    TargetSpec(name, requested.per_atom_derivative(request.wrt))
                )
    return tuple(specs)


def graph_from_configuration(
    configuration: Configuration,
    *,
    cutoff: float,
    z_table: AtomicNumberTable,
    head: int = 0,
    weight: float = 1.0,
    graph_inputs: Sequence[str] = (),
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
        graph_inputs: The inputs the model reads, by name, per structure or per
            atom. Each is taken from the structure's properties, or from
            :data:`~mace_core.graph.GRAPH_INPUT_DEFAULTS` or
            :data:`~mace_core.graph.NODE_INPUT_DEFAULTS` when it has none.

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
        **{
            name: (
                _node_input(configuration, name, len(numbers))
                if name in NODE_INPUT_DEFAULTS
                else _graph_input(configuration, name, GRAPH_INPUT_DEFAULTS[name])
            )
            for name in graph_inputs
        },
    }


def _node_input(configuration: Configuration, name: str, num_atoms: int) -> np.ndarray:
    """One per-atom input: ``[n_atoms]`` for a scalar, ``[n_atoms, k]`` otherwise.

    Raises:
        ValueError: If the structure gives a number of values other than one
            per atom and component.
    """
    default = NODE_INPUT_DEFAULTS[name]
    shape = (num_atoms,) if len(default) == 1 else (num_atoms, len(default))
    value = configuration.properties.get(name)
    if value is None:
        return (
            np.broadcast_to(np.asarray(default, dtype=float), shape)
            .reshape(shape)
            .copy()
        )
    array = np.asarray(value, dtype=float)
    if array.size != int(np.prod(shape)):
        raise ValueError(
            f"{name!r} has {array.size} value(s) and is {len(default)} per atom "
            f"of {num_atoms}: {array.tolist()}."
        )
    return array.reshape(shape)


def _graph_input(
    configuration: Configuration, name: str, default: tuple[float, ...]
) -> np.ndarray:
    """One per-structure input, shaped as the schema declares it.

    A structure that gives one with the wrong number of components is refused
    by name, rather than broadcast or truncated into another value.
    """
    value = configuration.properties.get(name)
    array = np.asarray(default if value is None else value, dtype=float).reshape(-1)
    if array.size != len(default):
        raise ValueError(
            f"{name!r} has {array.size} component(s) and is one per structure "
            f"of {len(default)}: {array.tolist()}."
        )
    return array if len(default) > 1 else array.reshape(())


def targets_from_configuration(
    configuration: Configuration, specs: Sequence[TargetSpec], num_atoms: int
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """The reference values and how much each counts, for one structure.

    A per-graph value gets a leading axis of one, so every target joins along
    axis zero whether it is one number per structure or three per atom. Without
    it the two kinds need two code paths in the collation, and the one that is
    wrong stays quiet.

    **A property this structure does not carry becomes zeros with weight
    zero**, rather than an error. A fitting database that labels energies for
    everything and forces for a tenth of it is ordinary, and refusing it would
    mean splitting it into two runs. The weight is what makes the term
    contribute nothing: not a mask a loss has to remember to apply, and not a
    NaN. A property no structure anywhere carries is a different thing, and the
    stage that reads the dataset refuses that one.

    Returns:
        The values, and the weight each carries for this structure. The weight
        is the configuration's own when it names one, one when it does not, and
        zero when the value is absent.
    """
    values: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}
    properties: Mapping[str, object] = configuration.properties
    for spec in specs:
        value = properties.get(spec.name)
        if value is None:
            values[spec.name] = np.zeros(
                (num_atoms, 3) if spec.per_atom else (1,), dtype=float
            )
            weights[spec.name] = 0.0
            continue
        array = np.asarray(value, dtype=float)
        values[spec.name] = array if spec.per_atom else array.reshape(1, *array.shape)
        weights[spec.name] = float(configuration.property_weights.get(spec.name, 1.0))
    return values, weights
