"""From a :class:`~mace_core.data.configuration.Configuration` to a graph sample.

Two steps, kept apart so the edges can come from anywhere:

* :func:`structure_graph` searches a structure's neighbour list, independently
  of any model, in a loader worker or in preprocessing;
* :func:`graph_sample` builds one structure's graph and targets from a list it
  is given, a stored one, or none when a device search joins the edges after
  collation.

The graph holds the structural core and every input the layout declares, with
its absent value when the structure has none (1 for ``total_spin``, 0 for the
others). The targets hold every label the layout declares, with a
``<label>_present`` flag from :meth:`Configuration.is_labelled` and a
``<label>_weight`` from the property weights; a label the structure lacks is
zeros with the flag false and the weight zero, and a present label with a zero
weight is still present. A property nobody declares never reaches either.

The per-sample value checks run here, on every structure, in the worker that
builds it, and name the structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mace_core.data.configuration import Configuration
from mace_core.graph.layout import GraphLayout
from mace_core.graph.schema import (
    ANY_TRAILING,
    NODE_LEVEL,
    FieldSpec,
    presence_key,
    weight_key,
)
from mace_core.neighbors.backends import NeighborListBackend
from mace_core.neighbors.geometry import graph_cell
from mace_core.neighbors.neighborhood import NeighborList, get_neighborhood

__all__ = ["SampleError", "graph_sample", "structure_graph"]

#: Targets a structure with a periodic axis needs a volume for.
_VOLUME_TARGETS = frozenset({"stress"})


class SampleError(ValueError):
    """A structure that cannot become a graph sample. The message names it."""


def structure_graph(
    config: Configuration, cutoff: float, backend: NeighborListBackend
) -> NeighborList:
    """The neighbour list of ``config``, searched with ``backend``.

    Args:
        config: The parsed structure. Not modified.
        cutoff: The neighbour cutoff, Angstrom.
        backend: The resolved backend, from
            :func:`~mace_core.neighbors.resolve_neighbor_backend`.
    """
    return get_neighborhood(
        config.positions, cutoff, pbc=config.pbc, cell=config.cell, backend=backend
    )


def graph_sample(
    config: Configuration,
    neighbors: NeighborList | None,
    layout: GraphLayout,
    *,
    source: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One structure's graph and targets.

    Args:
        config: The parsed structure. Not modified.
        neighbors: Its neighbour list, from :func:`structure_graph` or a
            stored one; ``None`` builds a sample without edges, whose batch a
            device search completes.
        layout: What the run batches.
        source: Where the structure came from, named in an error.

    Returns:
        ``(graph, targets)``, dicts of numpy arrays: floats as float64,
        indices as int64, flags as bool. Graph-level values carry no leading
        graph dimension.

    Raises:
        SampleError: naming the structure, for a value that cannot be batched:
            non-finite positions, cell or labels; an element outside the table;
            a head outside the head list; a periodic axis with an all-zero cell
            row; a stress label on a partly periodic structure whose cell has
            zero volume; an input, feature or label of the wrong shape; a
            declared feature the structure lacks; or a list with indices
            outside the structure or searched with another cutoff.
    """
    where = _where(config, source)
    n_atoms = len(config)
    pbc = (False, False, False) if config.pbc is None else tuple(config.pbc)
    positions = np.array(config.positions, dtype=np.float64)
    if not np.isfinite(positions).all():
        raise SampleError(f"{where} has a non-finite position.")
    if config.cell is not None and not np.isfinite(config.cell).all():
        raise SampleError(f"{where} has a non-finite cell.")
    try:
        cell = graph_cell(pbc, config.cell)
    except ValueError as error:
        raise SampleError(f"{where}: {error}") from None
    for axis in range(3):
        if pbc[axis] and not cell[axis].any():
            raise SampleError(
                f"{where} is periodic along axis {axis} but its lattice vector is "
                f"all zero, so there is no image to repeat. Give the cell a "
                f"non-zero row {axis}, or set pbc[{axis}] to False."
            )
    if config.head not in layout.heads:
        raise SampleError(
            f"{where} belongs to head {config.head!r}, which is not one of the "
            f"model's heads {list(layout.heads)}."
        )
    try:
        for z in config.atomic_numbers:
            layout.element_index(int(z))
    except ValueError as error:
        raise SampleError(f"{where}: {error}") from None

    graph: dict[str, Any] = {
        "positions": positions,
        "node_atomic_numbers": np.array(config.atomic_numbers, dtype=np.int64),
        "cell": np.array(cell),
        "pbc": np.array(pbc, dtype=bool),
        "head": np.int64(layout.heads.index(config.head)),
    }
    if neighbors is not None:
        graph.update(_edges(neighbors, n_atoms, pbc, layout.cutoff, where))
    for name, spec in layout.inputs.items():
        graph[name] = _input_value(
            name, spec, config.properties.get(name), n_atoms, where
        )
    return graph, _targets(config, layout, n_atoms, pbc, cell, where)


def _where(config: Configuration, source: str | None) -> str:
    label = f"the structure from {source}" if source else "a structure"
    return f"{label} ({len(config)} atoms, head {config.head!r})"


def _edges(
    neighbors: NeighborList,
    n_atoms: int,
    pbc: tuple[bool, ...],
    cutoff: float,
    where: str,
) -> dict[str, np.ndarray]:
    if float(neighbors.cutoff) != cutoff:
        raise SampleError(
            f"{where} comes with a neighbour list searched at cutoff "
            f"{neighbors.cutoff} Å; the model's cutoff is {cutoff} Å. Search it "
            f"again, or use the dataset prepared for this cutoff."
        )
    edge_index = np.asarray(neighbors.edge_index)
    unit_shifts = np.asarray(neighbors.unit_shifts)
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= n_atoms):
        raise SampleError(
            f"{where} comes with a neighbour list whose edges reach atom "
            f"{int(edge_index.max())} or {int(edge_index.min())}."
        )
    aperiodic = [axis for axis in range(3) if not pbc[axis]]
    if unit_shifts[:, aperiodic].any():
        raise SampleError(
            f"{where} comes with a neighbour list that has images along the "
            f"aperiodic axes {aperiodic}."
        )
    return {
        "edge_index": np.array(edge_index, dtype=np.int64),
        "shifts": np.array(neighbors.shifts, dtype=np.float64),
        "unit_shifts": unit_shifts.astype(np.float64),
    }


def _shape(spec: FieldSpec, n_atoms: int) -> tuple[int, ...]:
    """The shape one structure's value of a fixed-shape field has."""
    trailing = tuple(int(n) for n in spec.shape[1:])
    return (n_atoms, *trailing) if spec.shape[0] == NODE_LEVEL else trailing


def _input_value(
    name: str, spec: FieldSpec, value: Any, n_atoms: int, where: str
) -> np.ndarray:
    if spec.shape[-1] == ANY_TRAILING:
        return _feature_value(name, spec, value, n_atoms, where)
    shape = _shape(spec, n_atoms)
    if value is None:
        return np.full(shape, spec.absent_value, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise SampleError(
            f"{where} has input {name!r} of shape {array.shape}; the layout "
            f"declares {shape}."
        )
    if not np.isfinite(array).all():
        raise SampleError(f"{where} has a non-finite value in input {name!r}.")
    return array


def _feature_value(
    name: str, spec: FieldSpec, value: Any, n_atoms: int, where: str
) -> np.ndarray:
    if value is None:
        raise SampleError(
            f"{where} has no value for the embedding feature {name!r}, which the "
            f"model declares. An input cannot be made up; add it to the file."
        )
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise SampleError(
            f"{where} has embedding feature {name!r} of dtype {array.dtype}; a "
            f"feature is a real number."
        )
    array = array.astype(np.float64)
    if spec.shape[0] == NODE_LEVEL and (array.ndim == 0 or array.shape[0] != n_atoms):
        raise SampleError(
            f"{where} has feature {name!r}, declared per atom, of shape "
            f"{array.shape}; its first dimension must be the {n_atoms} atoms. It "
            f"is not reshaped."
        )
    return array


def _targets(
    config: Configuration,
    layout: GraphLayout,
    n_atoms: int,
    pbc: tuple[bool, ...],
    cell: np.ndarray,
    where: str,
) -> dict[str, Any]:
    targets: dict[str, Any] = {"weight": np.float64(config.weight)}
    weights = layout.target_fields()
    degenerate = any(pbc) and np.linalg.det(cell) == 0
    for name, spec in layout.targets.items():
        shape = _shape(spec, n_atoms)
        present = name in config.properties and config.is_labelled(name)
        if present:
            array = np.asarray(config.properties[name], dtype=np.float64)
            if array.shape != shape:
                raise SampleError(
                    f"{where} has label {name!r} of shape {array.shape}; the "
                    f"layout declares {shape}."
                )
            if not np.isfinite(array).all():
                raise SampleError(f"{where} has a non-finite value in label {name!r}.")
            if name in _VOLUME_TARGETS and degenerate:
                raise SampleError(
                    f"{where} carries a {name!r} label but its cell has zero "
                    f"volume, so there is no volume to compare it against. Give "
                    f"the aperiodic axis its vacuum row, or drop the label."
                )
        else:
            array = np.zeros(shape, dtype=np.float64)
        targets[name] = array
        flag_shape = (n_atoms,) if spec.shape[0] == NODE_LEVEL else ()
        targets[presence_key(name)] = np.full(flag_shape, present, dtype=bool)
        weight_spec = weights[weight_key(name)]
        weight = float(config.property_weights.get(name, 1.0)) if present else 0.0
        targets[weight_key(name)] = np.full(
            _shape(weight_spec, n_atoms), weight, dtype=np.float64
        )
    return targets
