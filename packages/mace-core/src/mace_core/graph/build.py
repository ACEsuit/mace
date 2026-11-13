"""From a :class:`~mace_core.data.configuration.Configuration` to a graph sample.

:func:`graph_sample` builds one structure's graph and targets from a neighbour
list it is given: one searched with
:func:`~mace_core.neighbors.get_neighborhood`, one stored with the data, or
none when a device search joins the edges after collation.

The graph holds the structural core and every input the layout declares, with
the input's absent value when the structure has none. The targets hold every
label the layout declares, with a ``<label>_present`` flag from
:meth:`Configuration.is_labelled` and a ``<label>_weight`` from the property
weights; a label the structure lacks is zeros with the flag false and the
weight zero, and a present label with a zero weight is still present. Every
declared name must be a declared property of the structure, so a misspelling
is an error rather than an absent value.

The per-sample value checks run here, on every structure, in the worker that
builds it, and name the structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mace_core.data.configuration import Configuration
from mace_core.graph.layout import STRUCTURE_WEIGHT, GraphLayout
from mace_core.graph.schema import FieldSpec, presence_key, weight_key
from mace_core.neighbors.geometry import graph_cell
from mace_core.neighbors.neighborhood import NeighborList, checked_edges

__all__ = ["SampleError", "graph_sample"]

#: Targets a structure with a periodic axis needs a volume for.
_VOLUME_TARGETS = frozenset({"stress"})


class SampleError(ValueError):
    """A structure that cannot become a graph sample. The message names it."""


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
        neighbors: Its neighbour list, searched or stored; ``None`` builds a
            sample without edges, whose batch a device search completes.
        layout: What the run batches.
        source: Where the structure came from, named in an error.

    Returns:
        ``(graph, targets)``, dicts of numpy arrays: floats as float64,
        indices as int64, flags as bool. Graph-level values carry no leading
        graph dimension.

    Raises:
        SampleError: naming the structure, for a value that cannot be batched:
            non-finite positions, cell or values; an element outside the
            table; a head outside the head list; a periodic axis with an
            all-zero cell row; a stress label on a partly periodic structure
            whose cell has zero volume; a declared name the structure does not
            declare; an input, feature or label of the wrong shape; a declared
            feature the structure lacks; or a malformed neighbour list, or one
            searched with another cutoff.
    """
    where = _where(config, source)
    n_atoms = len(config)
    pbc = (False, False, False) if config.pbc is None else tuple(config.pbc)
    positions = np.array(config.positions, dtype=np.float64)
    if not np.isfinite(positions).all():
        raise SampleError(f"{where} has a non-finite position.")
    try:
        cell = graph_cell(pbc, config.cell)
    except ValueError as error:
        raise SampleError(f"{where}: {error}") from None
    if config.head not in layout.heads:
        raise SampleError(
            f"{where} belongs to head {config.head!r}, which is not one of the "
            f"model's heads {list(layout.heads)}."
        )
    atomic_numbers = np.array(config.atomic_numbers, dtype=np.int64)
    unknown = np.setdiff1d(atomic_numbers, layout.atomic_numbers).tolist()
    if unknown:
        raise SampleError(
            f"{where} holds elements {unknown}, which are not in the model's "
            f"element table {list(layout.atomic_numbers)}."
        )

    graph: dict[str, Any] = {
        "positions": positions,
        "node_atomic_numbers": atomic_numbers,
        "cell": np.array(cell),
        "pbc": np.array(pbc, dtype=bool),
        "head": np.int64(layout.heads.index(config.head)),
    }
    if neighbors is not None:
        graph.update(_edges(neighbors, n_atoms, pbc, cell, layout.cutoff, where))
    for name, spec in layout.inputs.items():
        value = config.properties[name] if _labelled(config, name, where) else None
        graph[name] = _input_value(name, spec, value, n_atoms, where)
    for name, spec in layout.features.items():
        value = config.properties[name] if _labelled(config, name, where) else None
        graph[name] = _feature_value(name, spec, value, n_atoms, where)
    return graph, _targets(config, layout, n_atoms, pbc, cell, where)


def _where(config: Configuration, source: str | None) -> str:
    label = f"the structure from {source}" if source else "a structure"
    return f"{label} ({len(config)} atoms, head {config.head!r})"


def _labelled(config: Configuration, name: str, where: str) -> bool:
    try:
        return config.is_labelled(name)
    except ValueError as error:
        raise SampleError(
            f"{where}: the layout declares {name!r}, and {error}"
        ) from None


def _edges(
    neighbors: NeighborList,
    n_atoms: int,
    pbc: tuple[bool, ...],
    cell: np.ndarray,
    cutoff: float,
    where: str,
) -> dict[str, np.ndarray]:
    if float(neighbors.cutoff) != cutoff:
        raise SampleError(
            f"{where} comes with a neighbour list searched at cutoff "
            f"{neighbors.cutoff} Å; the model's cutoff is {cutoff} Å. Search it "
            f"again, or use the dataset prepared for this cutoff."
        )
    try:
        edge_index, unit_shifts = checked_edges(
            neighbors.edge_index,
            neighbors.unit_shifts,
            n_atoms,
            pbc,
            origin="its neighbour list",
        )
    except ValueError as error:
        raise SampleError(f"{where}: {error}") from None
    self_pairs = (edge_index[0] == edge_index[1]) & ~unit_shifts.any(axis=1)
    if self_pairs.any():
        raise SampleError(
            f"{where} comes with a neighbour list holding an atom's edge to "
            f"itself with a zero shift, which a search never returns. Search it "
            f"again."
        )
    return {
        "edge_index": edge_index,
        # Recomputed rather than read, so a list stored against another cell
        # cannot carry its displacements into this one.
        "shifts": unit_shifts @ cell,
        "unit_shifts": unit_shifts.astype(np.float64),
    }


def _input_value(
    name: str, spec: FieldSpec, value: Any, n_atoms: int, where: str
) -> np.ndarray:
    shape = spec.sample_shape(n_atoms)
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
            f"model declares. A feature cannot be made up; add it to the file."
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
    shape = spec.sample_shape(n_atoms)
    if array.shape != shape:
        raise SampleError(
            f"{where} has embedding feature {name!r} of shape {array.shape}; the "
            f"layout declares {shape}. It is not reshaped."
        )
    if not np.isfinite(array).all():
        raise SampleError(
            f"{where} has a non-finite value in embedding feature {name!r}."
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
    targets: dict[str, Any] = {STRUCTURE_WEIGHT: np.float64(config.weight)}
    fields = layout.target_fields()
    degenerate = any(pbc) and np.linalg.det(cell) == 0
    for name, spec in layout.targets.items():
        shape = spec.sample_shape(n_atoms)
        present = _labelled(config, name, where)
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
                    f"the cell three independent rows, a vacuum row along an "
                    f"aperiodic axis, or drop the label."
                )
        else:
            array = np.zeros(shape, dtype=np.float64)
        targets[name] = array
        targets[presence_key(name)] = np.full(
            fields[presence_key(name)].sample_shape(n_atoms), present, dtype=bool
        )
        weight = float(config.property_weights.get(name, 1.0)) if present else 0.0
        targets[weight_key(name)] = np.full(
            fields[weight_key(name)].sample_shape(n_atoms), weight, dtype=np.float64
        )
    return targets
