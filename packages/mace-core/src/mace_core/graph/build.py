"""From a :class:`~mace_core.data.configuration.Configuration` to a graph sample.

This is the one place a parsed structure meets the neighbour list and the
model's element and head tables. The sample it returns is numpy and is batched
by :func:`~mace_core.graph.collate.collate` or a framework's implementation of
it.

What a sample carries, besides the structure itself:

* ``head``: the index of :attr:`Configuration.head` in the model's ordered head
  list. A name outside the list is an error rather than a fallback to some
  other head;
* every label of the schema, with a ``<label>_present`` flag read from
  :meth:`Configuration.is_labelled` and ``<label>_weight`` from the property
  weights. A label the structure lacks, whether its key specification declared
  it or not, is written as zeros with the flag false and the weight zero; a
  present label with a zero weight is still present. Writing every label is
  what lets structures parsed with different key specifications, one per head,
  share a batch;
* every per-structure and per-atom input of the schema, with its default when
  the structure has no value: 1 for ``total_spin``, 0 for the others. The one
  exception is ``density_coefficients``, written only when declared, because
  its width is the structure's own;
* every declared embedding feature, by the storage it was declared with;
* any other property, untouched, to be carried as it is. Its shape is never
  guessed: an undeclared one-dimensional array is not taken to be per-atom.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mace_core.data.configuration import DEFAULT_HEAD, Configuration
from mace_core.elements.default_keys import Storage
from mace_core.elements.number_table import AtomicNumberTable
from mace_core.graph.collate import feature_levels
from mace_core.graph.schema import (
    GRAPH_LEVEL,
    GRAPH_SCHEMA,
    INPUT_KEYS,
    LABEL_KEYS,
    NODE_LEVEL,
    presence_key,
)
from mace_core.neighbors.backends import NeighborListBackend
from mace_core.neighbors.neighborhood import get_neighborhood

__all__ = ["graph_sample"]

#: Inputs whose absent value is not zero.
_INPUT_DEFAULTS: dict[str, float] = {"total_spin": 1.0}


def graph_sample(
    config: Configuration,
    *,
    atomic_number_table: AtomicNumberTable,
    cutoff: float,
    heads: Sequence[str] = (DEFAULT_HEAD,),
    features: Mapping[str, Storage] | None = None,
    neighbor_backend: NeighborListBackend | None = None,
) -> dict[str, Any]:
    """One structure as a graph sample.

    Args:
        config: The parsed structure. Not modified.
        atomic_number_table: The model's element table; ``node_attrs`` is the
            one-hot over it.
        cutoff: The neighbour cutoff, Angstrom.
        heads: The model's head names, in order.
        features: The declared embedding features, ``atom`` or ``graph``.
        neighbor_backend: The resolved neighbour-list backend; ``None`` uses
            matscipy.

    Returns:
        A dict of numpy arrays: floats as float64, indices as int64, flags as
        bool. Graph-level values carry no leading graph dimension.

    Raises:
        ValueError: if ``heads`` repeats a name, the head is not in ``heads``,
            an element is not in the table, a declared feature has no value or
            the wrong number of rows, a property has the wrong shape for its
            schema key or is named like a key the builder writes, or a feature
            is declared wrongly (see
            :func:`~mace_core.graph.collate.feature_levels`).
    """
    feature_level = feature_levels(features)
    repeated = sorted({head for head in heads if list(heads).count(head) > 1})
    if repeated:
        raise ValueError(
            f"the model's heads {list(heads)} repeat {repeated}; each head has "
            f"one index."
        )
    if config.head not in heads:
        raise ValueError(
            f"the structure belongs to head {config.head!r}, which is not one of "
            f"the model's heads {list(heads)}."
        )
    n_atoms = len(config)
    neighbors = get_neighborhood(
        config.positions,
        cutoff,
        pbc=config.pbc,
        cell=config.cell,
        backend=neighbor_backend,
    )
    element_indices = np.array(
        [atomic_number_table.z_to_index(z) for z in config.atomic_numbers],
        dtype=np.int64,
    )
    node_attrs = np.zeros((n_atoms, len(atomic_number_table)), dtype=np.float64)
    node_attrs[np.arange(n_atoms), element_indices] = 1.0

    sample: dict[str, Any] = {
        "positions": np.array(config.positions, dtype=np.float64),
        "node_attrs": node_attrs,
        "node_atomic_numbers": np.array(config.atomic_numbers, dtype=np.int64),
        "edge_index": np.array(neighbors.edge_index),
        "shifts": np.array(neighbors.shifts),
        "unit_shifts": neighbors.unit_shifts.astype(np.float64),
        "cell": np.array(neighbors.cell),
        "pbc": np.array(
            (False, False, False) if config.pbc is None else config.pbc, dtype=bool
        ),
        "head": np.int64(heads.index(config.head)),
        "weight": np.float64(config.weight),
    }
    for name in sorted(LABEL_KEYS):
        _add_label(sample, config, name, n_atoms)
    for name in sorted(INPUT_KEYS):
        if name in config.properties or name != "density_coefficients":
            sample[name] = _input_value(name, config.properties.get(name), n_atoms)
    for name, value in config.properties.items():
        if name in LABEL_KEYS or name in INPUT_KEYS:
            continue
        if name in GRAPH_SCHEMA:
            raise ValueError(
                f"the structure has a property named {name!r}, which is a graph "
                f"key the builder writes itself. Rename the property."
            )
        if name in feature_level:
            sample[name] = _feature_value(name, value, feature_level[name], n_atoms)
        else:
            sample[name] = value
    return sample


def _value_shape(name: str, n_atoms: int) -> tuple[int, ...]:
    """The shape one structure's value of a schema key has in a sample."""
    shape = GRAPH_SCHEMA[name].shape
    if shape[0] == GRAPH_LEVEL:
        return tuple(int(n) for n in shape[1:])
    return (n_atoms, *(int(n) for n in shape[1:]))


def _add_label(
    sample: dict[str, Any], config: Configuration, name: str, n_atoms: int
) -> None:
    shape = _value_shape(name, n_atoms)
    present = name in config.properties and config.is_labelled(name)
    if present:
        array = np.asarray(config.properties[name], dtype=np.float64)
        if array.shape != shape:
            raise ValueError(
                f"label {name!r} has shape {array.shape}; the graph stores {shape}."
            )
    else:
        array = np.zeros(shape, dtype=np.float64)
    sample[name] = array
    flag_shape = (n_atoms,) if GRAPH_SCHEMA[name].shape[0] == NODE_LEVEL else ()
    sample[presence_key(name)] = np.full(flag_shape, present, dtype=bool)
    weight_key = f"{name}_weight"
    if weight_key in GRAPH_SCHEMA:
        weight_shape = _value_shape(weight_key, n_atoms)
        weight = float(config.property_weights.get(name, 1.0)) if present else 0.0
        sample[weight_key] = np.full(weight_shape, weight, dtype=np.float64)


def _input_value(name: str, value: Any, n_atoms: int) -> np.ndarray:
    if name == "density_coefficients":
        if value is None:
            return np.zeros((n_atoms, 1), dtype=np.float64)
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != n_atoms:
            raise ValueError(
                f"density_coefficients has shape {array.shape}; the graph stores "
                f"[{n_atoms}, n_density_coefficients]."
            )
        return array
    shape = _value_shape(name, n_atoms)
    if value is None:
        return np.full(shape, _INPUT_DEFAULTS.get(name, 0.0), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name!r} has shape {array.shape}; the graph stores {shape}.")
    return array


def _feature_value(name: str, value: Any, level: str, n_atoms: int) -> np.ndarray:
    if value is None:
        raise ValueError(
            f"embedding feature {name!r} is declared but this structure has no "
            f"value for it. An input cannot be made up; add it to the file."
        )
    array = np.asarray(value)
    if level == NODE_LEVEL and (array.ndim == 0 or array.shape[0] != n_atoms):
        raise ValueError(
            f"feature {name!r} is declared per atom and has shape {array.shape}; "
            f"its first dimension must be the {n_atoms} atoms. It is not reshaped."
        )
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    return array
