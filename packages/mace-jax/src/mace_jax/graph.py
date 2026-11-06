"""Tensorize mace_core's natural graph batching contract as jax.Array."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
from ase import Atoms
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.elements.default_keys import Storage
from mace_core.graph import (
    GRAPH_SCHEMA,
    PaddingBudget,
    PaddingInfo,
    PaddingPolicy,
    collate,
    graph_sample,
    padding_budget,
    padding_length,
    unpad_outputs,
)
from mace_core.graph.collate import feature_levels
from mace_core.graph.schema import (
    DERIVED_KEYS,
    EDGE_LEVEL,
    GRAPH_LEVEL,
    NODE_LEVEL,
    level_of,
)

from mace_jax.neighbors import resolve_neighbor_backend

__all__ = ["graph_from_atoms", "pad_batch", "unpad_outputs"]


def graph_from_atoms(
    structures: Atoms | list[Atoms],
    *,
    elements: tuple[int, ...],
    cutoff: float,
    dtype: str = "float64",
    head: str = "default",
) -> dict[str, jnp.ndarray]:
    """One or more ASE structures -> unpadded flat graph, positions/cell in Å."""
    if isinstance(structures, Atoms):
        structures = [structures]
    samples = [
        graph_sample(
            Configuration(
                atomic_numbers=np.asarray(atoms.numbers),
                positions=np.asarray(atoms.positions),
                cell=np.asarray(atoms.cell),
                pbc=(bool(atoms.pbc[0]), bool(atoms.pbc[1]), bool(atoms.pbc[2])),
                head=head,
            ),
            atomic_number_table=AtomicNumberTable(elements),
            cutoff=cutoff,
            heads=(head,),
            neighbor_backend=resolve_neighbor_backend(),
        )
        for atoms in structures
    ]
    arrays = collate(samples)
    return {
        name: jnp.asarray(
            value, dtype=dtype if np.issubdtype(value.dtype, np.floating) else None
        )
        for name, value in arrays.items()
        if isinstance(value, np.ndarray)
    }


def pad_batch(
    data: Mapping[str, Any],
    policy: PaddingPolicy,
    r_max: float,
    *,
    atomic_number_table: AtomicNumberTable,
    features: Mapping[str, Storage] | None = None,
) -> tuple[dict[str, Any], PaddingInfo]:
    """Append inert DATA-3 slots and return the real counts separately."""
    nodes, edges = data["positions"].shape[0], data["edge_index"].shape[1]
    graphs = data["ptr"].shape[0] - 1
    info = PaddingInfo(nodes, edges, graphs)
    budget = padding_budget(policy, n_nodes=nodes, n_edges=edges, n_graphs=graphs)
    if budget is None or budget == PaddingBudget(nodes, edges, graphs):
        return dict(data), info

    # Matches legacy build_fake_padding_graph and the DATA-3 Torch construction.
    extra_graphs = budget.graphs - graphs
    extra_nodes = budget.nodes - nodes
    extra_edges = budget.edges - edges
    lengths = {
        GRAPH_LEVEL: extra_graphs,
        NODE_LEVEL: extra_nodes,
        EDGE_LEVEL: extra_edges,
    }
    length = padding_length(r_max)
    levels = feature_levels(features)
    padded = {}
    for key, value in data.items():
        spec = GRAPH_SCHEMA.get(key)
        if isinstance(value, tuple):
            if key == "structure_sizes":
                tail = (0,) * (extra_graphs - 1) + (extra_nodes,)
            else:
                fill = (
                    (spec.pad_value == "true") if spec and spec.kind == "host" else None
                )
                tail = (fill,) * extra_graphs
            padded[key] = value + tail
            continue
        axis = 1 if key == "edge_index" else 0
        if key == "ptr":
            rows = np.full(extra_graphs, nodes, dtype=value.dtype)
            rows[-1] = budget.nodes
        elif key in DERIVED_KEYS and key.endswith("_order"):
            rows = np.arange(edges, budget.edges, dtype=value.dtype)
        else:
            level = level_of(spec) if spec else levels.get(key)
            if level is None:
                raise ValueError(
                    f"{key!r} is neither a schema key nor a declared feature"
                )
            shape = list(value.shape)
            shape[axis] = lengths[level]
            rows = np.zeros(shape, dtype=value.dtype)
            fill = spec.pad_value if spec else "zero"
            if key in DERIVED_KEYS and extra_edges:
                rows[-1] = extra_edges
            elif fill in ("one", "true"):
                rows.fill(1)
            elif fill == "element_zero_atomic_number":
                rows.fill(atomic_number_table.index_to_z(0))
            elif fill in ("element_zero_one_hot", "unit_x", "padding_shift"):
                rows[:, 0] = length if fill == "padding_shift" else 1
            elif fill == "self_loop":
                rows.fill(budget.nodes - 1)
            elif fill == "padding_graph_id":
                rows.fill(budget.graphs - 1)
            elif fill in ("padding_cell", "padding_reciprocal_cell"):
                scale = length if fill == "padding_cell" else 2 * np.pi / length
                rows[:] = np.eye(3) * scale
            elif fill == "padding_volume":
                rows.fill(length**3)
            elif fill not in ("zero", "false"):
                raise ValueError(f"{key!r} has unknown padding rule {fill!r}")
        padded[key] = jnp.concatenate((value, rows), axis=axis)
    return padded, info
