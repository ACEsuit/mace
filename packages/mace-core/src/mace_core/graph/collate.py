"""Batching graph samples into one flat graph dict: the reference, in numpy.

A **sample** is one structure as :func:`~mace_core.graph.build.graph_sample`
produces it: node and edge keys as arrays over that structure's own atoms and
edges, with ``edge_index`` local to it, and graph keys as one value without a
leading graph dimension (``cell`` is ``[3, 3]``, ``energy`` a scalar).

:func:`collate` is the specification each framework implements:

* node and edge keys are concatenated; ``edge_index`` is offset by the number
  of nodes before its structure;
* graph keys are stacked on a new leading dimension;
* ``batch`` repeats each graph index once per node and ``ptr`` is the
  cumulative node count, starting at 0;
* ``rcell`` and ``volume`` are derived from the stacked effective cell:
  ``volume = det(cell)`` and ``rcell = 2π · inv(cellᵀ)``, exactly zero where
  the volume is zero;
* the host tuples ``structure_sizes``, ``structure_periodic`` and
  ``structure_padding`` are built from the samples.

A declared feature is batched by the storage its declaration names. A key that
is neither a schema key nor declared is carried as a tuple of the samples'
values, untouched: nothing guesses its shape.

The inputs are never modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mace_core.elements.default_keys import STORAGES, Storage
from mace_core.graph.schema import (
    EDGE_LEVEL,
    GRAPH_LEVEL,
    GRAPH_SCHEMA,
    NODE_LEVEL,
    NOT_KEYS,
    PRESENCE_SUFFIX,
    level_of,
)
from mace_core.graph.validation import validate_graph

__all__ = [
    "COLLATED_KEYS",
    "batching_levels",
    "collate",
    "derived_cell_quantities",
    "feature_levels",
    "sample_keys",
]

#: Keys a sample must not carry, because collation writes them.
COLLATED_KEYS: frozenset[str] = frozenset(
    {key for key, spec in GRAPH_SCHEMA.items() if spec.producer == "coll"}
    | {"rcell", "volume"}
)


def feature_levels(features: Mapping[str, Storage] | None) -> dict[str, str]:
    """The batching dimension of each declared feature: ``n_nodes`` for one
    stored per ``atom``, ``n_graphs`` per ``graph``.

    Raises:
        ValueError: for a storage other than ``atom`` or ``graph``, so a
            misspelling such as ``"atoms"`` is refused rather than read as
            per-structure; and for a feature named like a key of the graph
            schema, a derived name, or a presence flag. A schema key is read
            by its own row, and a feature cannot stand in for it.
    """
    levels: dict[str, str] = {}
    for name, storage in (features or {}).items():
        if name in GRAPH_SCHEMA or name in NOT_KEYS or name.endswith(PRESENCE_SUFFIX):
            raise ValueError(
                f"feature {name!r} is named like a key of the graph schema. A "
                f"schema key is read by its own row; give the feature a name of "
                f"its own."
            )
        if storage not in STORAGES:
            raise ValueError(
                f"feature {name!r} is declared per {storage!r}; use 'atom' or 'graph'."
            )
        levels[name] = NODE_LEVEL if storage == "atom" else GRAPH_LEVEL
    return levels


def sample_keys(samples: Sequence[Mapping[str, Any]]) -> list[str]:
    """The keys every sample carries.

    Raises:
        ValueError: if there are no samples, or if two carry different keys.
    """
    if not samples:
        raise ValueError("collate needs at least one sample.")
    keys = list(samples[0])
    for index, sample in enumerate(samples[1:], start=1):
        if set(sample) != set(keys):
            different = sorted(set(sample).symmetric_difference(keys))
            raise ValueError(
                f"sample {index} and sample 0 carry different keys: {different}. "
                f"Every structure of a batch carries the same keys."
            )
    return keys


def batching_levels(
    keys: Sequence[str], features: Mapping[str, Storage] | None
) -> dict[str, str | None]:
    """For each key, the dimension it is batched along: ``n_nodes``,
    ``n_edges``, ``n_graphs``, or ``None`` for a key carried as a tuple.

    Raises:
        ValueError: if a sample carries a key collation writes, or as
            :func:`feature_levels` does.
    """
    declared = feature_levels(features)
    levels: dict[str, str | None] = {}
    for key in keys:
        if key in COLLATED_KEYS:
            raise ValueError(
                f"a sample carries {key!r}, which collation computes. Remove it "
                f"from the sample."
            )
        if key in GRAPH_SCHEMA:
            levels[key] = level_of(GRAPH_SCHEMA[key])
        else:
            levels[key] = declared.get(key)
    return levels


def derived_cell_quantities(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rcell, volume)`` for a stack of effective cells ``[n_graphs, 3, 3]``."""
    volume = np.linalg.det(cell)
    nonzero = np.abs(volume) > 0
    invertible = np.where(nonzero[:, None, None], cell, np.eye(3, dtype=cell.dtype))
    rcell = 2 * np.pi * np.linalg.inv(np.swapaxes(invertible, -1, -2))
    rcell = np.where(nonzero[:, None, None], rcell, np.zeros_like(rcell))
    return rcell, volume


def collate(
    samples: Sequence[Mapping[str, Any]],
    *,
    features: Mapping[str, Storage] | None = None,
    float_dtype: str = "float64",
) -> dict[str, Any]:
    """Batch ``samples`` into one graph dict that passes
    :func:`~mace_core.graph.validation.validate_graph`.

    Args:
        samples: One mapping per structure, all with the same keys.
        features: The declared embedding features and where each is stored,
            ``atom`` or ``graph``.
        float_dtype: The floating dtype the samples carry, for validation.

    Raises:
        ValueError: if there are no samples, if they do not all carry the same
            keys, if a sample carries a key collation writes, or if a feature
            is declared wrongly (see :func:`feature_levels`).
    """
    keys = sample_keys(samples)
    levels = batching_levels(keys, features)

    node_counts = [int(np.shape(sample["positions"])[0]) for sample in samples]
    node_offsets = np.cumsum([0, *node_counts[:-1]])
    batched: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        level = levels[key]
        if level is None:
            batched[key] = tuple(values)
        elif key == "edge_index":
            batched[key] = np.concatenate(
                [
                    np.asarray(v) + offset
                    for v, offset in zip(values, node_offsets, strict=True)
                ],
                axis=1,
            )
        elif level in (NODE_LEVEL, EDGE_LEVEL):
            batched[key] = np.concatenate([np.asarray(v) for v in values], axis=0)
        else:
            batched[key] = np.stack([np.asarray(v) for v in values])

    batched["batch"] = np.repeat(np.arange(len(samples), dtype=np.int64), node_counts)
    batched["ptr"] = np.asarray([0, *np.cumsum(node_counts).tolist()], dtype=np.int64)
    batched["rcell"], batched["volume"] = derived_cell_quantities(batched["cell"])
    batched["structure_sizes"] = tuple(node_counts)
    batched["structure_periodic"] = tuple(bool(row.any()) for row in batched["pbc"])
    batched["structure_padding"] = (False,) * len(samples)
    validate_graph(batched, float_dtype=float_dtype)
    return batched
