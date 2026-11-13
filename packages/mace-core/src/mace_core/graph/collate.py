"""Batching graph samples into one flat graph dict: the reference, in numpy.

A **sample** is one structure as :func:`~mace_core.graph.build.graph_sample`
produces it: node and edge keys as arrays over that structure's own atoms and
edges, with ``edge_index`` local to it, and graph keys as one value without a
leading graph dimension (``cell`` is ``[3, 3]``, ``head`` a scalar).

:func:`collate` is the specification each framework implements:

* node and edge keys are concatenated; ``edge_index`` is offset by the number
  of nodes before its structure;
* graph keys are stacked on a new leading dimension;
* ``batch`` repeats each graph index once per node and ``ptr`` is the
  cumulative node count, starting at 0;
* the sizes go to the :class:`~mace_core.graph.info.GraphInfo` returned beside
  the dict, never into it.

Every key is read by the level its layout declares: a sample carrying a key
the layout does not declare is refused, so nothing guesses a shape. The edge
keys are all present in every sample, or absent from all of them while a
device search has still to join the edges. :func:`collate_targets` batches the
targets the same way.

The inputs are never modified.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np

from mace_core.graph.info import GraphInfo
from mace_core.graph.layout import GraphLayout
from mace_core.graph.schema import (
    EDGE_LEVEL,
    FLOAT,
    NODE_LEVEL,
    FieldSpec,
)
from mace_core.graph.validation import check_batch_invariants, check_layout

__all__ = [
    "COLLATED_KEYS",
    "collate",
    "collate_targets",
    "sample_keys",
]

#: Keys a sample must not carry, because collation writes them.
COLLATED_KEYS: frozenset[str] = frozenset({"batch", "ptr"})


def sample_keys(
    samples: Sequence[Mapping[str, Any]],
    fields: Mapping[str, FieldSpec],
    built: Collection[str] = (),
) -> list[str]:
    """The keys every sample carries, each one the layout declares.

    Args:
        samples: The samples.
        fields: What the layout declares for them.
        built: Declared keys that are built after collation, which no sample
            may carry: the batch-requirement fields.

    Raises:
        ValueError: if there are no samples, if two carry different keys, or
            if a sample carries a key collation writes, a key built after
            collation, or a key the layout does not declare.
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
    for key in keys:
        if key in COLLATED_KEYS:
            raise ValueError(
                f"a sample carries {key!r}, which collation computes. Remove it "
                f"from the sample."
            )
        if key in built:
            raise ValueError(
                f"a sample carries {key!r}, a batch-requirement field, which is "
                f"built on the collated batch. Remove it from the sample."
            )
        if key not in fields:
            raise ValueError(
                f"a sample carries {key!r}, which the layout does not declare. "
                f"Declare it, or leave it out of the sample."
            )
    return keys


def _join(values: list[np.ndarray], spec: FieldSpec) -> np.ndarray:
    if spec.level() in (NODE_LEVEL, EDGE_LEVEL):
        return np.concatenate(values, axis=0)
    return np.stack(values)


def _numpy_float(layout: GraphLayout) -> np.dtype:
    try:
        return np.dtype(layout.float_dtype)
    except TypeError:
        raise ValueError(
            f"numpy has no {layout.float_dtype}. Collate with the framework's "
            f"collate for this layout."
        ) from None


def collate(
    samples: Sequence[Mapping[str, Any]],
    layout: GraphLayout,
    *,
    check: bool = True,
    validate: bool = False,
) -> tuple[dict[str, np.ndarray], GraphInfo]:
    """Batch graph samples into one graph dict and its :class:`GraphInfo`.

    Args:
        samples: One graph per structure, all with the same keys. Floating
            fields are cast to the layout's floating dtype.
        layout: What the run batches.
        check: Run :func:`check_layout` on the batch. A caller that goes on
            to extend the batch (padding) turns it off and checks the result
            once.
        validate: Also run :func:`check_batch_invariants`, which reads the
            index arrays.

    Raises:
        ValueError: as :func:`sample_keys` does.
    """
    fields = layout.graph_fields()
    keys = sample_keys(samples, fields, layout.requirement_fields)
    float_dtype = _numpy_float(layout)
    node_counts = [int(np.shape(sample["positions"])[0]) for sample in samples]
    node_offsets = np.cumsum([0, *node_counts[:-1]])
    graph: dict[str, np.ndarray] = {}
    for key in keys:
        values = [np.asarray(sample[key]) for sample in samples]
        if key == "edge_index":
            graph[key] = np.concatenate(
                [v + offset for v, offset in zip(values, node_offsets, strict=True)],
                axis=1,
            )
        else:
            graph[key] = _join(values, fields[key])
            if fields[key].dtype == FLOAT:
                graph[key] = graph[key].astype(float_dtype, copy=False)
    graph["batch"] = np.repeat(np.arange(len(samples), dtype=np.int64), node_counts)
    graph["ptr"] = np.asarray([0, *np.cumsum(node_counts).tolist()], dtype=np.int64)
    n_edges = int(graph["edge_index"].shape[1]) if "edge_index" in graph else 0
    info = GraphInfo(
        num_graphs=len(samples),
        num_nodes=sum(node_counts),
        num_edges=n_edges,
        structure_sizes=tuple(node_counts),
    )
    if check:
        check_layout(graph, layout, info)
    if validate:
        check_batch_invariants(graph, info)
    return graph, info


def collate_targets(
    samples: Sequence[Mapping[str, Any]], layout: GraphLayout
) -> dict[str, np.ndarray]:
    """Batch the targets of the same structures, in the same order, floats in
    the layout's floating dtype.

    Raises:
        ValueError: if there are no samples, if two carry different keys, or
            if a sample carries a key the layout's targets do not declare.
    """
    fields = layout.target_fields()
    keys = sample_keys(samples, fields)
    float_dtype = _numpy_float(layout)
    targets = {}
    for key in keys:
        joined = _join([np.asarray(sample[key]) for sample in samples], fields[key])
        if fields[key].dtype == FLOAT:
            joined = joined.astype(float_dtype, copy=False)
        targets[key] = joined
    return targets
