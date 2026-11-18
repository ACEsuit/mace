"""Atomic descriptors reduced per structure.

The descriptors themselves are the backbone's node features, read with
:meth:`mace_torch.nn.backbone.MACEBackbone.atomic_descriptors`. This module
holds what a caller does with them afterwards, which no model computes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

__all__ = ["Aggregation", "aggregate_atomic_descriptors"]

#: How :func:`aggregate_atomic_descriptors` reduces per structure.
Aggregation = Literal["mean", "per_element_mean"] | None


def aggregate_atomic_descriptors(
    atomic_descriptors: Tensor,
    batch: Tensor,
    num_graphs: int,
    aggregation: Aggregation,
    element: Tensor | None = None,
    atomic_numbers: Sequence[int] = (),
) -> Tensor | list[dict[int, Tensor]]:
    """Reduce atomic descriptors per structure.

    Args:
        atomic_descriptors: ``[n_nodes, width]``, from
            :meth:`mace_torch.nn.backbone.MACEBackbone.atomic_descriptors`.
        batch: ``[n_nodes]``, int64, each node's structure.
        num_graphs: How many structures, ``ptr.numel() - 1``.
        aggregation: ``None`` returns the atomic descriptors as they are. ``"mean"``
            averages each structure's atoms, ``[num_graphs, width]``.
            ``"per_element_mean"`` averages each structure's atoms of each
            element present, one dict per structure keyed by atomic number.
        element: ``[n_nodes]``, int64, each node's column of the element
            table. Needed for ``"per_element_mean"`` only.
        atomic_numbers: The element table. Needed for ``"per_element_mean"``.
    """
    if aggregation is None:
        return atomic_descriptors
    if aggregation == "mean":
        totals = atomic_descriptors.new_zeros(num_graphs, atomic_descriptors.shape[1])
        totals = totals.index_add(0, batch, atomic_descriptors)
        counts = torch.bincount(batch, minlength=num_graphs).to(
            atomic_descriptors.dtype
        )
        return totals / counts.unsqueeze(-1)
    if aggregation == "per_element_mean":
        if element is None or not atomic_numbers:
            raise ValueError(
                "per_element_mean needs each node's element and the element table."
            )
        result: list[dict[int, Tensor]] = []
        for graph in range(num_graphs):
            in_graph = batch == graph
            means = {}
            for column, atomic_number in enumerate(atomic_numbers):
                mask = in_graph & (element == column)
                if bool(mask.any()):
                    means[int(atomic_number)] = atomic_descriptors[mask].mean(dim=0)
            result.append(means)
        return result
    raise ValueError(
        f"aggregation={aggregation!r} is not one this computes. The choices are "
        f"None, 'mean' and 'per_element_mean'."
    )
