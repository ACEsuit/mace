"""Graph samples to a batched torch graph dict, with no torch_geometric.

:func:`collate` implements :func:`mace_core.graph.collate` in torch and binds
the floating dtype: every floating array becomes ``dtype``, indices stay
``int64`` and flags ``bool``. ``ptr``, ``batch`` and ``shifts`` are bit for bit
what the frozen legacy collater produces on the same structures, and ``cell``
is the parsed cells stacked as ``[n_graphs, 3, 3]`` rather than concatenated as
``[3 * n_graphs, 3]``. The sizes come back beside the dict, in a
:class:`~mace_core.graph.GraphInfo`.

:func:`collate_targets` batches the targets of the same structures, and
:func:`build_batch` is the whole path from parsed structures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from mace_core.data.configuration import Configuration
from mace_core.graph import (
    GraphInfo,
    GraphLayout,
    check_batch_invariants,
    check_layout,
    graph_sample,
    structure_graph,
)
from mace_core.graph.collate import sample_keys
from mace_core.graph.schema import EDGE_LEVEL, GRAPH_LEVEL, NODE_LEVEL, level_of
from mace_core.neighbors import NeighborListBackend

__all__ = ["build_batch", "collate", "collate_targets", "dtype_name"]


def dtype_name(dtype: torch.dtype) -> str:
    """The schema's name for a torch dtype: ``torch.float64`` is ``float64``."""
    return str(dtype).removeprefix("torch.")


#: Where a batch is assembled and checked, named on every tensor created
#: here so that a process default device set with ``torch.set_default_device``
#: cannot move part of it.
_HOST = torch.device("cpu")


def _as_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """A host tensor; floating values in ``dtype``."""
    tensor = torch.as_tensor(value, device=_HOST)
    if tensor.is_floating_point():
        tensor = tensor.to(dtype)
    return tensor


def _join(tensors: list[torch.Tensor], level: str | None) -> torch.Tensor:
    if level in (NODE_LEVEL, EDGE_LEVEL):
        return torch.cat(tensors, dim=0)
    return torch.stack(tensors)


def _to(batch: dict[str, torch.Tensor], device: torch.device | str | None) -> dict:
    if device is None:
        return batch
    return {key: value.to(device) for key, value in batch.items()}


def collate(
    samples: Sequence[Mapping[str, Any]],
    layout: GraphLayout,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    validate: bool = False,
) -> tuple[dict[str, torch.Tensor], GraphInfo]:
    """Batch graph samples into one torch graph dict and its ``GraphInfo``.

    Args:
        samples: One graph per structure, as
            :func:`~mace_core.graph.graph_sample` returns it; arrays may be
            numpy or torch, on any device. Not modified.
        layout: What the run batches.
        dtype: The model's floating dtype. Given explicitly rather than read
            from ``torch.get_default_dtype()``, so a process default cannot
            change the batch.
        device: Where the tensors go. The batch is assembled and checked on
            the host first; ``None`` leaves it there.
        validate: Also run :func:`~mace_core.graph.check_batch_invariants`,
            which reads the index arrays.

    Raises:
        ValueError: as :func:`mace_core.graph.collate` does.
    """
    fields = layout.graph_fields()
    keys = sample_keys(samples, fields)
    node_counts = [int(sample["positions"].shape[0]) for sample in samples]
    batch: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [_as_tensor(sample[key], dtype) for sample in samples]
        if key == "edge_index":
            offset, shifted = 0, []
            for tensor, count in zip(tensors, node_counts, strict=True):
                shifted.append(tensor + offset)
                offset += count
            batch[key] = torch.cat(shifted, dim=1)
        else:
            batch[key] = _join(tensors, level_of(fields[key]))
    counts = torch.tensor(node_counts, dtype=torch.int64, device=_HOST)
    graph_ids = torch.arange(len(samples), device=_HOST)
    batch["batch"] = torch.repeat_interleave(graph_ids, counts)
    batch["ptr"] = torch.tensor(
        [0, *np.cumsum(node_counts).tolist()], dtype=torch.int64, device=_HOST
    )
    n_edges = int(batch["edge_index"].shape[1]) if "edge_index" in batch else 0
    info = GraphInfo(
        num_graphs=len(samples),
        num_nodes=sum(node_counts),
        num_edges=n_edges,
        structure_sizes=tuple(node_counts),
    )
    check_layout(batch, layout, info, float_dtype=dtype_name(dtype))
    if validate:
        check_batch_invariants(batch, info)
    return _to(batch, device), info


def collate_targets(
    samples: Sequence[Mapping[str, Any]],
    layout: GraphLayout,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Batch the targets of the same structures, in the same order.

    Raises:
        ValueError: as :func:`mace_core.graph.collate_targets` does.
    """
    fields = layout.target_fields()
    keys = sample_keys(samples, fields)
    batch = {
        key: _join(
            [_as_tensor(sample[key], dtype) for sample in samples],
            level_of(fields[key]) or GRAPH_LEVEL,
        )
        for key in keys
    }
    return _to(batch, device)


def build_batch(
    configurations: Sequence[Configuration],
    layout: GraphLayout,
    *,
    neighbor_backend: NeighborListBackend,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], GraphInfo]:
    """Parsed structures to one batched graph dict, its targets and its info.

    ``neighbor_backend`` searches every structure at the layout's cutoff. The
    caller resolves it once, with
    :func:`~mace_torch.neighbors.resolve_neighbor_backend` from its
    ``neighbor_list`` section, and passes the same object to every batch.

    Raises:
        SampleError: as :func:`~mace_core.graph.graph_sample` does.
        ValueError: as :func:`collate` does.
    """
    pairs = [
        graph_sample(
            configuration,
            structure_graph(configuration, layout.cutoff, neighbor_backend),
            layout,
        )
        for configuration in configurations
    ]
    graph, info = collate(
        [graph for graph, _ in pairs], layout, dtype=dtype, device=device
    )
    targets = collate_targets(
        [targets for _, targets in pairs], layout, dtype=dtype, device=device
    )
    return graph, targets, info
