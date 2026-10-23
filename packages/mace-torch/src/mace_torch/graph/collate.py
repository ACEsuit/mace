"""Graph samples to a batched torch graph dict, with no torch_geometric.

:func:`collate` implements :func:`mace_core.graph.collate` in torch and binds
the schema's floating dtype: every floating array becomes ``dtype``, indices
stay ``int64`` and flags ``bool``. ``ptr``, ``batch`` and ``shifts`` are bit
for bit what the frozen legacy collater produces on the same structures, and
``cell`` is the same values stacked as ``[n_graphs, 3, 3]`` rather than
concatenated as ``[3 * n_graphs, 3]``.

:func:`build_batch` is the whole path from parsed structures: it resolves the
neighbour-list backend once from its config section, builds one sample per
structure, and collates them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from mace_core.data.configuration import DEFAULT_HEAD, Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.elements.default_keys import Storage
from mace_core.graph import graph_sample, validate_graph
from mace_core.graph.collate import batching_levels, sample_keys
from mace_core.graph.schema import EDGE_LEVEL, NODE_LEVEL
from mace_core.neighbors import NeighborListConfig, get_neighbor_backend

__all__ = ["build_batch", "collate", "derived_cell_quantities", "dtype_name"]


def dtype_name(dtype: torch.dtype) -> str:
    """The schema's name for a torch dtype: ``torch.float64`` is ``float64``."""
    return str(dtype).removeprefix("torch.")


#: Where a batch is assembled and validated, named on every tensor created
#: here so that a process default device set with ``torch.set_default_device``
#: cannot move part of it.
_HOST = torch.device("cpu")


def _as_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """A host tensor; floating values in ``dtype``."""
    tensor = torch.as_tensor(value, device=_HOST)
    if tensor.is_floating_point():
        tensor = tensor.to(dtype)
    return tensor


def collate(
    samples: Sequence[Mapping[str, Any]],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    features: Mapping[str, Storage] | None = None,
) -> dict[str, Any]:
    """Batch graph samples into one torch graph dict.

    Args:
        samples: One mapping per structure, as
            :func:`~mace_core.graph.graph_sample` returns; arrays may be numpy
            or torch, on any device. Not modified.
        dtype: The model's floating dtype. Given explicitly rather than read
            from ``torch.get_default_dtype()``, so a process default cannot
            change the batch.
        device: Where the tensors go. The batch is assembled and validated on
            the host first; ``None`` leaves it there.
        features: The declared embedding features, ``atom`` or ``graph``.

    Returns:
        The graph dict. It has passed
        :func:`~mace_core.graph.validate_graph`.

    Raises:
        ValueError: as :func:`mace_core.graph.collate` does.
    """
    keys = sample_keys(samples)
    levels = batching_levels(keys, features)

    node_counts = [int(sample["positions"].shape[0]) for sample in samples]
    batched: dict[str, Any] = {}
    for key in keys:
        level = levels[key]
        if level is None:
            batched[key] = tuple(sample[key] for sample in samples)
            continue
        tensors = [_as_tensor(sample[key], dtype) for sample in samples]
        if key == "edge_index":
            offset, shifted = 0, []
            for tensor, count in zip(tensors, node_counts, strict=True):
                shifted.append(tensor + offset)
                offset += count
            batched[key] = torch.cat(shifted, dim=1)
        elif level in (NODE_LEVEL, EDGE_LEVEL):
            batched[key] = torch.cat(tensors, dim=0)
        else:
            batched[key] = torch.stack(tensors)

    counts = torch.tensor(node_counts, dtype=torch.int64, device=_HOST)
    graph_ids = torch.arange(len(samples), device=_HOST)
    batched["batch"] = torch.repeat_interleave(graph_ids, counts)
    batched["ptr"] = torch.tensor(
        [0, *np.cumsum(node_counts).tolist()], dtype=torch.int64, device=_HOST
    )
    batched["rcell"], batched["volume"] = derived_cell_quantities(batched["cell"])
    batched["structure_sizes"] = tuple(node_counts)
    batched["structure_periodic"] = tuple(bool(row.any()) for row in batched["pbc"])
    batched["structure_padding"] = (False,) * len(samples)
    validate_graph(batched, float_dtype=dtype_name(dtype))
    if device is not None:
        batched = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batched.items()
        }
    return batched


def derived_cell_quantities(cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rcell, volume)``: ``2π · inv(cellᵀ)``, zero where ``det(cell) == 0``,
    and ``det(cell)``."""
    volume = torch.linalg.det(cell)
    nonzero = (torch.abs(volume) > 0)[:, None, None]
    identity = torch.eye(3, dtype=cell.dtype, device=cell.device).expand_as(cell)
    rcell = 2 * torch.pi * torch.linalg.inv(torch.where(nonzero, cell, identity).mT)
    return torch.where(nonzero, rcell, torch.zeros_like(rcell)), volume


def build_batch(
    configurations: Sequence[Configuration],
    *,
    atomic_number_table: AtomicNumberTable,
    cutoff: float,
    heads: Sequence[str] = (DEFAULT_HEAD,),
    neighbor_list: NeighborListConfig | None = None,
    features: Mapping[str, Storage] | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Parsed structures to one batched graph dict.

    The neighbour-list backend named by ``neighbor_list`` is resolved once,
    here, and used for every structure.

    Raises:
        ValueError: as :func:`~mace_core.graph.graph_sample` and
            :func:`collate` do.
        NeighborBackendNotAvailableError: if the configured backend cannot be
            built.
    """
    backend = get_neighbor_backend((neighbor_list or NeighborListConfig()).kind)
    samples = [
        graph_sample(
            configuration,
            atomic_number_table=atomic_number_table,
            cutoff=cutoff,
            heads=heads,
            features=features,
            neighbor_backend=backend,
        )
        for configuration in configurations
    ]
    return collate(samples, dtype=dtype, device=device, features=features)
