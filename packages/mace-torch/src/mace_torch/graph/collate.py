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
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from mace_core.data.configuration import DEFAULT_HEAD, Configuration
from mace_core.edges import plan_edges
from mace_core.elements import AtomicNumberTable
from mace_core.elements.default_keys import Storage
from mace_core.graph import graph_sample, validate_graph
from mace_core.graph.collate import batching_levels, sample_keys
from mace_core.graph.schema import DERIVED_KEYS, EDGE_LEVEL, NODE_LEVEL
from mace_core.neighbors import (
    NeighborList,
    NeighborListBackend,
    get_neighborhoods,
    search_geometry,
)

from mace_torch.graph.edges import apply_edge_plan
from mace_torch.neighbors.providers import ResolvedNeighborProvider

__all__ = [
    "build_batch",
    "collate",
    "derived_cell_quantities",
    "dtype_name",
    "join_neighborhoods",
]


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
    neighborhoods: Sequence[NeighborList[torch.Tensor]] | None = None,
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
        neighborhoods: Lists already searched on ``device``, one per sample.
            Samples must omit edges. Only the host part is validated before
            transfer; the lists are joined on device without a host copy.

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
    if neighborhoods is not None:
        if any(key in batched for key in ("edge_index", "shifts", "unit_shifts")):
            raise ValueError(
                "device neighborhoods require samples without edges; "
                "use graph_sample(include_edges=False)."
            )
        batched.update(
            edge_index=torch.empty((2, 0), dtype=torch.int64, device=_HOST),
            shifts=torch.empty((0, 3), dtype=dtype, device=_HOST),
            unit_shifts=torch.empty((0, 3), dtype=dtype, device=_HOST),
        )
    validate_graph(batched, float_dtype=dtype_name(dtype))
    if device is not None:
        batched = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batched.items()
        }
    if neighborhoods is not None:
        batched = join_neighborhoods(batched, neighborhoods)
    return batched


def join_neighborhoods(
    data: Mapping[str, Any], neighborhoods: Sequence[NeighborList[torch.Tensor]]
) -> dict[str, Any]:
    """Join device-resident edge lists to collated host data before padding.

    Offsets come from host structure sizes; edge tensors are never read back
    to the host. The effective cells must match the host graph's cells.
    """
    sizes = data["structure_sizes"]
    if DERIVED_KEYS & data.keys() or (
        "edge_index" in data and data["edge_index"].shape[1]
    ):
        raise ValueError(
            "join_neighborhoods needs host data without edges or derived runs."
        )
    if len(neighborhoods) != len(sizes) or any(data["structure_padding"]):
        raise ValueError("join_neighborhoods needs one list per unpadded structure.")
    positions = data["positions"]
    shifted, shifts, unit_shifts = [], [], []
    offset = 0
    for index, (neighbors, count) in enumerate(zip(neighborhoods, sizes, strict=True)):
        edges = neighbors.edge_index
        if (
            edges.device != positions.device
            or edges.dtype != torch.int64
            or edges.ndim != 2
            or edges.shape[0] != 2
        ):
            raise ValueError(
                "neighbor edge_index must be [2, n_edges] int64 on the graph device."
            )
        if torch.any(edges < 0) or torch.any(edges >= count):
            raise ValueError(
                f"neighborhood {index} has an edge outside its {count} nodes."
            )
        for key in ("shifts", "unit_shifts"):
            value = getattr(neighbors, key)
            if value.device != positions.device or value.shape != (edges.shape[1], 3):
                raise ValueError(
                    f"neighborhood {index}: {key} must be [n_edges, 3] "
                    "on the graph device."
                )
        if not torch.equal(neighbors.cell.to(positions), data["cell"][index]):
            raise ValueError(
                f"neighborhood {index} has a different effective cell "
                "from the host graph."
            )
        shifted.append(edges + offset)
        shifts.append(neighbors.shifts.to(dtype=positions.dtype))
        unit_shifts.append(neighbors.unit_shifts.to(dtype=positions.dtype))
        offset += count
    return {
        **data,
        "edge_index": torch.cat(shifted, dim=1),
        "shifts": torch.cat(shifts),
        "unit_shifts": torch.cat(unit_shifts),
    }


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
    neighbor_backend: NeighborListBackend | ResolvedNeighborProvider,
    heads: Sequence[str] = (DEFAULT_HEAD,),
    features: Mapping[str, Storage] | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Parsed structures to one batched graph dict.

    ``neighbor_backend`` searches every structure. The caller resolves it once,
    with :func:`~mace_torch.neighbors.resolve_neighbor_backend` from its
    ``neighbor_list`` section, and passes the same object to every batch.

    Raises:
        ValueError: as :func:`~mace_core.graph.graph_sample` and
            :func:`collate` do.
    """
    if isinstance(neighbor_backend, ResolvedNeighborProvider):
        provider = neighbor_backend
        if device is not None:
            requested = torch.device(device)
            if requested.type != provider.device.type or (
                requested.index is not None and requested.index != provider.device.index
            ):
                raise ValueError(
                    f"batch device {requested} differs from provider "
                    f"device {provider.device}."
                )
        geometries = [
            search_geometry(c.positions, cutoff, c.pbc, c.cell) for c in configurations
        ]
        points = [
            torch.tensor(c.positions, dtype=torch.float64, device=provider.device)
            for c in configurations
        ]
        neighborhoods = provider.search_batch(points, cutoff, geometries)
        # The effective cell is already known on the host. Reuse it without
        # copying any part of a device neighbourhood back from the accelerator.
        samples = [
            graph_sample(
                c,
                atomic_number_table=atomic_number_table,
                cutoff=cutoff,
                heads=heads,
                features=features,
                neighborhood=replace(n, cell=g.cell),
                include_edges=False,
            )
            for c, n, g in zip(configurations, neighborhoods, geometries, strict=True)
        ]
        graph = collate(
            samples,
            dtype=dtype,
            device=provider.device,
            features=features,
            neighborhoods=neighborhoods,
        )
        guaranteed = provider.capabilities.features
        return apply_edge_plan(
            graph, plan_edges(guaranteed, guaranteed & {"sorted_by_sender"})
        )

    neighborhoods = get_neighborhoods(
        [c.positions for c in configurations],
        cutoff,
        [c.pbc for c in configurations],
        [c.cell for c in configurations],
        backend=neighbor_backend,
    )
    samples = [
        graph_sample(
            configuration,
            atomic_number_table=atomic_number_table,
            cutoff=cutoff,
            heads=heads,
            features=features,
            neighborhood=neighborhood,
        )
        for configuration, neighborhood in zip(
            configurations, neighborhoods, strict=True
        )
    ]
    return collate(samples, dtype=dtype, device=device, features=features)
