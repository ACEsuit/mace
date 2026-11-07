"""Resolve host or torch neighbour providers and negotiate their edge plan.

Torch providers register in ``mace.neighbor_backends.torch``. They receive the
core's SearchGeometry and return a NeighborList of tensors on the positions'
device. Resolution and optional batched search happen outside model forwards.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from mace_core.edges import (
    EdgeFeature,
    EdgePlan,
    EdgeRequirementError,
    NeighborListCapabilities,
    plan_edges,
)
from mace_core.neighbors import (
    NeighborBackendNotAvailableError,
    NeighborList,
    NeighborListConfig,
    SearchGeometry,
    get_neighbor_backend,
    neighbor_capabilities,
)
from mace_core.neighbors.backends import NEIGHBOR_BACKEND_GROUP, _registered
from mace_core.neighbors.neighborhood import _neighborhood

logger = logging.getLogger(__name__)
DEFAULT_NEIGHBOR_BACKEND = "vesin"
CUDA_NEIGHBOR_BACKEND = "nvalchemiops"


class TorchNeighborListBackend(Protocol):
    """Device search at a strict float64 squared-distance cutoff, in Angstrom.

    ``cell`` is geometry.cell and ``shifts`` is unit_shifts @ cell. No zero
    shift self-pairs are returned. All four arrays stay on positions.device;
    indices/shifts are int64, Cartesian shifts/cell are float64.
    """

    def capabilities(self) -> NeighborListCapabilities: ...

    def search(
        self, positions: torch.Tensor, cutoff: float, geometry: SearchGeometry
    ) -> NeighborList[torch.Tensor]: ...


class BatchedTorchNeighborListBackend(Protocol):
    """Optional stateless search; each result has the order of a single search."""

    def search_batch(
        self,
        positions: Sequence[torch.Tensor],
        cutoff: float,
        geometries: Sequence[SearchGeometry],
    ) -> Sequence[NeighborList[torch.Tensor]]: ...


@dataclass(frozen=True)
class ResolvedNeighborProvider:
    """Build report and provider; hold one instance per system if stateful."""

    name: str
    backend: Any
    framework: str | None
    capabilities: NeighborListCapabilities
    edge_plan: EdgePlan
    device: torch.device

    def search_batch(
        self,
        positions: Sequence[torch.Tensor],
        cutoff: float,
        geometries: Sequence[SearchGeometry],
    ) -> list[NeighborList[torch.Tensor]]:
        """Search once per batch if supported; otherwise once per structure.

        For a stateful provider, only a one-system sequence is accepted. A
        torch provider's tensors are never converted to host arrays here.
        """
        if len(positions) != len(geometries):
            raise ValueError("positions and geometries must name the same structures.")
        if self.capabilities.stateful and len(positions) > 1:
            raise ValueError(
                f"provider {self.name!r} is stateful; keep one instance per system."
            )
        if self.framework == "torch":
            inputs = [
                value.to(device=self.device, dtype=torch.float64) for value in positions
            ]
        else:
            inputs = [
                value.detach().cpu().numpy().astype(np.float64, copy=True)
                for value in positions
            ]
            for value in inputs:
                value.setflags(write=False)
        batch_search = getattr(self.backend, "search_batch", None)
        results = (
            list(batch_search(inputs, cutoff, geometries))
            if callable(batch_search)
            else [
                self.backend.search(p, cutoff, g)
                for p, g in zip(inputs, geometries, strict=True)
            ]
        )
        if len(results) != len(inputs):
            raise ValueError(
                f"provider {self.name!r}: search_batch must return "
                "one result per structure."
            )
        neighborhoods = []
        for result, points, geometry in zip(results, inputs, geometries, strict=True):
            if self.framework != "torch":
                host = _neighborhood(result[0], result[1], len(points), geometry)
                result = NeighborList(
                    **{
                        key: torch.tensor(getattr(host, key), device=self.device)
                        for key in ("edge_index", "unit_shifts", "shifts", "cell")
                    }
                )
            else:
                _check_result(result, len(points), geometry, self.device, self.name)
            neighborhoods.append(result)
        return neighborhoods


def _check_result(
    result: NeighborList[torch.Tensor],
    n_nodes: int,
    geometry: SearchGeometry,
    device: torch.device,
    name: str,
) -> None:
    if not isinstance(result, NeighborList):
        raise ValueError(f"provider {name!r} must return NeighborList tensors.")
    arrays = {
        key: getattr(result, key)
        for key in ("edge_index", "unit_shifts", "shifts", "cell")
    }
    for key, value in arrays.items():
        if not isinstance(value, torch.Tensor) or value.device != device:
            raise ValueError(f"provider {name!r}: {key} must be a tensor on {device}.")
        dtype = torch.int64 if key in ("edge_index", "unit_shifts") else torch.float64
        if value.dtype != dtype:
            raise ValueError(f"provider {name!r}: {key} must have dtype {dtype}.")
    edges = result.edge_index
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"provider {name!r}: edge_index must have shape [2, n_edges].")
    if result.unit_shifts.shape != (edges.shape[1], 3) or result.shifts.shape != (
        edges.shape[1],
        3,
    ):
        raise ValueError(
            f"provider {name!r}: unit_shifts and shifts must have shape [n_edges, 3]."
        )
    if torch.any(edges < 0) or torch.any(edges >= n_nodes):
        raise ValueError(f"provider {name!r}: edge_index is outside [0, {n_nodes}).")
    cell = torch.tensor(geometry.cell, dtype=torch.float64, device=device)
    if not torch.equal(result.cell, cell):
        raise ValueError(f"provider {name!r}: cell must be SearchGeometry.cell.")
    if not torch.equal(result.shifts, result.unit_shifts.to(torch.float64) @ cell):
        raise ValueError(f"provider {name!r}: shifts must be unit_shifts @ cell.")
    if torch.any((edges[0] == edges[1]) & (result.unit_shifts == 0).all(dim=1)):
        raise ValueError(f"provider {name!r}: zero-shift self pairs must be removed.")


def resolve_neighbor_provider(
    config: NeighborListConfig | None = None,
    *,
    required_features: frozenset[EdgeFeature] = frozenset(),
    device: torch.device | str = "cpu",
    default: str | None = None,
) -> ResolvedNeighborProvider:
    """Resolve one name from both groups, with its plan and a once-per-build warning.

    ``required_features`` is the configured kernel backend's edge_features.
    Host providers search on CPU and their output is transferred once; torch
    providers must support the requested device. On NVIDIA CUDA, the registered
    ``nvalchemiops`` provider is preferred when available. Otherwise vesin is
    used. Explicit names and caller-provided defaults always resolve strictly.
    """
    config = config or NeighborListConfig()
    registered = _registered("torch")
    device = torch.device(device)
    name = config.kind or default
    backend = None
    if name is None:
        name = DEFAULT_NEIGHBOR_BACKEND
        if (
            device.type == "cuda"
            and not torch.version.hip
            and CUDA_NEIGHBOR_BACKEND in registered
        ):
            try:
                backend = get_neighbor_backend(CUDA_NEIGHBOR_BACKEND, framework="torch")
            except NeighborBackendNotAvailableError as error:
                logger.info(
                    "CUDA neighbour default unavailable, using vesin: %s", error
                )
            else:
                name = CUDA_NEIGHBOR_BACKEND
    if backend is None:
        backend = get_neighbor_backend(name, config.settings, framework="torch")
    group = registered[name].group
    framework = None if group == NEIGHBOR_BACKEND_GROUP else "torch"
    capabilities = neighbor_capabilities(backend, name)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if framework is not None and device.type not in capabilities.devices:
        raise ValueError(
            f"provider {name!r} does not support {device.type!r}; "
            f"devices are {sorted(capabilities.devices)}."
        )
    try:
        plan = plan_edges(required_features, capabilities.features)
    except EdgeRequirementError as error:
        raise EdgeRequirementError(f"neighbour provider {name!r}: {error}") from error
    if config.kind is not None and plan.adapters:
        logger.warning(
            "neighbour provider %r needs edge adapters %s",
            name,
            [step.name for step in plan.adapters],
        )
    return ResolvedNeighborProvider(
        name, backend, framework, capabilities, plan, device
    )
