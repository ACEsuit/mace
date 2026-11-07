"""The neighbour-list backend this framework builds graphs with by default.

Host graph builders use vesin. Device-aware builders prefer the registered
``nvalchemiops`` provider on NVIDIA CUDA when available, with vesin otherwise.
Explicit backend names are resolved strictly through the provider registry.
"""

from __future__ import annotations

from mace_core import neighbors
from mace_core.neighbors import NeighborListBackend, NeighborListConfig

from mace_torch.neighbors.providers import (
    DEFAULT_NEIGHBOR_BACKEND,
    BatchedTorchNeighborListBackend,
    ResolvedNeighborProvider,
    TorchNeighborListBackend,
    resolve_neighbor_provider,
)

__all__ = [
    "DEFAULT_NEIGHBOR_BACKEND",
    "BatchedTorchNeighborListBackend",
    "ResolvedNeighborProvider",
    "TorchNeighborListBackend",
    "resolve_neighbor_backend",
    "resolve_neighbor_provider",
]


def resolve_neighbor_backend(
    config: NeighborListConfig | None = None,
) -> NeighborListBackend:
    """The backend ``config`` names, or this framework's default.

    See :func:`mace_core.neighbors.resolve_neighbor_backend`.
    """
    return neighbors.resolve_neighbor_backend(config, DEFAULT_NEIGHBOR_BACKEND)
