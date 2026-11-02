"""The neighbour-list backend this framework builds graphs with by default.

A ``neighbor_list`` config section that names no backend gets vesin, the
fastest backend on the CPU. This package requires vesin and registers its
backend (:mod:`mace_torch.neighbors.vesin_backend`), so the default is
installed wherever this package is. Any other backend is used only when the
section names it.
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
