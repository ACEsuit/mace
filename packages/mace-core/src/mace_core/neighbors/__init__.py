"""Neighbour lists: the edge set of a structure and the cell it is measured in.

:func:`get_neighborhood` and :func:`get_neighborhoods` normalize host search
results. The search itself is pluggable, selected by ``neighbor_list``
(:class:`NeighborListConfig`). Frameworks also register native providers;
the search box and returned cell always come from :func:`search_geometry`.

The host helpers use numpy. No framework is imported.
"""

from mace_core.neighbors.all_pairs import AllPairsBackend
from mace_core.neighbors.backends import (
    DEFAULT_NEIGHBOR_BACKEND,
    NEIGHBOR_BACKEND_GROUP,
    BatchedNeighborListBackend,
    DiscoveredNeighborBackend,
    DuplicateNeighborBackendError,
    NeighborBackendNotAvailableError,
    NeighborListBackend,
    NeighborListConfig,
    available_neighbor_backends,
    get_neighbor_backend,
    neighbor_capabilities,
    resolve_neighbor_backend,
)
from mace_core.neighbors.geometry import SearchGeometry, search_geometry
from mace_core.neighbors.neighborhood import (
    NeighborList,
    get_neighborhood,
    get_neighborhoods,
)

__all__ = [
    "DEFAULT_NEIGHBOR_BACKEND",
    "NEIGHBOR_BACKEND_GROUP",
    "AllPairsBackend",
    "BatchedNeighborListBackend",
    "DiscoveredNeighborBackend",
    "DuplicateNeighborBackendError",
    "NeighborBackendNotAvailableError",
    "NeighborList",
    "NeighborListBackend",
    "NeighborListConfig",
    "SearchGeometry",
    "available_neighbor_backends",
    "get_neighbor_backend",
    "get_neighborhood",
    "get_neighborhoods",
    "neighbor_capabilities",
    "resolve_neighbor_backend",
    "search_geometry",
]
