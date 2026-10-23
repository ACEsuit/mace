"""Neighbour lists: the edge set of a structure and the cell it is measured in.

:func:`get_neighborhood` is the one entry point every graph builder calls, and
no dataset backend ships neighbour code of its own. The search itself is a
pluggable :class:`NeighborListBackend`, selected by the ``neighbor_list``
config section (:class:`NeighborListConfig`); the search box and the returned
cell are decided once, by :func:`search_geometry`, whichever backend runs.

Everything here is numpy. No framework is imported.
"""

from mace_core.neighbors.backends import (
    DEFAULT_NEIGHBOR_BACKEND,
    NEIGHBOR_BACKEND_GROUP,
    DiscoveredNeighborBackend,
    DuplicateNeighborBackendError,
    NeighborBackendNotAvailableError,
    NeighborListBackend,
    NeighborListConfig,
    available_neighbor_backends,
    get_neighbor_backend,
)
from mace_core.neighbors.brute_force import BruteForceBackend
from mace_core.neighbors.geometry import SearchGeometry, search_geometry
from mace_core.neighbors.neighborhood import NeighborList, get_neighborhood

__all__ = [
    "DEFAULT_NEIGHBOR_BACKEND",
    "NEIGHBOR_BACKEND_GROUP",
    "BruteForceBackend",
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
    "search_geometry",
]
