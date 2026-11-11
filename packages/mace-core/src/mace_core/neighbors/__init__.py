"""Neighbour lists: the edge set of a structure, recorded with its cutoff.

:func:`get_neighborhood` is the one entry point every graph builder calls, and
no dataset backend ships neighbour code of its own. The search itself is a
pluggable :class:`NeighborListBackend`, selected by the ``neighbor_list``
config section (:class:`NeighborListConfig`). A backend that bins may take its
search box from :func:`search_geometry`; the cell the graph carries is the
parsed one, :func:`graph_cell`, whichever backend runs.

Everything here is numpy. No framework is imported.
"""

from mace_core.neighbors.all_pairs import AllPairsBackend
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
    resolve_neighbor_backend,
)
from mace_core.neighbors.geometry import SearchGeometry, graph_cell, search_geometry
from mace_core.neighbors.neighborhood import NeighborList, get_neighborhood

__all__ = [
    "DEFAULT_NEIGHBOR_BACKEND",
    "NEIGHBOR_BACKEND_GROUP",
    "AllPairsBackend",
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
    "graph_cell",
    "resolve_neighbor_backend",
    "search_geometry",
]
