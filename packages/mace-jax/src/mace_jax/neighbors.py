"""The neighbour-list backend this framework builds graphs with by default.

A ``neighbor_list`` config section that names no backend gets matscipy, which
imports no framework. vesin would be faster, but importing it also imports
torch whenever torch is installed. The package requires the
``mace-core[matscipy]`` extra, so the default is installed wherever this
package is. Any other backend is used only when the section names it.
"""

from __future__ import annotations

from mace_core import neighbors
from mace_core.neighbors import NeighborListBackend, NeighborListConfig

__all__ = ["DEFAULT_NEIGHBOR_BACKEND", "resolve_neighbor_backend"]

#: The backend used when the config names none.
DEFAULT_NEIGHBOR_BACKEND = "matscipy"


def resolve_neighbor_backend(
    config: NeighborListConfig | None = None,
) -> NeighborListBackend:
    """The backend ``config`` names, or this framework's default.

    See :func:`mace_core.neighbors.resolve_neighbor_backend`.
    """
    return neighbors.resolve_neighbor_backend(config, DEFAULT_NEIGHBOR_BACKEND)
