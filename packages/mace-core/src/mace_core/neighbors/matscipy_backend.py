"""The legacy backend: matscipy's linked-cell search.

The legacy neighbour list searches with matscipy, so this backend reproduces
it edge for edge in the same order. A test that compares against legacy
including the order names it. matscipy is the ``mace-core[matscipy]`` extra,
and it is mace-jax's default.

matscipy bins atoms into cells of the box it is given, which is why it searches
the :attr:`~mace_core.neighbors.geometry.SearchGeometry.search_cell` rather
than the physical cell: along an aperiodic axis there is nothing to bin into
otherwise. It never reports the zero-shift self pair.
"""

from __future__ import annotations

import numpy as np

try:
    from matscipy.neighbours import neighbour_list
except ImportError as missing:
    raise ImportError(
        "the matscipy neighbour-list backend needs matscipy: "
        "pip install 'mace-core[matscipy]'"
    ) from missing

from mace_core.neighbors.geometry import SearchGeometry

__all__ = ["MatscipyBackend"]


class MatscipyBackend:
    """Neighbour search through :func:`matscipy.neighbours.neighbour_list`."""

    def search(
        self, positions: np.ndarray, cutoff: float, geometry: SearchGeometry
    ) -> tuple[np.ndarray, np.ndarray]:
        """See :meth:`~mace_core.neighbors.backends.NeighborListBackend.search`.

        Edges come out in matscipy's order.
        """
        senders, receivers, unit_shifts = neighbour_list(
            quantities="ijS",
            pbc=geometry.pbc,
            cell=geometry.search_cell,
            positions=positions,
            cutoff=cutoff,
        )
        return np.stack((senders, receivers)), unit_shifts
