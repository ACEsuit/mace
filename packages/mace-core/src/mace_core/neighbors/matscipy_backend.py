"""The default backend: matscipy's linked-cell search.

matscipy bins atoms into cells of the box it is given, which is why it searches
the :attr:`~mace_core.neighbors.geometry.SearchGeometry.search_cell` rather
than the physical cell: along an aperiodic axis there is nothing to bin into
otherwise. It never reports the zero-shift self pair.
"""

from __future__ import annotations

import numpy as np
from matscipy.neighbours import neighbour_list

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
