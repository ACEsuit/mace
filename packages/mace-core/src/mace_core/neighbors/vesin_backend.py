"""vesin's cell-list search, mace-torch's default.

vesin bins atoms into cells of the box it is given, so it searches the
:attr:`~mace_core.neighbors.geometry.SearchGeometry.search_cell`, as matscipy
does, and takes the periodicity per axis. It compares squared lengths against
the cutoff and never reports the zero-shift self pair.

Its edges come out in an order of its own, which differs from matscipy's. The
order does not depend on ``n_threads``, so the thread count changes the speed
and never a number. matscipy stays registered because the legacy list searches
with it, and a comparison against legacy that includes the edge order has to
name it.

vesin is the ``mace-core[vesin]`` extra. It is imported when the backend is
built rather than with this module, because importing vesin also imports torch
or cupy whenever either is installed, and ``mace-core`` reaches no framework.
Without vesin, importing this module fails, so the registry lists the backend
as not loaded and says how to install it.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import numpy as np

from mace_core.neighbors.geometry import SearchGeometry

if find_spec("vesin") is None:
    raise ImportError(
        "the vesin neighbour-list backend needs vesin: pip install 'mace-core[vesin]'"
    )

__all__ = ["VesinBackend"]


class VesinBackend:
    """Neighbour search through :class:`vesin.NeighborList`.

    Args:
        n_threads: CPU threads per search. ``0`` lets vesin choose:
            ``OMP_NUM_THREADS`` when it is set, otherwise every available core.
            The default is one thread, because graphs are usually built in
            several data-loader workers at once and each of them would
            otherwise take every core.

    Raises:
        ValueError: if ``n_threads`` is not a non-negative integer.
    """

    def __init__(self, n_threads: int = 1) -> None:
        if isinstance(n_threads, bool) or not isinstance(n_threads, int):
            raise ValueError(
                f"n_threads must be an integer, not {type(n_threads).__name__}."
            )
        if n_threads < 0:
            raise ValueError(
                f"n_threads is {n_threads}. It must be 0, for vesin's own choice, "
                f"or a positive number of threads."
            )
        from vesin import NeighborList

        self._neighbor_list: Any = NeighborList
        self.n_threads = n_threads

    def search(
        self, positions: np.ndarray, cutoff: float, geometry: SearchGeometry
    ) -> tuple[np.ndarray, np.ndarray]:
        """See :meth:`~mace_core.neighbors.backends.NeighborListBackend.search`.

        Edges come out in vesin's order.
        """
        neighbor_list = self._neighbor_list(
            cutoff=cutoff, full_list=True, n_threads=self.n_threads
        )
        senders, receivers, unit_shifts = neighbor_list.compute(
            points=positions,
            box=geometry.search_cell,
            periodic=list(geometry.pbc),
            quantities="ijS",
        )
        return np.stack((senders, receivers)), unit_shifts
