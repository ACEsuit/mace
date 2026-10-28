"""The all-pairs backend: every pair of atoms, every image that can reach.

It bins nothing, so it needs no artificial box, and it is short enough to be
read as a definition of the edge set. It compares squared lengths, as matscipy
does: a length rounded by a square root can land on the cutoff when the exact
one is below it. It is the oracle the conformance harness
measures other backends against and a reasonable choice for small structures.
Cost is ``O(n_atoms^2 x n_images)``.

**How many images.** Along a periodic axis with reciprocal vector ``b``, an
edge with integer shift ``n`` has a fractional separation of
``(f_j - f_i) + n`` on that axis, and its length is at least that separation
divided by ``|b|``. So no edge needs ``|n|`` beyond
``cutoff * |b| + spread``, where ``spread`` is the range of the atoms'
fractional coordinates on the axis. That bound is exact, which is why this
backend can enumerate a fixed range rather than grow one until it stops
changing.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from mace_core.neighbors.geometry import SearchGeometry

__all__ = ["AllPairsBackend"]


class AllPairsBackend:
    """Neighbour search by enumerating every pair and every reachable image."""

    def search(
        self, positions: np.ndarray, cutoff: float, geometry: SearchGeometry
    ) -> tuple[np.ndarray, np.ndarray]:
        """See :meth:`~mace_core.neighbors.backends.NeighborListBackend.search`.

        Edges come out sorted by sender, then receiver, then shift.
        """
        positions = np.asarray(positions, dtype=np.float64)
        ranges = _image_ranges(positions, cutoff, geometry)
        sender_blocks, receiver_blocks, shift_blocks = [], [], []
        difference = positions[None, :, :] - positions[:, None, :]  # [i, j] = r_j - r_i
        for shift in itertools.product(*ranges):
            unit_shift = np.array(shift, dtype=np.int64)
            vectors = difference + unit_shift @ geometry.search_cell
            squared = np.sum(vectors * vectors, axis=-1)
            senders, receivers = np.nonzero(squared < cutoff * cutoff)
            sender_blocks.append(senders)
            receiver_blocks.append(receivers)
            shift_blocks.append(np.broadcast_to(unit_shift, (len(senders), 3)))
        senders = np.concatenate(sender_blocks).astype(np.int64)
        receivers = np.concatenate(receiver_blocks).astype(np.int64)
        unit_shifts = np.concatenate(shift_blocks).astype(np.int64).reshape(-1, 3)
        order = np.lexsort(
            (
                unit_shifts[:, 2],
                unit_shifts[:, 1],
                unit_shifts[:, 0],
                receivers,
                senders,
            )
        )
        return np.stack((senders[order], receivers[order])), unit_shifts[order]


def _image_ranges(
    positions: np.ndarray, cutoff: float, geometry: SearchGeometry
) -> list[range]:
    """The integer shifts to try along each axis; only ``0`` on aperiodic ones."""
    # Rows b_k with a_i . b_k = delta_ik.
    reciprocal = np.linalg.inv(geometry.search_cell).T
    fractional = positions @ np.linalg.inv(geometry.search_cell)
    ranges = []
    for axis in range(3):
        if not geometry.pbc[axis]:
            ranges.append(range(1))
            continue
        spread = (
            float(fractional[:, axis].max() - fractional[:, axis].min())
            if len(positions)
            else 0.0
        )
        reach = math.ceil(cutoff * float(np.linalg.norm(reciprocal[axis])) + spread)
        ranges.append(range(-reach, reach + 1))
    return ranges
