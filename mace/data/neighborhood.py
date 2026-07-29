from typing import Optional, Tuple

import numpy as np
from matscipy.neighbours import neighbour_list


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    identity = np.identity(3, dtype=float)

    # matscipy cannot bin atoms along a non-periodic axis, so we blow up the
    # cell there just for the neighbour search. Size it from the actual atom
    # extent (+ cutoff padding) rather than from max(abs(positions)): the old
    # `max(abs(positions)) * 5 * cutoff` depended on the absolute coordinate
    # origin and produced huge cells, which blow up PolarMACE's k-space
    # electrostatics into GPU OOM. Extent-based padding gives identical
    # neighbour lists at a fraction of the volume.
    extended_cell = np.array(cell, dtype=float, copy=True)
    for dim in range(3):
        if not pbc[dim]:
            extent = positions[:, dim].max() - positions[:, dim].min()
            extended_cell[dim, :] = (extent + 2 * cutoff + 1) * identity[dim, :]

    # The neighbour search uses the blown-up cell, but we must not *return* it
    # when any axis is periodic: stress later normalizes by det(cell), so a
    # fake volume along the vacuum axis silently rescales the stress of slabs
    # and other partially periodic systems. Return the physical cell there.
    # Fully aperiodic systems keep the extended cell (stress is meaningless
    # and long-range models need a non-degenerate cell).
    #
    # NOTE: electrostatic models (e.g. PolarMACE) also read this cell as their
    # k-space box and, crucially, its det() as the volume their slab / molecule
    # dipole corrections divide by. Returning the physical cell is what gives
    # those corrections the right volume; the inflated cell would silently scale
    # them away. (Adequate vacuum is still the caller's job, as in any 3D-Ewald
    # slab calculation.)
    if any(pbc):
        cell = np.array(cell, dtype=float, copy=True)
        # A non-periodic axis whose physical row is all zeros (e.g. a slab built
        # with zero vacuum) would leave det(cell)=0, which NaNs the stress
        # (division by volume) and blows up rcell. Keep the extended row there.
        for dim in range(3):
            if not pbc[dim] and not cell[dim].any():
                cell[dim] = extended_cell[dim]
    else:
        cell = extended_cell

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=extended_cell,
        positions=positions,
        cutoff=cutoff,
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts, cell
