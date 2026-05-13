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

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    # Extend cell in non-periodic directions using the actual extent of
    # the atoms plus cutoff padding.  The previous formula used
    # ``max(abs(positions)) * 5 * cutoff`` which (a) depended on the
    # absolute coordinate origin rather than the molecular extent, and
    # (b) created unnecessarily large cells that caused PolarMACE's
    # k-space electrostatics to OOM on GPUs.  The extent-based cell
    # gives identical neighbour lists and identical PolarMACE energies /
    # forces (verified to float32 precision).
    if not pbc_x:
        extent_x = positions[:, 0].max() - positions[:, 0].min()
        cell[0, :] = (extent_x + 2 * cutoff + 1) * identity[0, :]
    if not pbc_y:
        extent_y = positions[:, 1].max() - positions[:, 1].min()
        cell[1, :] = (extent_y + 2 * cutoff + 1) * identity[1, :]
    if not pbc_z:
        extent_z = positions[:, 2].max() - positions[:, 2].min()
        cell[2, :] = (extent_z + 2 * cutoff + 1) * identity[2, :]

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
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

