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
    max_positions = np.max(np.absolute(positions)) + 1
    # blow up the cell in non-periodic directions so matscipy can bin the atoms
    # (constant may need bumping for models with more than 5 layers)
    extended_cell = np.array(cell, dtype=float, copy=True)
    for dim in range(3):
        if not pbc[dim]:
            extended_cell[dim, :] = max_positions * 5 * cutoff * identity[dim, :]

    # only use the blown-up cell for the neighbour search, don't return it:
    # stress divides by det(cell) later, so the fake volume shrinks it (#1509).
    # keep it for fully aperiodic systems though (no stress there anyway)
    if any(pbc):
        cell = np.array(cell, dtype=float, copy=True)
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
