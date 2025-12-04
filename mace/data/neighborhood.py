from typing import List, Optional, Tuple, Sequence

import numpy as np
from matscipy.neighbours import neighbour_list

def get_neighborhood_batched(
    positions_list: Sequence[np.ndarray],  # list of [num_positions_i, 3]
    cutoff: float,
    pbc_list: Optional[Sequence[Optional[Tuple[bool, bool, bool]]]] = None,
    cell_list: Optional[Sequence[Optional[np.ndarray]]] = None,  # list of [3, 3]
    true_self_interaction: bool = False,
) -> Tuple[
    List[np.ndarray],  # edge_index_list
    List[np.ndarray],  # shifts_list
    List[np.ndarray],  # unit_shifts_list
    List[np.ndarray],  # cell_list_out
]:
    """
    For now: trivial batched version that just loops over structures and
    calls get_neighborhood for each one.
    """
    if pbc_list is None:
        pbc_list = [None] * len(positions_list)
    if cell_list is None:
        cell_list = [None] * len(positions_list)

    edge_index_list: List[np.ndarray] = []
    shifts_list: List[np.ndarray] = []
    unit_shifts_list: List[np.ndarray] = []
    cell_list_out: List[np.ndarray] = []

    for positions, pbc, cell in zip(positions_list, pbc_list, cell_list):
        edge_index, shifts, unit_shifts, cell_out = get_neighborhood(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            true_self_interaction=true_self_interaction,
        )
        edge_index_list.append(edge_index)
        shifts_list.append(shifts)
        unit_shifts_list.append(unit_shifts)
        cell_list_out.append(cell_out)

    return edge_index_list, shifts_list, unit_shifts_list, cell_list_out
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
    max_positions = np.max(np.absolute(positions)) + 1
    # Extend cell in non-periodic directions
    # For models with more than 5 layers, the multiplicative constant needs to be increased.
    # temp_cell = np.copy(cell)
    if not pbc_x:
        cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        cell[2, :] = max_positions * 5 * cutoff * identity[2, :]

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
