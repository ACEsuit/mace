from typing import Dict, Optional, Tuple

import numpy as np
from matscipy.neighbours import neighbour_list

# A direction counts as free only if what is left of it after removing the
# periodic span is a real direction rather than round-off.
_DEGENERATE = 1e-8


def _orthogonal_residual(
    vector: np.ndarray, basis: list  # [3], list of orthonormal [3]
) -> np.ndarray:
    residual = np.array(vector, dtype=float)
    for axis in basis:
        residual = residual - (residual @ axis) * axis
    return residual


def _aperiodic_search_directions(
    pbc: Tuple[bool, bool, bool],
    cell: np.ndarray,  # [3, 3]
) -> Dict[int, np.ndarray]:
    """One unit vector per non-periodic axis, to build the fictitious box along.

    Each is orthogonal to every periodic lattice vector, so no periodic image
    can carry an atom along it. That is the property the search box needs: a
    box the atoms can be moved out of gets them wrapped by matscipy, which
    reports the wrap as a shift the caller's unwrapped positions know nothing
    about. Along a Cartesian axis any lattice vector with a component there
    does exactly that.

    Whenever the Cartesian axis is already free, which covers every cell built
    the usual way, it is the one returned, so the search and the returned cell
    are unchanged for those.
    """
    aperiodic = [dim for dim in range(3) if not pbc[dim]]
    if not aperiodic:
        return {}

    identity = np.identity(3, dtype=float)
    if not any(pbc):
        # Nothing can move an atom anywhere, so every axis is free. Worth its
        # own line: it is the molecule case, and it is the common one.
        return {dim: identity[dim] for dim in aperiodic}
    # Orthonormal basis of the directions a periodic image can move an atom
    # along. The non-periodic directions are then chosen outside its span, and
    # each is added to it so two of them cannot come out parallel.
    spanned: list = []
    for dim in range(3):
        if not pbc[dim]:
            continue
        residual = _orthogonal_residual(cell[dim], spanned)
        scale = max(float(np.linalg.norm(cell[dim])), 1.0)
        if np.linalg.norm(residual) > _DEGENERATE * scale:
            spanned.append(residual / np.linalg.norm(residual))

    directions: Dict[int, np.ndarray] = {}
    for dim in aperiodic:
        # The axis this row stands for first, then the other two as a fallback
        # for a cell whose periodic vectors happen to span it.
        for candidate in (identity[dim], identity[0], identity[1], identity[2]):
            residual = _orthogonal_residual(candidate, spanned)
            norm = float(np.linalg.norm(residual))
            if norm > _DEGENERATE:
                directions[dim] = residual / norm
                spanned.append(directions[dim])
                break
    return directions


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

    # matscipy cannot bin atoms along a non-periodic axis, so we blow up the
    # cell there just for the neighbour search. Two properties of that box
    # matter, and it needs both.
    #
    # Its size comes from the atom extent (+ cutoff padding) rather than from
    # max(abs(positions)): the old `max(abs(positions)) * 5 * cutoff` depended
    # on the absolute coordinate origin and produced huge cells, which blow up
    # PolarMACE's k-space electrostatics into GPU OOM. Extent-based padding
    # gives identical neighbour lists at a fraction of the volume.
    #
    # Its direction is orthogonal to every periodic lattice vector rather than
    # along a Cartesian axis, and the atoms are moved into it. Both halves are
    # needed to make the neighbour list independent of where the caller put the
    # structure: the offset handles a structure that starts outside the box,
    # the direction stops a periodic image from carrying one back out.
    search_directions = _aperiodic_search_directions(pbc, cell)
    search_positions = np.array(positions, dtype=float, copy=True)
    extended_cell = np.array(cell, dtype=float, copy=True)
    for dim, direction in search_directions.items():
        projection = search_positions @ direction
        extent = projection.max() - projection.min()
        extended_cell[dim, :] = (extent + 2 * cutoff + 1) * direction
        # A cutoff of clearance rather than flush against the wall: on the wall
        # matscipy pays for the boundary bins and the search measurably slows
        # down. The offset is rigid, so it changes no interatomic distance, and
        # the caller's positions are untouched.
        search_positions -= (projection.min() - cutoff) * direction

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
        positions=search_positions,
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
