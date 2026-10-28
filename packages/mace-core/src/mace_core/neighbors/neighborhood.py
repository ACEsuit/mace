"""The neighbour list of one structure: the edge set and the cell it is measured in.

:func:`get_neighborhood` is a pure function. It copies what it needs and never
writes into its arguments, so a caller has nothing to protect with a copy of
its own.

The pinned semantics, each one a property the all-pairs backend states
directly:

* the list is **full and directed**: an edge ``i -> j`` with distance
  ``< cutoff`` (strict) appears once per direction. The comparison is on
  float64 squared lengths, ``|r|^2 < cutoff^2``, as matscipy makes it;
* the only pair removed is the **zero-shift self pair**, ``i == j`` with
  ``unit_shifts == 0``. A periodic image of an atom is a real neighbour, so a
  single atom in a cell smaller than the cutoff has edges, all to itself;
* a structure with **no edges** is legal and returns empty arrays;
* ``pbc=None`` means no axis is periodic, and a missing or all-zero cell
  stands for the identity before any aperiodic row is extended;
* the **effective cell** is one of three regimes, decided by
  :func:`~mace_core.neighbors.geometry.search_geometry` before the backend runs
  and never by the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mace_core.neighbors.backends import NeighborListBackend
from mace_core.neighbors.geometry import search_geometry

__all__ = ["NeighborList", "get_neighborhood"]


@dataclass(frozen=True)
class NeighborList:
    """The edges of one structure, and the cell their shifts are measured in.

    Attributes:
        edge_index: ``[2, n_edges]`` int64; row 0 the sender, row 1 the receiver.
        unit_shifts: ``[n_edges, 3]`` int64 lattice-vector counts. The edge
            vector is ``positions[receiver] - positions[sender] + shifts``.
        shifts: ``[n_edges, 3]`` float64 displacements in Angstrom, exactly
            ``unit_shifts @ cell`` with the **returned** cell.
        cell: ``[3, 3]`` float64, the effective cell (see
            :mod:`mace_core.neighbors.geometry` for its three regimes).

    Every array is read-only.
    """

    edge_index: np.ndarray
    unit_shifts: np.ndarray
    shifts: np.ndarray
    cell: np.ndarray


def get_neighborhood(
    positions: Any,
    cutoff: float,
    pbc: Any = None,
    cell: Any = None,
    *,
    backend: NeighborListBackend,
) -> NeighborList:
    """Every directed edge shorter than ``cutoff``, and the effective cell.

    Args:
        positions: ``[n_atoms, 3]`` Cartesian positions in Angstrom.
        cutoff: Strict distance bound in Angstrom.
        pbc: Three flags, one per lattice vector; ``None`` is fully aperiodic.
        cell: ``[3, 3]`` lattice vectors as rows, in Angstrom, or ``None``.
        backend: The resolved backend, from
            :func:`~mace_core.neighbors.backends.get_neighbor_backend`.

    Raises:
        ValueError: if an input has the wrong shape (see
            :func:`~mace_core.neighbors.geometry.search_geometry`), or if the
            backend returns arrays of the wrong shape or non-integer shifts.
    """
    positions = np.array(positions, dtype=np.float64, copy=True)
    positions.setflags(write=False)
    geometry = search_geometry(positions, cutoff, pbc, cell)
    edge_index, unit_shifts = backend.search(positions, float(cutoff), geometry)
    edge_index, unit_shifts = _checked_edges(edge_index, unit_shifts, len(positions))

    true_self_edge = (edge_index[0] == edge_index[1]) & np.all(unit_shifts == 0, axis=1)
    edge_index = edge_index[:, ~true_self_edge]
    unit_shifts = unit_shifts[~true_self_edge]
    shifts = np.dot(unit_shifts, geometry.cell)
    for array in (edge_index, unit_shifts, shifts):
        array.setflags(write=False)
    return NeighborList(
        edge_index=edge_index,
        unit_shifts=unit_shifts,
        shifts=shifts,
        cell=geometry.cell,
    )


def _checked_edges(
    edge_index: Any, unit_shifts: Any, n_atoms: int
) -> tuple[np.ndarray, np.ndarray]:
    """The backend's arrays as int64, or a ``ValueError`` naming what is wrong."""
    edge_index = np.asarray(edge_index)
    unit_shifts = np.asarray(unit_shifts)
    if edge_index.size == 0:
        edge_index = edge_index.reshape(2, 0)
    if unit_shifts.size == 0:
        unit_shifts = unit_shifts.reshape(0, 3)
    n_edges = edge_index.shape[-1] if edge_index.ndim == 2 else -1
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"the neighbour backend returned edge_index of shape "
            f"{edge_index.shape}. Expected [2, n_edges]."
        )
    if unit_shifts.shape != (n_edges, 3):
        raise ValueError(
            f"the neighbour backend returned unit_shifts of shape "
            f"{unit_shifts.shape} for {n_edges} edges. Expected [n_edges, 3]."
        )
    for name, array in (("edge_index", edge_index), ("unit_shifts", unit_shifts)):
        if array.size and not np.issubdtype(array.dtype, np.integer):
            raise ValueError(
                f"the neighbour backend returned {name} as {array.dtype}. "
                f"Expected integers."
            )
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= n_atoms):
        raise ValueError(
            f"the neighbour backend returned an edge to atom "
            f"{int(edge_index.max())} or {int(edge_index.min())} in a structure "
            f"of {n_atoms} atoms."
        )
    return edge_index.astype(np.int64), unit_shifts.astype(np.int64)
