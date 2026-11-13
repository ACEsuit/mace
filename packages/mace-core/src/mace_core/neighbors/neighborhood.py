"""The neighbour list of one structure: its edge set, recorded with its cutoff.

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
* ``pbc=None`` means no axis is periodic, a missing cell is the zero cell, and
  a periodic axis with an all-zero lattice vector is refused;
* images exist only along periodic axes, so every unit shift is zero along an
  aperiodic one, and ``shifts`` are measured in the cell the graph carries,
  :func:`~mace_core.neighbors.geometry.graph_cell`. The box a backend searches
  in is its own and never comes back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mace_core.neighbors.backends import NeighborListBackend
from mace_core.neighbors.geometry import graph_cell, search_geometry

__all__ = ["NeighborList", "checked_edges", "get_neighborhood"]


@dataclass(frozen=True)
class NeighborList:
    """The edges of one structure, and the cutoff they were searched with.

    Attributes:
        edge_index: ``[2, n_edges]`` int64; row 0 the sender, row 1 the receiver.
        unit_shifts: ``[n_edges, 3]`` int64 lattice-vector counts, zero along
            every aperiodic axis. The edge vector is
            ``positions[receiver] - positions[sender] + shifts``.
        shifts: ``[n_edges, 3]`` float64 displacements in Angstrom, exactly
            ``unit_shifts @ graph_cell(pbc, cell)``.
        cutoff: The strict distance bound the list was searched with, in
            Angstrom. A list is only valid for a model of the same cutoff.

    Every array is read-only.
    """

    edge_index: np.ndarray
    unit_shifts: np.ndarray
    shifts: np.ndarray
    cutoff: float


def get_neighborhood(
    positions: Any,
    cutoff: float,
    pbc: Any = None,
    cell: Any = None,
    *,
    backend: NeighborListBackend,
) -> NeighborList:
    """Every directed edge shorter than ``cutoff``.

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
            backend returns arrays of the wrong shape, non-integer shifts, or
            an image along an aperiodic axis.
    """
    positions = np.array(positions, dtype=np.float64, copy=True)
    positions.setflags(write=False)
    geometry = search_geometry(positions, cutoff, pbc, cell)
    edge_index, unit_shifts = backend.search(positions, float(cutoff), geometry)
    edge_index, unit_shifts = checked_edges(
        edge_index, unit_shifts, len(positions), geometry.pbc
    )

    true_self_edge = (edge_index[0] == edge_index[1]) & np.all(unit_shifts == 0, axis=1)
    edge_index = edge_index[:, ~true_self_edge]
    unit_shifts = unit_shifts[~true_self_edge]
    shifts = np.dot(unit_shifts, graph_cell(geometry.pbc, cell))
    for array in (edge_index, unit_shifts, shifts):
        array.setflags(write=False)
    return NeighborList(
        edge_index=edge_index,
        unit_shifts=unit_shifts,
        shifts=shifts,
        cutoff=float(cutoff),
    )


def checked_edges(
    edge_index: Any,
    unit_shifts: Any,
    n_atoms: int,
    pbc: tuple[bool, ...],
    *,
    origin: str = "the neighbour backend",
) -> tuple[np.ndarray, np.ndarray]:
    """A neighbour list's index arrays as int64, checked the same way whoever
    produced them: a search, or a list stored with the data.

    Args:
        edge_index: ``[2, n_edges]`` integer atom indices.
        unit_shifts: ``[n_edges, 3]`` integer lattice-vector counts.
        n_atoms: The atoms of the structure.
        pbc: Its periodicity per axis.
        origin: What produced the arrays, named in an error.

    Raises:
        ValueError: naming ``origin``, for a wrong shape, a non-integer dtype,
            an index outside the structure, or an image along an aperiodic
            axis.
    """
    edge_index = np.asarray(edge_index)
    unit_shifts = np.asarray(unit_shifts)
    if edge_index.size == 0:
        edge_index = edge_index.reshape(2, 0)
    if unit_shifts.size == 0:
        unit_shifts = unit_shifts.reshape(0, 3)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"{origin} gave edge_index of shape {edge_index.shape}. Expected "
            f"[2, n_edges]."
        )
    n_edges = edge_index.shape[1]
    if unit_shifts.shape != (n_edges, 3):
        raise ValueError(
            f"{origin} gave unit_shifts of shape {unit_shifts.shape} for "
            f"{n_edges} edges. Expected [n_edges, 3]."
        )
    for name, array in (("edge_index", edge_index), ("unit_shifts", unit_shifts)):
        if array.size and not np.issubdtype(array.dtype, np.integer):
            raise ValueError(
                f"{origin} gave {name} as {array.dtype}. Expected integers."
            )
    outside = edge_index[(edge_index < 0) | (edge_index >= n_atoms)]
    if outside.size:
        raise ValueError(
            f"{origin} gave an edge to atom {int(outside[0])} in a structure of "
            f"{n_atoms} atoms."
        )
    aperiodic = [axis for axis in range(3) if not pbc[axis]]
    if unit_shifts[:, aperiodic].any():
        raise ValueError(
            f"{origin} gave an image along the aperiodic axes {aperiodic}. "
            f"Images exist only along periodic axes."
        )
    return edge_index.astype(np.int64), unit_shifts.astype(np.int64)
