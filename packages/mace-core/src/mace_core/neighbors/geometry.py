"""The box a neighbour search bins into, and the cell the graph carries.

These are two different matrices, and which one comes back with the graph is
a decision with three outcomes. It is made here, by one pure function, so the
graph builder and every neighbour-list backend read it from the same place and
no backend can compute a regime of its own.

**The search cell.** A binning search cannot bin along an axis that is not
periodic, so every aperiodic row is replaced, for the search only, by an
axis-aligned row sized from the atom extent along that axis::

    search_cell[axis] = (extent + 2 * cutoff + 1) * e_axis
    extent            = max(positions[:, axis]) - min(positions[:, axis])

The size depends on the extent and not on the absolute coordinates, so the
same molecule placed far from the origin gets the same box. A box that grew
with the origin is what made the long-range models' reciprocal space run out
of memory.

**The returned cell**, the effective cell the graph carries and the stress and
the reciprocal-space path divide by:

=================================================  ==============================
regime                                             returned cell
=================================================  ==============================
no axis periodic                                   the search cell
some axis periodic, every physical row non-zero    the physical cell
some axis periodic, an aperiodic row all zero      the physical cell, with that
                                                   row taken from the search cell
=================================================  ==============================

A fully aperiodic structure has no meaningful stress, and a long-range model
needs a non-degenerate box, so it gets the search cell. A structure with a
periodic axis keeps its physical cell, because its stress divides by the cell
volume and a vacuum row invented for the search would rescale it. The one
exception is a physical row that is all zero, which would make the volume zero;
that row alone is taken from the search cell.

Neither matrix is ever written back to the
:class:`~mace_core.data.configuration.Configuration` the positions came from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["SearchGeometry", "search_geometry"]


@dataclass(frozen=True)
class SearchGeometry:
    """Where a neighbour search looks, and the cell its graph carries.

    Attributes:
        pbc: ``(3,)`` per-axis periodicity, as plain ``bool``.
        search_cell: ``[3, 3]`` row-vector lattice in Angstrom that the search
            bins into. Its periodic rows are the physical lattice vectors; every
            aperiodic row is axis-aligned and sized from the atom extent.
        cell: ``[3, 3]`` the effective cell in Angstrom, chosen by the regime
            table in this module's docstring. ``shifts`` are computed against
            it, as ``unit_shifts @ cell``.

    Both arrays are read-only copies.
    """

    pbc: tuple[bool, bool, bool]
    search_cell: np.ndarray
    cell: np.ndarray


def search_geometry(
    positions: Any,
    cutoff: float,
    pbc: Any = None,
    cell: Any = None,
) -> SearchGeometry:
    """The search box and the effective cell for one structure.

    Args:
        positions: ``[n_atoms, 3]`` Cartesian positions in Angstrom.
        cutoff: The neighbour cutoff in Angstrom, positive and finite.
        pbc: Three flags, one per lattice vector. ``None`` means no axis is
            periodic.
        cell: ``[3, 3]`` lattice vectors as rows, in Angstrom. ``None`` or an
            all-zero matrix stands for the identity before any row is
            extended, which is what an aperiodic structure normally carries.

    Nothing passed in is modified, and nothing returned shares memory with it.

    Raises:
        ValueError: if a shape is wrong, the cutoff is not a positive finite
            number, a ``pbc`` entry is not a boolean, or a periodic axis has an
            all-zero lattice vector, which leaves the search nothing to repeat.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions has shape {positions.shape}. Expected [n_atoms, 3]."
        )
    cutoff = _distance(cutoff)
    periodic = _periodicity(pbc)
    lattice = _lattice(cell)

    for axis in range(3):
        if periodic[axis] and not lattice[axis].any():
            raise ValueError(
                f"axis {axis} is periodic but its lattice vector is all zero, "
                f"so there is no image to repeat. Give the cell a non-zero row "
                f"{axis}, or set pbc[{axis}] to False."
            )

    search_cell = lattice.copy()
    for axis in range(3):
        if not periodic[axis]:
            extent = (
                float(positions[:, axis].max() - positions[:, axis].min())
                if len(positions)
                else 0.0
            )
            search_cell[axis] = (extent + 2 * cutoff + 1) * np.eye(3)[axis]

    if any(periodic):
        effective_cell = lattice.copy()
        for axis in range(3):
            if not periodic[axis] and not effective_cell[axis].any():
                effective_cell[axis] = search_cell[axis]
    else:
        effective_cell = search_cell.copy()

    search_cell.setflags(write=False)
    effective_cell.setflags(write=False)
    return SearchGeometry(pbc=periodic, search_cell=search_cell, cell=effective_cell)


def _distance(cutoff: Any) -> float:
    """``cutoff`` as a float, if it is a positive finite number."""
    try:
        value = float(cutoff)
    except (TypeError, ValueError):
        value = math.nan
    if isinstance(cutoff, bool) or not (math.isfinite(value) and value > 0):
        raise ValueError(
            f"cutoff is {cutoff!r}. Expected a positive, finite distance in Angstrom."
        )
    return value


def _periodicity(pbc: Any) -> tuple[bool, bool, bool]:
    if pbc is None:
        return (False, False, False)
    flags = tuple(pbc)
    if len(flags) != 3 or not all(isinstance(f, bool | np.bool_) for f in flags):
        raise ValueError(
            f"pbc is {pbc!r}. Expected three booleans, one per lattice vector."
        )
    return (bool(flags[0]), bool(flags[1]), bool(flags[2]))


def _lattice(cell: Any) -> np.ndarray:
    if cell is None:
        return np.eye(3, dtype=np.float64)
    lattice = np.array(cell, dtype=np.float64, copy=True)
    if lattice.shape != (3, 3):
        raise ValueError(
            f"cell has shape {lattice.shape}. Expected [3, 3], lattice vectors as rows."
        )
    if not lattice.any():
        return np.eye(3, dtype=np.float64)
    return lattice
