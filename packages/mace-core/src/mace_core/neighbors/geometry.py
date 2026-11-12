"""The box a neighbour search bins into, and the cell the graph carries.

These are two different matrices with different readers.

**The search cell** belongs to the search. A binning search cannot bin along an
axis that is not periodic, so every aperiodic row is replaced, for the search
only, by an axis-aligned row sized from the atom extent along that axis::

    search_cell[axis] = (extent + 2 * cutoff + 1) * e_axis
    extent            = max(positions[:, axis]) - min(positions[:, axis])

The size depends on the extent and not on the absolute coordinates, so the
same molecule placed far from the origin gets the same box. A backend that
bins may take this box from :func:`search_geometry`; it never leaves the
search.

**The graph cell** is the cell as parsed. It is what the graph carries and what
``shifts`` are measured in, :func:`graph_cell`: the parsed cell, zero when
there is none. Nothing is invented: a periodic axis needs a non-zero lattice
vector, and a structure without one is refused rather than given an identity
cell. No row is extended and none is patched: unit shifts along an aperiodic
axis are zero, so the aperiodic rows never enter a shift, and nothing
downstream divides by a volume it did not ask for. The cell therefore never
depends on the positions.

Neither matrix is ever written back to the
:class:`~mace_core.data.configuration.Configuration` the positions came from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["SearchGeometry", "graph_cell", "search_geometry"]


@dataclass(frozen=True)
class SearchGeometry:
    """Where a neighbour search looks.

    Attributes:
        pbc: ``(3,)`` per-axis periodicity, as plain ``bool``.
        search_cell: ``[3, 3]`` row-vector lattice in Angstrom that the search
            bins into. Its periodic rows are the physical lattice vectors; every
            aperiodic row is axis-aligned and sized from the atom extent.
            Read-only.
    """

    pbc: tuple[bool, bool, bool]
    search_cell: np.ndarray


def search_geometry(
    positions: Any,
    cutoff: float,
    pbc: Any = None,
    cell: Any = None,
) -> SearchGeometry:
    """The search box for one structure.

    Args:
        positions: ``[n_atoms, 3]`` Cartesian positions in Angstrom.
        cutoff: The neighbour cutoff in Angstrom, positive and finite.
        pbc: Three flags, one per lattice vector. ``None`` means no axis is
            periodic.
        cell: ``[3, 3]`` lattice vectors as rows, in Angstrom, or ``None``,
            which is the zero cell. Only the periodic rows are read.

    Nothing passed in is modified, and nothing returned shares memory with it.

    Raises:
        ValueError: if a shape is wrong, the cutoff is not a positive finite
            number, a ``pbc`` entry is not a boolean, the cell is not finite,
            or a periodic axis has an all-zero lattice vector, which leaves the
            search nothing to repeat.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions has shape {positions.shape}. Expected [n_atoms, 3]."
        )
    cutoff = _distance(cutoff)
    periodic = _periodicity(pbc)
    lattice = _lattice(cell, periodic)

    search_cell = lattice.copy()
    for axis in range(3):
        if not periodic[axis]:
            extent = (
                float(positions[:, axis].max() - positions[:, axis].min())
                if len(positions)
                else 0.0
            )
            search_cell[axis] = (extent + 2 * cutoff + 1) * np.eye(3)[axis]

    search_cell.setflags(write=False)
    return SearchGeometry(pbc=periodic, search_cell=search_cell)


def graph_cell(pbc: Any = None, cell: Any = None) -> np.ndarray:
    """The cell a structure's graph carries, ``[3, 3]`` float64 in Angstrom:
    the cell as given, or zero when there is none. No row is ever extended or
    patched, so a slab whose aperiodic row is zero keeps that row and a cell
    of zero volume. Read-only.

    Raises:
        ValueError: if ``pbc`` or ``cell`` has the wrong form, the cell is not
            finite, or a periodic axis has an all-zero lattice vector (see
            :func:`search_geometry`).
    """
    carried = _lattice(cell, _periodicity(pbc))
    carried.setflags(write=False)
    return carried


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


def _lattice(cell: Any, periodic: tuple[bool, bool, bool]) -> np.ndarray:
    """``cell`` as a new float64 array, zero for ``None``, checked."""
    if cell is None:
        lattice = np.zeros((3, 3), dtype=np.float64)
    else:
        lattice = np.array(cell, dtype=np.float64, copy=True)
    if lattice.shape != (3, 3):
        raise ValueError(
            f"cell has shape {lattice.shape}. Expected [3, 3], lattice vectors as rows."
        )
    if not np.isfinite(lattice).all():
        raise ValueError("the cell has a non-finite entry.")
    for axis in range(3):
        if periodic[axis] and not lattice[axis].any():
            raise ValueError(
                f"axis {axis} is periodic but its lattice vector is all zero, "
                f"so there is no image to repeat. Give the cell a non-zero row "
                f"{axis}, or set pbc[{axis}] to False."
            )
    return lattice
