"""An independent, brute-force neighbour list, and the vocabulary to compare
one neighbour list against another.

``mace/data/neighborhood.py`` delegates the search to matscipy, so a test that
only checks matscipy against itself checks nothing. This module enumerates
periodic images in plain numpy at O(N^2 x images) and is deliberately slow,
obvious and framework-free: it is the reference the shipped list is measured
against, here and in the rewrite, where every pluggable neighbour backend has
to reproduce it.

Three things are worth stating outright, because each is a decision a
reimplementation can silently get wrong:

* **The comparison is on integer unit shifts, not on displacement vectors.**
  ``get_neighborhood`` builds its shifts as ``unit_shifts @ cell`` (one
  matmul) while any hand-written reference accumulates
  ``n0*a0 + n1*a1 + n2*a2``; the two agree to floating-point round-off, not
  bit for bit, and comparing them needs a tolerance nobody should have to
  pick. The integer triple carries exactly the same information and compares
  exactly, so that is the canonical form here. ``canonical_edges_from_shifts``
  exists for a backend that reports only vectors, and it rounds.

* **The returned cell is not the oracle's business.** ``get_neighborhood``
  returns four values, and which cell comes back is a three-way policy
  decision of that function (see its comments and the tests that pin it).
  An oracle that reproduced the policy could not falsify it, so this one
  returns the three edge quantities and nothing else.

* **The cutoff is strict** (``d < cutoff``, not ``<=``) and self-pairs are
  dropped only at zero shift, which is precisely
  ``true_self_interaction=False``. Periodic self-images survive: one atom in a
  cell smaller than the cutoff has neighbours, all of them itself.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "brute_force_neighborhood",
    "canonical_edges",
    "canonical_edges_from_shifts",
    "assert_neighbourhoods_match",
]

Edge = Tuple[int, int, int, int, int]

#: The image range is grown until the edge set stops changing rather than
#: fixed, because a strongly skewed cell can put an image that a fixed range
#: of +-1 or +-2 would miss inside the cutoff. Two consecutive identical edge
#: sets end the search; the ceiling only exists so a mistake fails loudly
#: instead of looping forever.
_MAX_IMAGE_RADIUS = 12


def brute_force_neighborhood(
    positions: np.ndarray,
    cutoff: float,
    pbc: Optional[Sequence[bool]] = None,
    cell: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate every pair within ``cutoff``, images included.

    Args:
        positions: ``[n_atoms, 3]`` Cartesian coordinates.
        cutoff: strict distance bound, in the same units as ``positions``.
        pbc: per-axis periodicity; defaults to fully aperiodic.
        cell: ``[3, 3]`` row-vector lattice. Only rows whose axis is periodic
            are ever used, so an aperiodic call may pass ``None``.

    Returns:
        ``(edge_index, shifts, unit_shifts)`` with the same layout
        ``get_neighborhood`` uses: ``edge_index`` is ``[2, n_edges]`` as
        ``(sender, receiver)``, ``shifts`` is ``[n_edges, 3]`` in Angstrom and
        ``unit_shifts`` is the integer ``[n_edges, 3]`` such that
        ``positions[j] - positions[i] + shift`` is the edge vector.
    """
    positions = np.asarray(positions, dtype=float)
    if pbc is None:
        pbc = (False, False, False)
    if cell is None:
        cell = np.zeros((3, 3), dtype=float)
    cell = np.asarray(cell, dtype=float)
    if any(pbc) and not cell.any():
        raise ValueError("a periodic axis needs a non-zero cell")

    previous: Optional[List[Edge]] = None
    radius = 1
    while radius <= _MAX_IMAGE_RADIUS:
        found = _pairs_within(positions, cutoff, pbc, cell, radius)
        if found == previous:
            break
        previous, radius = found, radius + 1
    else:
        raise RuntimeError(
            f"the brute-force edge set was still growing at image radius "
            f"{_MAX_IMAGE_RADIUS}; the cutoff ({cutoff}) is large compared to "
            f"the cell and this reference would be incomplete"
        )

    edges = previous or []
    unit_shifts = np.array([edge[2:] for edge in edges], dtype=float).reshape(-1, 3)
    edge_index = np.array([[e[0] for e in edges], [e[1] for e in edges]], dtype=int)
    edge_index = edge_index.reshape(2, -1)
    shifts = unit_shifts @ cell
    return edge_index, shifts, unit_shifts


def _pairs_within(
    positions: np.ndarray,
    cutoff: float,
    pbc: Sequence[bool],
    cell: np.ndarray,
    radius: int,
) -> List[Edge]:
    ranges = [range(-radius, radius + 1) if pbc[d] else range(1) for d in range(3)]
    found: List[Edge] = []
    for i in range(len(positions)):
        for j in range(len(positions)):
            for n_0 in ranges[0]:
                for n_1 in ranges[1]:
                    for n_2 in ranges[2]:
                        if i == j and n_0 == 0 and n_1 == 0 and n_2 == 0:
                            continue  # the only self-pair that is dropped
                        shift = n_0 * cell[0] + n_1 * cell[1] + n_2 * cell[2]
                        delta = positions[j] + shift - positions[i]
                        if float(np.linalg.norm(delta)) < cutoff:
                            found.append((i, j, n_0, n_1, n_2))
    return sorted(found)


def canonical_edges(edge_index: np.ndarray, unit_shifts: np.ndarray) -> List[Edge]:
    """A neighbour list as a sorted list of ``(i, j, n0, n1, n2)`` integers.

    Exact by construction: no rounding, no tolerance. Two lists that describe
    the same edges compare equal whatever order they were produced in.
    """
    edge_index = np.asarray(edge_index).reshape(2, -1)
    unit_shifts = np.asarray(unit_shifts).reshape(-1, 3)
    if edge_index.shape[1] != unit_shifts.shape[0]:
        raise ValueError(
            f"{edge_index.shape[1]} edges but {unit_shifts.shape[0]} unit shifts"
        )
    integral = np.rint(unit_shifts)
    if not np.array_equal(integral, unit_shifts):
        raise ValueError(
            "unit shifts must be integers; got non-integral values, which "
            "means these are displacement vectors -- use "
            "canonical_edges_from_shifts for those"
        )
    return sorted(
        (int(i), int(j), int(n_0), int(n_1), int(n_2))
        for (i, j), (n_0, n_1, n_2) in zip(edge_index.T, integral)
    )


def canonical_edges_from_shifts(
    edge_index: np.ndarray, shifts: np.ndarray, decimals: int = 6
) -> List[Tuple[int, int, float, float, float]]:
    """The same, for a source that reports displacement vectors only.

    Rounds, because two implementations sum the lattice vectors in different
    orders. Prefer :func:`canonical_edges` whenever integer shifts exist.
    """
    edge_index = np.asarray(edge_index).reshape(2, -1)
    shifts = np.asarray(shifts, dtype=float).reshape(-1, 3)
    return sorted(
        (int(i), int(j)) + tuple(round(float(v), decimals) for v in shift)
        for (i, j), shift in zip(edge_index.T, shifts)
    )


def assert_neighbourhoods_match(
    got: Iterable, expected: Iterable, context: str = ""
) -> None:
    """Compare two canonical edge lists and report the difference, not just
    that there is one -- an edge count mismatch on its own says nothing about
    which images went missing."""
    got, expected = list(got), list(expected)
    if got == expected:
        return
    missing = [edge for edge in expected if edge not in got]
    extra = [edge for edge in got if edge not in expected]
    where = f" ({context})" if context else ""
    raise AssertionError(
        f"neighbour lists differ{where}: {len(got)} edges against "
        f"{len(expected)} in the reference; "
        f"missing {missing[:8]}{' ...' if len(missing) > 8 else ''}, "
        f"unexpected {extra[:8]}{' ...' if len(extra) > 8 else ''}"
    )
