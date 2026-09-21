"""The map from canonical weights to a backend's own path order.

A backend is free to enumerate the symmetric-contraction paths its own way. It
is not free to reinterpret a checkpoint, so something has to carry the weights
across, once, when the model is built.

**The map is not a scaled permutation, and assuming it is loses weights
silently.** Order and normalization are pinned by
:mod:`mace_core.clebsch_gordan`, but the enumerated coupling paths are linearly
dependent, so which of them survives the reduction is a free choice that no
convention constrains. Two implementations that resolve it differently span the
same space with vectors related by a change of basis rather than a reordering.
Measured against ``cuequivariance`` on the grid the layout contract uses, four
of five points do come out as a signed permutation; the fifth,
``0e+1o+2e+3o -> 0e+1o+2e`` at body order three, decomposes into 44 blocks of
size one, one of size two and one of size three, all inside the ``2e`` slot.
Its worst block has a condition number of 2.79, so the general case costs a
small dense solve and nothing more.

What the structure buys, and why it is worth deriving rather than asserting: the
map always decomposes into independent blocks, and every block that a backend
shares with the canonical enumeration comes out at size one. A backend can
therefore report *which* coupling trees it disagreed about, instead of a
checkpoint quietly meaning something else.

This module derives the map from the two bases. It runs at build time, never in
a forward pass, and it imports no framework: a backend hands over its own basis
as a plain array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ReorderBlock", "ReorderError", "WeightReorder", "derive_reorder"]

#: Entries below this fraction of the map's largest entry are structural zeros.
#: The gap either side of it is many orders of magnitude on every grid point
#: measured, so the exact value is not delicate.
_STRUCTURAL_ZERO = 1e-9


class ReorderError(ValueError):
    """Two bases that a weight map cannot be derived between."""


@dataclass(frozen=True)
class ReorderBlock:
    """One independent piece of the map.

    Attributes:
        canonical: The canonical path positions this block covers.
        backend: The backend path positions it covers.
        matrix: Shape ``(len(canonical), len(backend))``. Row ``i`` expands
            canonical basis vector ``canonical[i]`` over the backend's vectors.
    """

    canonical: tuple[int, ...]
    backend: tuple[int, ...]
    matrix: np.ndarray

    @property
    def size(self) -> int:
        """How many paths the block couples together."""
        return len(self.canonical)


@dataclass(frozen=True)
class WeightReorder:
    """A canonical weight array carried into a backend's own path order.

    Weights transform against the basis, not with it: if a canonical vector
    expands as ``C_k = sum_j M[k, j] B_j``, then a weight on ``C_k`` lands on
    the backend paths as ``v = M.T @ w``. :meth:`apply` does that, and
    :meth:`inverse` gives the map back.

    Attributes:
        blocks: The independent pieces, ordered by their first canonical path.
        residual: How far ``M @ backend_basis`` sat from the canonical basis
            when the map was derived, in maximum absolute error. It is recorded
            rather than discarded so a caller can assert on it.
    """

    blocks: tuple[ReorderBlock, ...]
    residual: float

    @property
    def max_block_size(self) -> int:
        """The largest block. One means the map is a scaled permutation."""
        return max((block.size for block in self.blocks), default=0)

    @property
    def is_scaled_permutation(self) -> bool:
        """Whether every block is 1x1, so the map only reorders and rescales."""
        return self.max_block_size <= 1

    @property
    def path_count(self) -> int:
        """How many paths the map covers."""
        return sum(block.size for block in self.blocks)

    def block_sizes(self) -> dict[int, int]:
        """How many blocks of each size, for a golden to pin."""
        sizes: dict[int, int] = {}
        for block in self.blocks:
            sizes[block.size] = sizes.get(block.size, 0) + 1
        return dict(sorted(sizes.items()))

    def apply(self, weights: np.ndarray, axis: int = -2) -> np.ndarray:
        """Carry canonical weights onto the backend's paths.

        Args:
            weights: Any array with a path axis, typically ``[Z, A, mul]``.
            axis: Which axis is the path axis.

        Returns:
            An array of the same shape, with the path axis reordered and mixed.

        Raises:
            ReorderError: If the path axis is not the length the map covers.
        """
        moved = np.moveaxis(np.asarray(weights), axis, 0)
        if moved.shape[0] != self.path_count:
            raise ReorderError(
                f"the weights have {moved.shape[0]} paths on axis {axis} and "
                f"this map covers {self.path_count}. Check that the map was "
                f"derived for this descriptor."
            )
        out = np.zeros_like(moved, dtype=np.result_type(moved.dtype, np.float64))
        for block in self.blocks:
            taken = moved[list(block.canonical)]
            out[list(block.backend)] = np.tensordot(block.matrix.T, taken, axes=1)
        return np.moveaxis(out, 0, axis)

    def inverse(self) -> WeightReorder:
        """The map back, block by block.

        Raises:
            ReorderError: If a block is singular, which means the two bases do
                not in fact span the same space.
        """
        inverted = []
        for block in self.blocks:
            try:
                matrix = np.linalg.inv(block.matrix)
            except np.linalg.LinAlgError as error:
                raise ReorderError(
                    f"the block on canonical paths {block.canonical} is "
                    f"singular, so the two bases do not span the same space."
                ) from error
            inverted.append(ReorderBlock(block.backend, block.canonical, matrix))
        return WeightReorder(tuple(inverted), self.residual)


def _components(pattern: np.ndarray) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Group rows and columns that the sparsity pattern ties together."""
    rows = pattern.shape[0]
    parent = list(range(rows + pattern.shape[1]))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for row, column in zip(*np.nonzero(pattern), strict=True):
        left, right = find(int(row)), find(rows + int(column))
        if left != right:
            parent[left] = right

    grouped: dict[int, tuple[list[int], list[int]]] = {}
    for row in range(rows):
        grouped.setdefault(find(row), ([], []))[0].append(row)
    for column in range(pattern.shape[1]):
        grouped.setdefault(find(rows + column), ([], []))[1].append(column)
    return [
        (tuple(group_rows), tuple(group_columns))
        for group_rows, group_columns in sorted(
            grouped.values(), key=lambda pair: (pair[0] or [rows], pair[1])
        )
    ]


def derive_reorder(
    canonical_basis: np.ndarray,
    backend_basis: np.ndarray,
    tolerance: float = 1e-10,
) -> WeightReorder:
    """Derive the weight map between two bases of the same space.

    Both arrays carry one basis vector per leading index and are flattened over
    everything else, so their trailing shape has only to agree in total size.

    Args:
        canonical_basis: The canonical basis, path axis first.
        backend_basis: The backend's own basis, path axis first.
        tolerance: How far the reconstruction may sit from the canonical basis
            before the two are declared to span different spaces.

    Returns:
        The block-diagonal map, carrying its own reconstruction residual.

    Raises:
        ReorderError: If the two bases hold a different number of paths, or if
            the backend's basis cannot reproduce the canonical one within
            ``tolerance``. The second is the real failure: it means the backend
            is not computing the same function space, and loading a checkpoint
            into it would be a silent reinterpretation.
    """
    canonical = np.asarray(canonical_basis, dtype=np.float64)
    backend = np.asarray(backend_basis, dtype=np.float64)
    if canonical.shape[0] == 0 and backend.shape[0] == 0:
        # An output irrep unreachable at this body order. It still occupies its
        # slot in the canonical array, and reshaping a zero-sized array to
        # ``(0, -1)`` is ambiguous rather than empty, so it is answered here.
        return WeightReorder((), 0.0)
    canonical = canonical.reshape(canonical.shape[0], -1)
    backend = backend.reshape(backend.shape[0], -1)
    if canonical.shape[0] != backend.shape[0]:
        raise ReorderError(
            f"the canonical basis holds {canonical.shape[0]} paths and the "
            f"backend's holds {backend.shape[0]}. A weight map between them "
            f"would have to drop or invent a path."
        )
    if canonical.shape[1] != backend.shape[1]:
        raise ReorderError(
            f"the two bases are written over spaces of {canonical.shape[1]} and "
            f"{backend.shape[1]} components, so they are not two views of the "
            f"same operator."
        )
    matrix = np.linalg.lstsq(backend.T, canonical.T, rcond=None)[0].T
    residual = float(np.abs(matrix @ backend - canonical).max())
    if residual > tolerance:
        raise ReorderError(
            f"the backend's basis reproduces the canonical one only to "
            f"{residual:.3e}, against a tolerance of {tolerance:.3e}. The two "
            f"span different spaces, so no weight map exists and loading a "
            f"checkpoint into this backend would change what the model computes."
        )

    largest = float(np.abs(matrix).max())
    pattern = np.abs(matrix) > _STRUCTURAL_ZERO * max(largest, 1.0)
    blocks = tuple(
        ReorderBlock(rows, columns, matrix[np.ix_(list(rows), list(columns))])
        for rows, columns in _components(pattern)
        if rows and columns
    )
    covered = sum(block.size for block in blocks)
    if covered != canonical.shape[0]:
        raise ReorderError(
            f"the map leaves {canonical.shape[0] - covered} canonical paths "
            f"unconnected, which means a canonical basis vector is zero in the "
            f"backend's basis."
        )
    return WeightReorder(blocks, residual)
