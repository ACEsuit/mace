"""Boundaries: between the two bases, and between the two layouts.

Two conversions live here and they are very different in character.

**Layout** is free. Canonical weights are one flat ``[Z, A, mul]`` array over
the reduced basis, in the pinned path order, and the per-body-order tensors a
model holds are contiguous slices of it. Going either way is a concatenate or a
split, and ``mul_ir`` to ``ir_mul`` is a reshape. Nothing is computed.

**Basis** is not free, and it is the common path rather than the exotic one.
The legacy command line defaults its reduced flag to false, so the typical
checkpoint in the wild carries the full basis, and loading one converts it.

The full basis spans a larger space than the reduced one, but the extra
directions are pure gauge: on symmetric inputs they produce no function the
reduced basis cannot. So ``full -> reduced`` is exact and unique, and
``reduced -> full`` is exact and under-determined, which is why it returns the
minimum-norm representative and is for export only. A checkpoint is always
written reduced.

Neither direction is bit-exact in weight space, and it cannot be: the map is a
projection. What it preserves is the function, so it is tested on values and
never on weight equality.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.reduced_basis import (
    full_symmetric_tensor_product_basis,
    reduced_symmetric_tensor_product_basis,
    symmetrize,
)

__all__ = [
    "from_canonical",
    "full_to_reduced",
    "ir_mul_to_mul_ir",
    "mul_ir_to_ir_mul",
    "reduced_to_full",
    "to_canonical",
]


def _projection(irreps_in: str | Irreps, correlation: int, target: str) -> np.ndarray:
    """The matrix carrying full-basis weights onto reduced-basis ones.

    Returns shape ``(n_full, n_reduced)``. Its transpose is the weight map: a
    full path contributes to a reduced one in proportion to how much of it
    survives symmetrization.
    """
    full = full_symmetric_tensor_product_basis(irreps_in, correlation, target)[target]
    reduced = reduced_symmetric_tensor_product_basis(irreps_in, correlation, target)[
        target
    ]
    if full.shape[0] == 0 or reduced.shape[0] == 0:
        return np.zeros((full.shape[0], reduced.shape[0]), dtype=np.float64)
    # symmetrize sums over the permutations rather than averaging, so it
    # carries a factor of correlation!. Dividing it out here is what makes the
    # conversion preserve the function outright, instead of preserving it up to
    # a scale a caller would have to know about. Measured: without this the
    # converted weights give exactly 3! times the original at correlation 3.
    symmetrized = np.stack(
        [
            symmetrize(path, correlation).reshape(-1) / math.factorial(correlation)
            for path in full
        ]
    )
    flat_reduced = reduced.reshape(reduced.shape[0], -1)
    solution, *_ = np.linalg.lstsq(flat_reduced.T, symmetrized.T, rcond=None)
    return np.ascontiguousarray(solution.T)


def full_to_reduced(
    weights: np.ndarray,
    irreps_in: str | Irreps,
    correlation: int,
    target: str,
    axis: int = -2,
) -> np.ndarray:
    """Carry full-basis weights onto the reduced basis. Exact and unique.

    Args:
        weights: Any shape, with the path axis at ``axis``. The canonical
            ``[Z, A, mul]`` arrangement puts it at ``-2``, which is the default.
        irreps_in: The input irreps the basis was built over.
        correlation: The body order.
        target: The output irrep these weights belong to.
        axis: Which axis indexes the paths.

    Returns:
        The same array with the path axis replaced by the reduced one.

    Raises:
        ValueError: If the path axis does not have the length the full basis
            has. The message gives both numbers, since the usual cause is
            weights built for a different correlation.
    """
    matrix = _projection(irreps_in, correlation, target)
    moved = np.moveaxis(weights, axis, -1)
    if moved.shape[-1] != matrix.shape[0]:
        raise ValueError(
            f"the path axis has length {moved.shape[-1]}, but the full basis "
            f"for {target!r} at correlation {correlation} has "
            f"{matrix.shape[0]} paths."
        )
    return np.moveaxis(moved @ matrix, -1, axis)


def reduced_to_full(
    weights: np.ndarray,
    irreps_in: str | Irreps,
    correlation: int,
    target: str,
    axis: int = -2,
) -> np.ndarray:
    """Carry reduced-basis weights back onto the full basis. Export only.

    The map is under-determined, so this returns the minimum-norm
    representative: of all the full-basis weight vectors that produce the same
    function, the one closest to the origin. A checkpoint never takes this
    route, because every persisted v1 checkpoint is reduced.
    """
    matrix = _projection(irreps_in, correlation, target)
    moved = np.moveaxis(weights, axis, -1)
    if moved.shape[-1] != matrix.shape[1]:
        raise ValueError(
            f"the path axis has length {moved.shape[-1]}, but the reduced "
            f"basis for {target!r} at correlation {correlation} has "
            f"{matrix.shape[1]} paths."
        )
    recovered = moved @ np.linalg.pinv(matrix)
    return np.moveaxis(recovered, -1, axis)


def to_canonical(per_order: Sequence[np.ndarray], axis: int = -2) -> np.ndarray:
    """Join the per-body-order weight tensors into one flat canonical array.

    Args:
        per_order: The tensors in ascending body order, which is the order the
            path enumeration uses and therefore the order on disk.
        axis: The path axis in each tensor.

    Returns:
        One array, the concatenation along the path axis.
    """
    if not per_order:
        raise ValueError("there are no weight tensors to join")
    return np.concatenate(list(per_order), axis=axis)


def from_canonical(
    flat: np.ndarray, counts: Sequence[int], axis: int = -2
) -> list[np.ndarray]:
    """Split a canonical array back into per-body-order tensors.

    Args:
        flat: The canonical array.
        counts: How many paths each body order carries, ascending.
        axis: The path axis.

    Raises:
        ValueError: If the counts do not add up to the path axis length. The
            message gives both totals, because the usual cause is a correlation
            mismatch and the shapes are otherwise plausible.
    """
    total = int(sum(counts))
    if flat.shape[axis] != total:
        raise ValueError(
            f"the path axis has length {flat.shape[axis]} but the counts "
            f"{list(counts)} add up to {total}."
        )
    edges = np.cumsum(list(counts))[:-1]
    return [np.ascontiguousarray(part) for part in np.split(flat, edges, axis=axis)]


def mul_ir_to_ir_mul(values: np.ndarray, irreps: str | Irreps) -> np.ndarray:
    """Reinterpret the last axis from ``mul_ir`` layout to ``ir_mul``.

    ``mul_ir`` is canonical and is what this package stores: the multiplicity
    index varies slowest, so a ``2x1o`` term is two contiguous blocks of three.
    ``ir_mul`` interleaves them instead. Some backends want the other one, and
    for a single term the change is a reshape and a transpose rather than a
    computation.

    Raises:
        ValueError: If the declaration has more than one term. With several
            terms the two layouts are not related by one reshape, and pretending
            otherwise would silently scramble the values.
    """
    parsed = irreps if isinstance(irreps, Irreps) else Irreps.parse(irreps)
    if len(parsed.terms) != 1:
        raise ValueError(
            f"the layout change is a reshape only for a single term, and "
            f"{parsed} has {len(parsed.terms)}. Convert the terms separately."
        )
    mul, ir = parsed.terms[0]
    head = values.shape[:-1]
    return np.ascontiguousarray(
        values.reshape(*head, mul, ir.dimension).swapaxes(-1, -2).reshape(*head, -1)
    )


def ir_mul_to_mul_ir(values: np.ndarray, irreps: str | Irreps) -> np.ndarray:
    """The inverse of :func:`mul_ir_to_ir_mul`."""
    parsed = irreps if isinstance(irreps, Irreps) else Irreps.parse(irreps)
    if len(parsed.terms) != 1:
        raise ValueError(
            f"the layout change is a reshape only for a single term, and "
            f"{parsed} has {len(parsed.terms)}. Convert the terms separately."
        )
    mul, ir = parsed.terms[0]
    head = values.shape[:-1]
    return np.ascontiguousarray(
        values.reshape(*head, ir.dimension, mul).swapaxes(-1, -2).reshape(*head, -1)
    )
