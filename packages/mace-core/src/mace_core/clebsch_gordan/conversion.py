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

**Which full basis matters.** A trained model's weights are written against the
basis that model was built with. Two full bases of the same space can differ by
an order and a rotation inside it, and the frozen tree's does: it spans exactly
this package's space and is not this package's basis. So weights from an
artifact are converted against the artifact's own basis, passed as ``source``,
and a projection derived from this package's full basis alone would compute
something else.

**Why symmetrized.** The contraction feeds the same vector to every input slot,
so only the part of a path that is symmetric under permuting the slots computes
anything. Fitting the unsymmetrized tensor treats the slots as independent,
finds the full rank (23 for a scalar at body order three where the reduced
basis has 8) and leaves a residual of order one, which looks like a broken
projection rather than a wrong formulation. The residual of the symmetrized fit
is measured on every call and refused above a tolerance, since a reduced basis
short of a direction still produces weights of the right shape.
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
    "PROJECTION_TOLERANCE",
    "BasisConversionError",
    "from_canonical",
    "full_to_reduced",
    "ir_mul_to_mul_ir",
    "mul_ir_to_ir_mul",
    "reduced_to_full",
    "to_canonical",
]


#: The largest relative residual of a projection that still counts as exact.
#: The fit is an identity between two spans, so what it leaves is rounding, and
#: at float64 that is measured around 1e-15.
PROJECTION_TOLERANCE = 1e-14


class BasisConversionError(ValueError):
    """A conversion between bases that would not preserve the function."""


def _target_text(target: str | Irrep) -> str:
    """The canonical spelling of one output irrep, refusing a declaration."""
    return str(target if isinstance(target, Irrep) else Irrep.parse(target))

def _output_dtype(weights: np.ndarray) -> np.dtype:
    """The dtype a conversion hands back: the input's, if it is floating."""
    if np.issubdtype(weights.dtype, np.floating):
        return weights.dtype
    return np.dtype(np.float64)

def _projection(
    irreps_in: str | Irreps,
    correlation: int,
    target: str | Irrep,
    source: np.ndarray | None = None,
    tolerance: float = PROJECTION_TOLERANCE,
) -> np.ndarray:
    """The matrix carrying full-basis weights onto reduced-basis ones.

    Args:
        source: The full basis the weights are written against, path axis
            first: ``[paths, target dimension, d, ..., d]`` with
            ``correlation`` input axes. ``None`` is this package's own.

    Returns shape ``(n_full, n_reduced)``. Its transpose is the weight map: a
    full path contributes to a reduced one in proportion to how much of it
    survives symmetrization.

    Raises:
        BasisConversionError: If the source has the wrong shape, or the reduced
            basis does not reproduce its symmetric part, which is a basis short
            of a direction or two bases of different spaces.
    """
    target = _target_text(target)
    reduced = reduced_symmetric_tensor_product_basis(irreps_in, correlation, target)[
        target
    ]
    if source is None:
        full = full_symmetric_tensor_product_basis(irreps_in, correlation, target)[
            target
        ]
    else:
        full = np.asarray(source, dtype=np.float64)
        expected = reduced.shape[1:]
        if full.shape[1:] != expected:
            raise BasisConversionError(
                f"the source basis for {target!r} at correlation {correlation} "
                f"has shape {full.shape}, and a path over {irreps_in!r} has "
                f"shape {expected}: [target dimension, then one axis per input "
                f"slot]. Pass it with the path axis first."
            )
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
    scale = float(np.abs(symmetrized).max())
    residual = float(np.abs(flat_reduced.T @ solution - symmetrized.T).max())
    if scale > 0.0 and residual / scale > tolerance:
        raise BasisConversionError(
            f"the reduced basis for {target!r} at correlation {correlation} over "
            f"{irreps_in!r} does not reproduce the source basis: the relative "
            f"residual is {residual / scale:.3e} against {tolerance:.1e}. The "
            f"source has {full.shape[0]} paths and the reduced basis "
            f"{reduced.shape[0]}; converting would change what the model "
            f"computes."
        )
    return np.ascontiguousarray(solution.T)


def full_to_reduced(
    weights: np.ndarray,
    irreps_in: str | Irreps,
    correlation: int,
    target: str | Irrep,
    axis: int = -2,
    source: np.ndarray | None = None,
) -> np.ndarray:
    """Carry full-basis weights onto the reduced basis. Exact and unique.

    Args:
        weights: Any shape, with the path axis at ``axis``. The canonical
            ``[Z, A, mul]`` arrangement puts it at ``-2``, which is the default.
        irreps_in: The input irreps the basis was built over.
        correlation: The body order.
        target: The output irrep these weights belong to.
        axis: Which axis indexes the paths.
        source: The full basis the weights were trained against, path axis
            first. Required for weights from an artifact; see the module
            docstring for why this package's own basis is not a substitute.

    Returns:
        The same array with the path axis replaced by the reduced one.

    Raises:
        BasisConversionError: If the path axis does not have the length the
            full basis has, or the projection is not exact. The message gives
            both numbers, since the usual cause is weights built for a
            different correlation.
    """
    weights = np.asarray(weights)
    matrix = _projection(irreps_in, correlation, target, source)
    moved = np.moveaxis(weights, axis, -1).astype(np.float64)
    if moved.shape[-1] != matrix.shape[0]:
        raise BasisConversionError(
            f"the path axis has length {moved.shape[-1]}, but the full basis "
            f"for {target!r} at correlation {correlation} has "
            f"{matrix.shape[0]} paths."
        )
    return np.moveaxis(moved @ matrix, -1, axis).astype(_output_dtype(weights))


def reduced_to_full(
    weights: np.ndarray,
    irreps_in: str | Irreps,
    correlation: int,
    target: str | Irrep,
    axis: int = -2,
) -> np.ndarray:
    """Carry reduced-basis weights back onto the full basis. Export only.

    The map is under-determined, so this returns the minimum-norm
    representative: of all the full-basis weight vectors that produce the same
    function, the one closest to the origin. A checkpoint never takes this
    route, because every persisted v1 checkpoint is reduced.

    Arguments, dtype handling and errors are those of :func:`full_to_reduced`
    with the two bases exchanged.
    """
    weights = np.asarray(weights)
    matrix = _projection(irreps_in, correlation, target)
    moved = np.moveaxis(weights, axis, -1).astype(np.float64)
    if moved.shape[-1] != matrix.shape[1]:
        raise ValueError(
            f"the path axis has length {moved.shape[-1]}, but the reduced "
            f"basis for {str(target)!r} at correlation {correlation} has "
            f"{matrix.shape[1]} paths."
        )
    recovered = moved @ np.linalg.pinv(matrix)
    return np.moveaxis(recovered, -1, axis).astype(_output_dtype(weights))
