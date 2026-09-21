"""Projecting symmetric-contraction weights from the full basis onto the reduced one.

Every published MACE artifact, and both tiny anchors, were trained against the
**full** symmetric basis. The rewrite stores only the **reduced** one, so a
converter has to carry the weights across, and this is that step.

It works because the extra directions of the full basis are pure gauge: the
contraction feeds the same vector in at every slot, so two full-basis weight
vectors that differ only in an antisymmetric direction compute the same
function. The reduced basis keeps one representative per function, and the map
onto it is a least-squares solve.

Two things about the shape of the problem make this cheap and safe.

The conversion is **function-preserving, not weight-preserving**. There is no
sense in which a reduced weight "equals" a full one, so nothing here or in the
tests compares weights; what is compared is what the contraction computes.

The projection is **one small matrix per (body order, target irrep)**, derived
from the two bases alone. Weights are indexed ``[element, path, channel]`` and
only the path axis moves, so the same matrix serves every element and every
channel of a model.

**The source basis is an argument, not an assumption.** A trained model's
weights are written against the basis that model was built with, and two bases
of the same space are not interchangeable: they differ by an order and a
rotation inside it. Measured on the tiny anchor, the frozen tree's full basis
and this one span exactly the same space and are not the same basis, so a
projection derived from this one alone and applied to those weights computes
something else. Reading the source basis out of the artifact is therefore part
of the conversion, and it is not a verification against the frozen tree: what
the rank and the preserved function are checked against is stated in the tests.

The other direction is refused. Reduced to full is under-determined: it asks
which of infinitely many gauge representatives to invent.
"""

from __future__ import annotations

import numpy as np

from mace_core.clebsch_gordan.reduced_basis import (
    full_symmetric_tensor_product_basis,
    reduced_symmetric_tensor_product_basis,
)

__all__ = ["ProjectionError", "as_path_first", "project_weights", "projection_matrix"]

#: Directions sampled when identifying the projection. The system is
#: overdetermined by more than an order of magnitude at the sizes that occur, so
#: this is an identification and not a fit.
_SAMPLES = 400


class ProjectionError(RuntimeError):
    """Raised when a projection cannot be trusted rather than returning it."""


def _design(basis: np.ndarray, directions: np.ndarray, order: int) -> np.ndarray:
    """What each path computes on each sampled direction.

    Args:
        basis: ``[paths, components_out, d, ...]`` with ``order`` input axes.
        directions: ``[samples, d]``.
        order: The body order.

    Returns:
        ``[samples * components_out, paths]``.
    """
    letters = "abcde"[:order]
    inputs = ",".join(f"s{letter}" for letter in letters)
    subscripts = f"po{letters},{inputs}->sop"
    values = np.einsum(subscripts, basis, *([directions] * order))
    return values.reshape(-1, basis.shape[0])


def as_path_first(basis: np.ndarray, order: int) -> np.ndarray:
    """Put a basis in ``[paths, components_out, d, ...]`` whatever it arrived as.

    The frozen tree stores the path axis last, and drops the output-component
    axis entirely when the target is a scalar, which is the flattening that
    makes a scalar contraction look like one fewer dimension than it is.

    Args:
        basis: Either this package's layout or the frozen tree's.
        order: The body order, which fixes how many input axes there are.
    """
    if basis.ndim == order + 2 and basis.shape[0] != basis.shape[-1]:
        return basis
    moved = np.moveaxis(basis, -1, 0)
    if moved.ndim == order + 1:
        moved = moved[:, None]
    return np.ascontiguousarray(moved)


def projection_matrix(
    irreps_in: str,
    order: int,
    target: str,
    source: np.ndarray | None = None,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """The ``[reduced_paths, source_paths]`` map onto the reduced basis.

    Args:
        irreps_in: One channel's input declaration.
        order: The body order.
        target: The output irrep.
        source: The basis the weights are written against. ``None`` uses this
            package's own full basis, which is right only when the weights came
            from it. Weights from a trained artifact carry that artifact's
            basis, and it has to be passed.
        tolerance: The largest residual that still counts as exact. The
            projection is an identity between two spans, so its residual is
            rounding error; anything above this is a basis that does not span
            what it claims to.

    Raises:
        ProjectionError: If the reduced basis cannot reproduce the source. That
            is a statement about the bases, not about conditioning, so it is
            refused rather than returned with a warning.
    """
    full = (
        full_symmetric_tensor_product_basis(irreps_in, order, target)[target]
        if source is None
        else as_path_first(np.asarray(source, dtype=float), order)
    )
    reduced = reduced_symmetric_tensor_product_basis(irreps_in, order, target)[target]

    generator = np.random.default_rng(0)
    directions = generator.normal(size=(_SAMPLES, full.shape[-1]))
    design_full = _design(full, directions, order)
    design_reduced = _design(reduced, directions, order)

    matrix, *_ = np.linalg.lstsq(design_reduced, design_full, rcond=None)
    residue = float(np.abs(design_reduced @ matrix - design_full).max())
    if residue > tolerance:
        raise ProjectionError(
            f"the reduced basis for {irreps_in!r} at order {order} into "
            f"{target!r} does not reproduce the full basis: the largest "
            f"residual is {residue:.3e} against a tolerance of "
            f"{tolerance:.3e}. The full basis has {full.shape[0]} paths and "
            f"the reduced one {reduced.shape[0]}. Either the reduced basis is "
            f"missing a direction or the two are not bases of the same space."
        )
    return matrix


def project_weights(
    weights: np.ndarray,
    irreps_in: str,
    order: int,
    target: str,
    source: np.ndarray | None = None,
) -> np.ndarray:
    """Carry one contraction's weights onto the reduced basis.

    Args:
        weights: ``[elements, source_paths, channels]``, as the artifact stores
            them.
        irreps_in: One channel's input declaration.
        order: The body order.
        target: The output irrep.
        source: The basis those weights are written against. See
            :func:`projection_matrix`.

    Returns:
        ``[elements, reduced_paths, channels]``.
    """
    matrix = projection_matrix(irreps_in, order, target, source=source)
    if weights.shape[1] != matrix.shape[1]:
        raise ProjectionError(
            f"these weights have {weights.shape[1]} paths and the full basis "
            f"for {irreps_in!r} at order {order} into {target!r} has "
            f"{matrix.shape[1]}. The weights were trained against a different "
            f"basis than the one named, so converting them would be silently "
            f"wrong."
        )
    return np.einsum("rf,zfc->zrc", matrix, weights)
