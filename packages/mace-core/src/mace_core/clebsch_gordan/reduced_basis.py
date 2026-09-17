"""The reduced symmetric tensor-product basis.

One basis, every device, every backend. On the frozen tree this basis is
reachable only when ``cuequivariance`` happens to be installed, so a CPU or AMD
host silently trains a different and more heavily parametrized network for the
same hyperparameters: 29 parameters against 86 on the measured grid point.
Nothing here branches on what is installed, because a basis is model state and
not a property of the machine.

**The path order and the per-path normalization defined here are the on-disk
weight format.** Both are stated below, and both are chosen rather than
inherited:

*Order.* Paths are enumerated by coupling one input factor at a time, outermost
first, iterating intermediate irreps in the total order of
:class:`~mace_core.clebsch_gordan.irreps.Irrep` and input slices in the order
the declaration writes them. The enumeration is therefore a pure function of
``(irreps_in, correlation, keep_ir)``, with no dependence on dictionary
iteration or on floating-point comparisons.

*Reduction.* The symmetric product is invariant under permuting its factors, so
the enumerated paths are linearly dependent. They are symmetrized and then
reduced by a modified Gram-Schmidt **in enumeration order**, which is what makes
the surviving set deterministic: an SVD would give the same span with an
arbitrary basis inside it and a sign that moves between LAPACK versions.

*Normalization.* Each surviving path is scaled to unit Frobenius norm, and its
sign fixed so that its first structurally non-zero entry is positive.

Verified against the anchor the ticket measures, ``irreps_in=0e+1o+2e+3o`` at
``correlation=3``: 13 paths for ``0e``, 16 for ``1o``, 20 for ``2e``, and so 29
for ``keep_ir=0e+1o``. Those are the numbers the legacy cueq-only path produces.
"""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np

from mace_core.clebsch_gordan.irreps import Irrep, Irreps
from mace_core.clebsch_gordan.real_basis import wigner_3j_real

__all__ = ["path_count", "reduced_symmetric_tensor_product_basis"]

#: Below this, a symmetrized path is taken to lie in the span of the ones
#: before it. The gap either side of it is many orders of magnitude on every
#: grid point measured, so the exact value is not delicate.
_INDEPENDENCE_TOLERANCE = 1e-9


def _reachable(irreps_in: Irreps, correlation: int) -> list[Irrep]:
    """Every irrep reachable by coupling ``correlation`` factors, in order."""
    reachable = {ir for _, ir in irreps_in}
    for _ in range(correlation - 1):
        grown = set()
        for left in reachable:
            for _, right in irreps_in:
                grown.update(left.couple(right))
        reachable |= grown
    return sorted(reachable)


def _paths(irreps_in: Irreps, correlation: int, target: Irrep):
    """Every coupling path to ``target``, in the pinned enumeration order.

    Yields arrays of shape ``(target.dimension,) + (irreps_in.dimension,) *
    correlation``.
    """
    width = irreps_in.dimension
    if correlation == 1:
        for piece, ir in irreps_in.slices():
            if ir == target:
                path = np.zeros((ir.dimension, width))
                path[:, piece] = np.eye(ir.dimension)
                yield path
        return
    for intermediate in _reachable(irreps_in, correlation - 1):
        for left in _paths(irreps_in, correlation - 1, intermediate):
            for piece, ir in irreps_in.slices():
                if target not in set(intermediate.couple(ir)):
                    continue
                coupling = wigner_3j_real(target.degree, intermediate.degree, ir.degree)
                path = np.zeros(
                    (target.dimension, *left.shape[1:], width), dtype=np.float64
                )
                path[..., piece] = np.einsum("oml,m...->o...l", coupling, left)
                yield path


def _symmetrize(path: np.ndarray, correlation: int) -> np.ndarray:
    """Average a path over the permutations of its input axes.

    Axis 0 carries the output irrep and is held fixed; the remaining
    ``correlation`` axes are the interchangeable factors.
    """
    total = np.zeros_like(path)
    for order in itertools.permutations(range(1, correlation + 1)):
        total += np.transpose(path, (0, *order))
    return total


def _independent(rows: list[np.ndarray]) -> list[np.ndarray]:
    """Modified Gram-Schmidt in order, keeping what is independent.

    Returns the kept rows as they were given, not the orthogonalized ones: the
    basis this package stores is the natural one, and orthogonality is only the
    test for whether a path added anything.
    """
    kept: list[np.ndarray] = []
    orthogonal: list[np.ndarray] = []
    for row in rows:
        residual = row.astype(np.float64).copy()
        for direction in orthogonal:
            residual -= float(residual @ direction) * direction
        norm = float(np.linalg.norm(residual))
        if norm <= _INDEPENDENCE_TOLERANCE:
            continue
        orthogonal.append(residual / norm)
        kept.append(row)
    return kept


def _canonical(path: np.ndarray) -> np.ndarray:
    """Unit Frobenius norm, with the first non-zero entry positive."""
    norm = float(np.linalg.norm(path))
    scaled = path / norm
    flat = scaled.reshape(-1)
    leading = np.flatnonzero(np.abs(flat) > _INDEPENDENCE_TOLERANCE)
    if leading.size and flat[leading[0]] < 0:
        scaled = -scaled
    return np.ascontiguousarray(scaled)


@cache
def _basis_for(irreps_in_text: str, correlation: int, target_text: str) -> np.ndarray:
    irreps_in = Irreps.parse(irreps_in_text)
    target = Irreps.parse(target_text).terms[0][1]
    enumerated = list(_paths(irreps_in, correlation, target))
    if not enumerated:
        shape = (0, target.dimension, *(irreps_in.dimension,) * correlation)
        return np.zeros(shape, dtype=np.float64)
    symmetrized = [_symmetrize(p, correlation) for p in enumerated]
    flat = [s.reshape(-1) for s in symmetrized]
    kept = _independent(flat)
    basis = np.stack([_canonical(row.reshape(symmetrized[0].shape)) for row in kept])
    basis.flags.writeable = False
    return basis


def reduced_symmetric_tensor_product_basis(
    irreps_in: str | Irreps,
    correlation: int,
    keep_ir: str | Irreps,
    dtype: str = "float64",
) -> dict[str, np.ndarray]:
    """The reduced basis, one array per kept output irrep.

    Args:
        irreps_in: The input irreps, as a declaration string or an
            :class:`~mace_core.clebsch_gordan.irreps.Irreps`. Multiplicities are
            part of the declaration and widen the basis accordingly.
        correlation: How many factors the symmetric product couples. The body
            order, written as ``nu`` in the papers.
        keep_ir: Which output irreps to build. Each is returned separately,
            because each carries its own weights.
        dtype: ``"float64"`` or ``"float32"``. The basis is always *computed* at
            float64, since it is build-time data and its cost is paid once; this
            only sets what is handed back.

    Returns:
        A mapping from each kept irrep's string form to an array of shape
        ``(n_paths, ir.dimension) + (irreps_in.dimension,) * correlation``. The
        leading axis is the path axis, in the pinned order, and is the axis the
        weights multiply.

    Raises:
        ValueError: If ``correlation`` is below 1, or ``dtype`` is neither
            supported name.
    """
    if correlation < 1:
        raise ValueError(f"correlation must be at least 1, got {correlation}")
    if dtype not in ("float64", "float32"):
        raise ValueError(
            f"dtype must be 'float64' or 'float32', got {dtype!r}. The basis is "
            f"always computed at float64; this only sets the returned type."
        )
    text = str(irreps_in if isinstance(irreps_in, Irreps) else Irreps.parse(irreps_in))
    wanted = keep_ir if isinstance(keep_ir, Irreps) else Irreps.parse(keep_ir)
    out = {}
    for _, ir in wanted:
        basis = _basis_for(text, correlation, str(ir))
        out[str(ir)] = basis.astype(dtype, copy=True)
    return out


def path_count(irreps_in: str | Irreps, correlation: int, keep_ir: str | Irreps) -> int:
    """How many trainable paths the basis carries, summed over the kept irreps.

    This is the number that changed with the host on the frozen tree, which is
    why it has a golden of its own.
    """
    basis = reduced_symmetric_tensor_product_basis(irreps_in, correlation, keep_ir)
    return sum(int(array.shape[0]) for array in basis.values())
