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

*Selection.* Which of the dependent paths survives is a free choice, and it is
the one that order and normalization do not pin. Two implementations that
enumerate differently span the same space with different vectors, and no
reordering relates them: measured against ``cuequivariance`` on the grid the
layout ticket uses, four of five points agree up to a signed permutation while
the fifth needs two small dense blocks, of size 2 and 3, in the ``2e`` slot at
body order three. So every surviving path carries its :class:`CouplingTree`,
the sequence of consumed input slices and running intermediate irreps that
produced it. A label names a path independently of where it sits, which is what
lets a backend match by name instead of by position and say which trees
disagreed when they do.

*Normalization.* Each surviving path is scaled to unit Frobenius norm, and its
sign fixed so that its first structurally non-zero entry is positive.

Verified against the anchor the ticket measures, ``irreps_in=0e+1o+2e+3o`` at
``correlation=3``: 13 paths for ``0e``, 16 for ``1o``, 20 for ``2e``, and so 29
for ``keep_ir=0e+1o``. Those are the numbers the legacy cueq-only path produces.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache

import numpy as np

from mace_core.clebsch_gordan.irreps import Irrep, Irreps
from mace_core.clebsch_gordan.real_basis import wigner_3j_real

__all__ = [
    "CouplingTree",
    "full_path_labels",
    "full_symmetric_tensor_product_basis",
    "path_count",
    "path_labels",
    "reduced_symmetric_tensor_product_basis",
    "symmetrize",
]


@dataclass(frozen=True, order=True)
class CouplingTree:
    """Which coupling path a basis vector came from.

    A path couples the input factors one at a time, outermost last. Step ``k``
    records the index of the input slice consumed at that step, counting the
    slices of ``irreps_in`` in the order
    :meth:`~mace_core.clebsch_gordan.irreps.Irreps.slices` yields them, and the
    running intermediate irrep that coupling produced. The last step's irrep is
    therefore the output irrep, and the number of steps is the body order.

    The label is the path's identity in the file format. Positions move when a
    backend segments the weights its own way; a label does not.

    Attributes:
        steps: One ``(slice_index, intermediate)`` pair per coupled factor.
    """

    steps: tuple[tuple[int, Irrep], ...]

    @property
    def correlation(self) -> int:
        """How many input factors the path couples, written ``nu`` in the papers."""
        return len(self.steps)

    @property
    def target(self) -> Irrep:
        """The output irrep the path lands on."""
        return self.steps[-1][1]

    @property
    def factors(self) -> tuple[int, ...]:
        """The consumed input slices, innermost first."""
        return tuple(index for index, _ in self.steps)

    def __str__(self) -> str:
        """``"0:0e|1:1o|2:2e"``: one ``slice:intermediate`` per step."""
        return "|".join(f"{index}:{irrep}" for index, irrep in self.steps)

    @classmethod
    def parse(cls, text: str) -> CouplingTree:
        """Read back what :meth:`__str__` writes.

        Raises:
            ValueError: If a step is not ``<slice index>:<irrep>``. The message
                quotes the offending step.
        """
        steps = []
        for piece in text.split("|"):
            index, _, irrep = piece.partition(":")
            if not _ or not index.strip().isdigit():
                raise ValueError(
                    f"{piece!r} is not a coupling step in {text!r}. Expected "
                    f"'<slice index>:<irrep>', for example '1:1o'."
                )
            steps.append((int(index), Irreps.parse(irrep.strip()).terms[0][1]))
        return cls(tuple(steps))


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

    Yields ``(tree, array)``, the array of shape ``(target.dimension,) +
    (irreps_in.dimension,) * correlation`` and the tree naming the path.
    """
    width = irreps_in.dimension
    if correlation == 1:
        for index, (piece, ir) in enumerate(irreps_in.slices()):
            if ir == target:
                path = np.zeros((ir.dimension, width))
                path[:, piece] = np.eye(ir.dimension)
                yield CouplingTree(((index, target),)), path
        return
    for intermediate in _reachable(irreps_in, correlation - 1):
        for tree, left in _paths(irreps_in, correlation - 1, intermediate):
            for index, (piece, ir) in enumerate(irreps_in.slices()):
                if target not in set(intermediate.couple(ir)):
                    continue
                coupling = wigner_3j_real(target.degree, intermediate.degree, ir.degree)
                path = np.zeros(
                    (target.dimension, *left.shape[1:], width), dtype=np.float64
                )
                path[..., piece] = np.einsum("oml,m...->o...l", coupling, left)
                yield CouplingTree((*tree.steps, (index, target))), path


def symmetrize(path: np.ndarray, correlation: int) -> np.ndarray:
    """Average a path over the permutations of its input axes.

    Axis 0 carries the output irrep and is held fixed; the remaining
    ``correlation`` axes are the interchangeable factors.
    """
    total = np.zeros_like(path)
    for order in itertools.permutations(range(1, correlation + 1)):
        total += np.transpose(path, (0, *order))
    return total


def _independent(rows: list[np.ndarray]) -> list[int]:
    """Modified Gram-Schmidt in order, returning which rows are independent.

    It returns positions rather than vectors for two reasons: the basis this
    package stores is the natural one and not the orthogonalized one, so the
    caller wants the row it passed in; and the surviving positions are what
    carry each path's label across the reduction.
    """
    kept: list[int] = []
    orthogonal: list[np.ndarray] = []
    for position, row in enumerate(rows):
        residual = row.astype(np.float64).copy()
        for direction in orthogonal:
            residual -= float(residual @ direction) * direction
        norm = float(np.linalg.norm(residual))
        if norm <= _INDEPENDENCE_TOLERANCE:
            continue
        orthogonal.append(residual / norm)
        kept.append(position)
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
def _basis_for(
    irreps_in_text: str, correlation: int, target_text: str
) -> tuple[np.ndarray, tuple[CouplingTree, ...]]:
    irreps_in = Irreps.parse(irreps_in_text)
    target = Irreps.parse(target_text).terms[0][1]
    enumerated = list(_paths(irreps_in, correlation, target))
    if not enumerated:
        shape = (0, target.dimension, *(irreps_in.dimension,) * correlation)
        return np.zeros(shape, dtype=np.float64), ()
    trees = [tree for tree, _ in enumerated]
    symmetrized = [symmetrize(path, correlation) for _, path in enumerated]
    kept = _independent([block.reshape(-1) for block in symmetrized])
    basis = np.stack([_canonical(symmetrized[position]) for position in kept])
    basis.flags.writeable = False
    return basis, tuple(trees[position] for position in kept)


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
        basis, _labels = _basis_for(text, correlation, str(ir))
        out[str(ir)] = basis.astype(dtype, copy=True)
    return out


def path_labels(
    irreps_in: str | Irreps, correlation: int, keep_ir: str | Irreps
) -> dict[str, tuple[CouplingTree, ...]]:
    """The coupling tree of every path in the reduced basis, in path order.

    One tuple per kept output irrep, aligned element for element with the
    leading axis of :func:`reduced_symmetric_tensor_product_basis`, so
    ``labels[ir][k]`` names the path whose weights sit at position ``k``.

    This is what a checkpoint records beside the weights. A backend that
    enumerates the same trees matches them by name and converts with a signed
    permutation; one that keeps a different subset of the dependent paths
    resolves the rest by a small solve, and can report which trees it did not
    recognise instead of quietly reinterpreting the weights.

    Args:
        irreps_in: The input irreps, as a declaration string or an
            :class:`~mace_core.clebsch_gordan.irreps.Irreps`.
        correlation: How many factors the symmetric product couples.
        keep_ir: Which output irreps to label.

    Returns:
        A mapping from each kept irrep's string form to its tuple of
        :class:`CouplingTree`.
    """
    text = str(irreps_in if isinstance(irreps_in, Irreps) else Irreps.parse(irreps_in))
    wanted = keep_ir if isinstance(keep_ir, Irreps) else Irreps.parse(keep_ir)
    return {str(ir): _basis_for(text, correlation, str(ir))[1] for _, ir in wanted}


def path_count(irreps_in: str | Irreps, correlation: int, keep_ir: str | Irreps) -> int:
    """How many trainable paths the basis carries, summed over the kept irreps.

    This is the number that changed with the host on the frozen tree, which is
    why it has a golden of its own.
    """
    basis = reduced_symmetric_tensor_product_basis(irreps_in, correlation, keep_ir)
    return sum(int(array.shape[0]) for array in basis.values())


@cache
def _full_basis_for(
    irreps_in_text: str, correlation: int, target_text: str
) -> tuple[np.ndarray, tuple[CouplingTree, ...]]:
    irreps_in = Irreps.parse(irreps_in_text)
    target = Irreps.parse(target_text).terms[0][1]
    enumerated = list(_paths(irreps_in, correlation, target))
    if not enumerated:
        shape = (0, target.dimension, *(irreps_in.dimension,) * correlation)
        return np.zeros(shape, dtype=np.float64), ()
    trees = [tree for tree, _ in enumerated]
    paths = [path for _, path in enumerated]
    kept = _independent([path.reshape(-1) for path in paths])
    basis = np.stack([_canonical(paths[position]) for position in kept])
    basis.flags.writeable = False
    return basis, tuple(trees[position] for position in kept)


def full_symmetric_tensor_product_basis(
    irreps_in: str | Irreps,
    correlation: int,
    keep_ir: str | Irreps,
    dtype: str = "float64",
) -> dict[str, np.ndarray]:
    """The unreduced basis: every coupling path, before the symmetry is used.

    Same enumeration, same order and same per-path normalization as
    :func:`reduced_symmetric_tensor_product_basis`; what it skips is the step
    that removes the paths the permutation symmetry makes redundant.

    It exists for one reason. The typical checkpoint in the wild carries this
    basis, because the legacy command line defaults the reduced flag to false,
    so converting a full-basis artifact is the common path and not the exotic
    one. Its extra directions are pure gauge: they span a larger space but
    produce the same functions on symmetric inputs.

    On the anchor ``0e+1o+2e+3o`` at correlation 3 summed over body orders, it
    is 28 paths for ``0e``, 58 for ``1o`` and 73 for ``2e``, against 13, 16 and
    20 reduced. The 86 of ``keep_ir=0e+1o`` against 29 is the gap that used to
    open and close with what was installed.
    """
    if correlation < 1:
        raise ValueError(f"correlation must be at least 1, got {correlation}")
    text = str(irreps_in if isinstance(irreps_in, Irreps) else Irreps.parse(irreps_in))
    wanted = keep_ir if isinstance(keep_ir, Irreps) else Irreps.parse(keep_ir)
    return {
        str(ir): _full_basis_for(text, correlation, str(ir))[0].astype(dtype, copy=True)
        for _, ir in wanted
    }


def full_path_labels(
    irreps_in: str | Irreps, correlation: int, keep_ir: str | Irreps
) -> dict[str, tuple[CouplingTree, ...]]:
    """The coupling tree of every path in the unreduced basis, in path order.

    The counterpart of :func:`path_labels` for
    :func:`full_symmetric_tensor_product_basis`. The reduced labels are a
    subsequence of these, which is what makes the projection between the two
    bases readable: a reduced path keeps its own name rather than acquiring a
    new index.
    """
    text = str(irreps_in if isinstance(irreps_in, Irreps) else Irreps.parse(irreps_in))
    wanted = keep_ir if isinstance(keep_ir, Irreps) else Irreps.parse(keep_ir)
    return {str(ir): _full_basis_for(text, correlation, str(ir))[1] for _, ir in wanted}
