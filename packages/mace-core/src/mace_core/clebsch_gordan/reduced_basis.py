"""The reduced symmetric tensor-product basis, and the unreduced one beside it.

The path order, the reduction, the path selection and the per-path
normalization implemented here are the on-disk weight format. They are stated
once, in the docstring of :mod:`mace_core.clebsch_gordan`, together with the
measured anchor; this module implements them and does not restate them.

A call builds the basis of exactly one body order. A model of correlation
``nu`` builds one per body order ``1 .. nu`` and concatenates their weights in
ascending body order (:func:`~mace_core.clebsch_gordan.conversion.to_canonical`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache

import numpy as np

from mace_core.clebsch_gordan.irreps import Irrep, Irreps, IrrepsError, _is_integer
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

    A path couples the input slices one at a time, and the steps are recorded
    in that order: step 0 takes one slice as it is, and every later step
    couples the running irrep with one more slice into a new running irrep.
    Each step records the index of the slice it consumed, counting the slices
    of ``irreps_in`` in the order
    :meth:`~mace_core.clebsch_gordan.irreps.Irreps.slices` yields them, and
    the running irrep it produced. The last step's irrep is therefore the
    output irrep, and the number of steps is the body order.

    The label is the path's identity in the file format. Positions move when a
    backend segments the weights its own way; a label does not.

    Attributes:
        steps: One ``(slice_index, intermediate)`` pair per coupled factor, in
            coupling order. At least one.

    Raises:
        ValueError: If ``steps`` is empty or a step is not a pair of a
            non-negative integer and an
            :class:`~mace_core.clebsch_gordan.irreps.Irrep`.
    """

    steps: tuple[tuple[int, Irrep], ...]

    def __post_init__(self) -> None:
        steps = tuple(tuple(step) for step in self.steps)
        if not steps:
            raise ValueError("a coupling tree needs at least one step")
        for step in steps:
            if (
                len(step) != 2
                or not _is_integer(step[0])
                or step[0] < 0
                or not isinstance(step[1], Irrep)
            ):
                raise ValueError(
                    f"{step!r} is not a coupling step. Expected a pair of a "
                    f"non-negative slice index and an Irrep, for example "
                    f"(1, Irrep(1, -1))."
                )
        object.__setattr__(
            self, "steps", tuple((int(index), irrep) for index, irrep in steps)
        )

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
        """The consumed input slices, in coupling order: first coupled first."""
        return tuple(index for index, _ in self.steps)

    def __str__(self) -> str:
        """``"0:0e|1:1o|2:2e"``: one ``slice:intermediate`` per step."""
        return "|".join(f"{index}:{irrep}" for index, irrep in self.steps)

    @classmethod
    def parse(cls, text: str) -> CouplingTree:
        """Read back exactly what :meth:`__str__` writes.

        Raises:
            ValueError: If a step is not ``<slice index>:<irrep>`` with a
                single irrep and no spaces. The message quotes the offending
                step, so ``"0:2x1o"`` and ``"0:0e+1o"`` are refused rather than
                read as ``0:1o`` and ``0:0e``.
        """
        steps = []
        for piece in text.split("|"):
            index, separator, irrep = piece.partition(":")
            if not separator or not (index.isascii() and index.isdigit()):
                raise ValueError(
                    f"{piece!r} is not a coupling step in {text!r}. Expected "
                    f"'<slice index>:<irrep>', for example '1:1o'."
                )
            try:
                intermediate = Irrep.parse(irrep)
            except IrrepsError as error:
                raise ValueError(
                    f"{piece!r} is not a coupling step in {text!r}: {error}"
                ) from error
            steps.append((int(index), intermediate))
        return cls(tuple(steps))


#: Below this, a symmetrized path is taken to lie in the span of the ones
#: before it. The gap either side of it is many orders of magnitude on every
#: grid point measured, so the exact value is not delicate.
_INDEPENDENCE_TOLERANCE = 1e-9


def _check_correlation(correlation: int) -> None:
    if not _is_integer(correlation) or correlation < 1:
        raise ValueError(
            f"correlation must be an integer of at least 1, got {correlation!r}. "
            f"It is the body order of one basis: a model of correlation 3 asks "
            f"for 1, 2 and 3 in turn."
        )


def _check_dtype(dtype: str) -> None:
    if dtype not in ("float64", "float32"):
        raise ValueError(
            f"dtype must be 'float64' or 'float32', got {dtype!r}. The basis is "
            f"always computed at float64; this only sets the returned type."
        )


def _input_text(irreps_in: str | Irreps) -> str:
    """The canonical spelling of ``irreps_in``, which is the cache key."""
    return str(irreps_in if isinstance(irreps_in, Irreps) else Irreps.parse(irreps_in))


def _output_irreps(keep_ir: str | Irreps) -> list[Irrep]:
    """The irreps ``keep_ir`` asks for, each once and without a multiplicity.

    Each kept irrep is one key of the returned mapping and carries its own
    weights, so a multiplicity has no meaning here and a repeated irrep would
    collapse into one key. Both are refused instead of silently merged.
    """
    wanted = keep_ir if isinstance(keep_ir, Irreps) else Irreps.parse(keep_ir)
    seen: list[Irrep] = []
    for multiplicity, irrep in wanted:
        if multiplicity != 1:
            raise ValueError(
                f"keep_ir {str(keep_ir)!r} gives {irrep} a multiplicity of "
                f"{multiplicity}. It names output irreps, each built once; write "
                f"{str(irrep)!r} instead."
            )
        if irrep in seen:
            raise ValueError(
                f"keep_ir {str(keep_ir)!r} lists {irrep} more than once. Each "
                f"output irrep is built once; list it a single time."
            )
        seen.append(irrep)
    return seen


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
    """Sum a path over the permutations of its input axes.

    Axis 0 carries the output irrep and is held fixed; the remaining
    ``correlation`` axes are the interchangeable factors. The result is the
    sum, not the average, so it is ``correlation!`` times the symmetric
    projection. The reduced basis renormalizes every path afterwards, so the
    factor does not reach it; a caller that needs the projection itself, as
    :mod:`~mace_core.clebsch_gordan.conversion` does, divides it out.
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


def _no_paths(
    irreps_in: Irreps, correlation: int, target: Irrep
) -> tuple[np.ndarray, tuple[CouplingTree, ...]]:
    """A basis of no paths, in the shape the caller expects one in.

    Two different situations reach it and neither is an error: the output irrep
    is not reachable at all, or it is reachable and everything reaching it is
    antisymmetric. Both mean the symmetric product does not carry it, and a
    caller iterating over output irreps wants a zero-path array rather than an
    exception from one slot of the loop.
    """
    shape = (0, target.dimension, *(irreps_in.dimension,) * correlation)
    empty = np.zeros(shape, dtype=np.float64)
    empty.flags.writeable = False
    return empty, ()


@cache
def _basis_for(
    irreps_in_text: str, correlation: int, target_text: str
) -> tuple[np.ndarray, tuple[CouplingTree, ...]]:
    irreps_in = Irreps.parse(irreps_in_text)
    target = Irrep.parse(target_text)
    enumerated = list(_paths(irreps_in, correlation, target))
    empty = _no_paths(irreps_in, correlation, target)
    if not enumerated:
        return empty
    trees = [tree for tree, _ in enumerated]
    symmetrized = [symmetrize(path, correlation) for _, path in enumerated]
    kept = _independent([block.reshape(-1) for block in symmetrized])
    if not kept:
        # Paths that exist and vanish under symmetrization. An irrep can be
        # reachable and still be carried entirely by the antisymmetric part,
        # `1o x 1o -> 1e` being the cross product, and a symmetric product has
        # none of it. That is a basis of no paths, not a failure: a caller
        # asking for such an output irrep has asked for something the symmetric
        # product does not contain, and gets an answer saying so.
        return empty
    basis = np.stack([_canonical(symmetrized[position]) for position in kept])
    basis.flags.writeable = False
    return basis, tuple(trees[position] for position in kept)


def reduced_symmetric_tensor_product_basis(
    irreps_in: str | Irreps,
    correlation: int,
    keep_ir: str | Irreps,
    dtype: str = "float64",
) -> dict[str, np.ndarray]:
    """The reduced basis of one body order, one array per kept output irrep.

    Args:
        irreps_in: The input irreps, as a declaration string or an
            :class:`~mace_core.clebsch_gordan.irreps.Irreps`. Multiplicities are
            part of the declaration and widen the basis accordingly.
        correlation: The body order of this basis, written ``nu`` in the
            papers: exactly how many factors the symmetric product couples. A
            model of correlation ``nu`` calls this once per body order
            ``1 .. nu``.
        keep_ir: Which output irreps to build, each once and without a
            multiplicity, such as ``"0e+1o"``. Each is returned separately,
            because each carries its own weights.
        dtype: ``"float64"`` or ``"float32"``. The basis is always *computed* at
            float64, since it is build-time data and its cost is paid once; this
            only sets what is handed back.

    Returns:
        A mapping from each kept irrep's string form to an array of shape
        ``(n_paths, ir.dimension) + (irreps_in.dimension,) * correlation``. The
        leading axis is the path axis, in the pinned order, and is the axis the
        weights multiply. An irrep the symmetric product does not carry at this
        body order gets ``n_paths == 0``.

    Raises:
        ValueError: If ``correlation`` is not an integer of at least 1,
            ``dtype`` is neither supported name, or ``keep_ir`` gives an irrep a
            multiplicity or lists it twice.
        IrrepsError: If either declaration is malformed.
    """
    _check_correlation(correlation)
    _check_dtype(dtype)
    text = _input_text(irreps_in)
    return {
        str(ir): _basis_for(text, correlation, str(ir))[0].astype(dtype, copy=True)
        for ir in _output_irreps(keep_ir)
    }


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
        correlation: The body order, an integer of at least 1.
        keep_ir: Which output irreps to label, each once.

    Returns:
        A mapping from each kept irrep's string form to its tuple of
        :class:`CouplingTree`.

    Raises:
        ValueError: As :func:`reduced_symmetric_tensor_product_basis`.
    """
    _check_correlation(correlation)
    text = _input_text(irreps_in)
    return {
        str(ir): _basis_for(text, correlation, str(ir))[1]
        for ir in _output_irreps(keep_ir)
    }


def path_count(irreps_in: str | Irreps, correlation: int, keep_ir: str | Irreps) -> int:
    """How many trainable paths the basis of one body order carries.

    Summed over the kept irreps, at exactly ``correlation``; a model of
    correlation ``nu`` carries the sum of this over body orders ``1 .. nu``.
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
    target = Irrep.parse(target_text)
    enumerated = list(_paths(irreps_in, correlation, target))
    empty = _no_paths(irreps_in, correlation, target)
    if not enumerated:
        return empty
    trees = [tree for tree, _ in enumerated]
    paths = [path for _, path in enumerated]
    kept = _independent([path.reshape(-1) for path in paths])
    if not kept:
        return empty
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

    Same enumeration, same order, same arguments, same validation and same
    per-path normalization as :func:`reduced_symmetric_tensor_product_basis`;
    what it skips is the step that removes the paths the permutation symmetry
    makes redundant.

    It exists for one reason. The typical checkpoint in the wild carries this
    basis, because the legacy command line defaults the reduced flag to false,
    so converting a full-basis artifact is the common path and not the exotic
    one. Its extra directions are pure gauge: they span a larger space but
    produce the same functions on symmetric inputs. The path counts on the
    measured anchor are in :mod:`mace_core.clebsch_gordan`.

    Raises:
        ValueError: As :func:`reduced_symmetric_tensor_product_basis`.
    """
    _check_correlation(correlation)
    _check_dtype(dtype)
    text = _input_text(irreps_in)
    return {
        str(ir): _full_basis_for(text, correlation, str(ir))[0].astype(dtype, copy=True)
        for ir in _output_irreps(keep_ir)
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

    Raises:
        ValueError: As :func:`reduced_symmetric_tensor_product_basis`.
    """
    _check_correlation(correlation)
    text = _input_text(irreps_in)
    return {
        str(ir): _full_basis_for(text, correlation, str(ir))[1]
        for ir in _output_irreps(keep_ir)
    }
