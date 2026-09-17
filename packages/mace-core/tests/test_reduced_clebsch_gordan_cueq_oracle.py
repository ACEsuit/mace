"""The second oracle: cuequivariance, compared on spans rather than on entries.

`cuequivariance` is a **test-only** dependency here. `mace_core` must never
import it, and `tests/architecture/test_no_vendor_deps.py` asserts that. This
file imports it precisely because it is the independent derivation the basis is
checked against, and it is numpy-level descriptor mathematics that runs with no
GPU present.

**Spans, not entries.** The two bases describe the same subspace in different
real spherical-harmonic conventions: v1 uses the textbook one, m = -l..+l, and
e3nn, which is what cuequivariance's O3 delegates to, uses a different
orthogonal basis of the same space. They agree up to a signed permutation at
l = 0 and l = 1 and diverge at l = 2, where the transformation mixes m = 0 with
m = +2 through a rotation.

That difference is gauge. The weights multiplying the basis are learned, so
nothing observable depends on it, and the legacy converter absorbs it the same
way it already absorbs the node embedding's factor of sqrt(num_elements). What
must agree, and what is asserted here, is the subspace and its dimension:
same rank, same span, no direction in one that the other cannot reach.
"""

import importlib.util

import numpy as np
import pytest
from mace_core.clebsch_gordan.reduced_basis import (
    reduced_symmetric_tensor_product_basis,
)

if importlib.util.find_spec("cuequivariance") is None:  # pragma: no cover
    pytest.skip("needs cuequivariance as a second oracle", allow_module_level=True)

import cuequivariance as cue

pytestmark = pytest.mark.cueq

# Single multiplicities throughout, so mul_ir and ir_mul coincide on the input
# axes and the comparison is about the basis rather than about a layout.
GRID = [
    ("0e+1o", 2, "0e"),
    ("0e+1o", 2, "1o"),
    ("0e+1o+2e", 2, "2e"),
    ("0e+1o+2e", 3, "0e"),
    ("0e+1o+2e", 3, "1o"),
    ("0e+1o+2e+3o", 3, "0e"),
    ("0e+1o+2e+3o", 3, "1o"),
    ("0e+1o+2e+3o", 3, "2e"),
]
RANK_TOLERANCE = 1e-9


def cueq_basis(irreps: str, correlation: int, target: str) -> np.ndarray:
    """cuequivariance's basis, rearranged to this package's axis order."""
    basis = cue.reduced_symmetric_tensor_product_basis(
        cue.Irreps("O3", irreps),
        correlation,
        keep_ir=cue.Irreps("O3", target),
        layout=cue.ir_mul,
    )
    (segment,) = basis.segments
    array = np.asarray(segment, dtype=np.float64)
    # (D, ..., D, ir.dim, n_paths) -> (n_paths, ir.dim, D, ..., D)
    return np.ascontiguousarray(np.moveaxis(array, (-1, -2), (0, 1)))


@pytest.mark.parametrize(("irreps", "correlation", "target"), GRID)
def test_the_path_count_agrees_with_the_second_oracle(irreps, correlation, target):
    """The number that used to change with what was installed, now checked
    against the library it used to require."""
    mine = reduced_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    theirs = cueq_basis(irreps, correlation, target)
    assert mine.shape[0] == theirs.shape[0]
    assert mine.shape == theirs.shape


@pytest.mark.parametrize(("irreps", "correlation", "target"), GRID)
def test_the_two_bases_span_the_same_subspace(irreps, correlation, target):
    mine = reduced_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    theirs = cueq_basis(irreps, correlation, target)
    flat_mine = mine.reshape(mine.shape[0], -1)
    flat_theirs = theirs.reshape(theirs.shape[0], -1)

    rank_mine = np.linalg.matrix_rank(flat_mine, tol=RANK_TOLERANCE)
    rank_theirs = np.linalg.matrix_rank(flat_theirs, tol=RANK_TOLERANCE)
    rank_joined = np.linalg.matrix_rank(
        np.concatenate([flat_mine, flat_theirs]), tol=RANK_TOLERANCE
    )
    assert rank_mine == rank_theirs == rank_joined, (
        f"the two bases for {target!r} over {irreps!r} at correlation "
        f"{correlation} do not span the same subspace: this package has rank "
        f"{rank_mine}, cuequivariance has {rank_theirs}, and together they "
        f"reach {rank_joined}. A difference of convention cannot change the "
        f"span, so this is a real disagreement."
    )


@pytest.mark.parametrize(("irreps", "correlation", "target"), GRID)
def test_every_path_of_one_basis_is_reachable_from_the_other(
    irreps, correlation, target
):
    """Span equality by rank could in principle hide a degeneracy, so this
    solves for the change of basis and checks the residual directly."""
    mine = reduced_symmetric_tensor_product_basis(irreps, correlation, target)[target]
    theirs = cueq_basis(irreps, correlation, target)
    flat_mine = mine.reshape(mine.shape[0], -1)
    flat_theirs = theirs.reshape(theirs.shape[0], -1)
    change, *_ = np.linalg.lstsq(flat_mine.T, flat_theirs.T, rcond=None)
    residual = np.abs(flat_mine.T @ change - flat_theirs.T).max()
    assert residual < 1e-10, (
        f"cuequivariance's basis is not a linear combination of this one; "
        f"largest residual {residual:.3e}."
    )
