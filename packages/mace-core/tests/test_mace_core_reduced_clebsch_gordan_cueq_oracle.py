"""The second oracle: cuequivariance, compared on spans rather than on entries.

`cuequivariance` is a **test-only** dependency here. `mace_core` must never
import it, and `tests/architecture/test_no_vendor_deps.py` asserts that. This
file imports it precisely because it is the independent derivation the basis is
checked against, and it is numpy-level descriptor mathematics that runs with no
GPU present.

**Spans, not entries.** The real basis is the one stated in
`mace_core.clebsch_gordan.real_basis`, and cuequivariance's O3 tables agree
with it up to a normalization and one sign per triple. What makes the entries
differ is the path selection: the enumerated paths are linearly dependent, the
two libraries keep different subsets of them, and the package docstring of
`mace_core.clebsch_gordan` records how far apart those subsets are. Neither
choice changes the subspace, so what is asserted here is the subspace and its
dimension: same rank, same span, no direction in one that the other cannot
reach. The choice this package makes is pinned on its own, in
`test_canonical_layout_written_out.py`.
"""

import numpy as np
import pytest
from mace_core.clebsch_gordan.real_basis import wigner_3j_real
from mace_core.clebsch_gordan.reduced_basis import (
    reduced_symmetric_tensor_product_basis,
)

# `importorskip` rather than a guard plus a plain import: the lint job installs
# the packages and the toolchain and nothing else, so a module-level import of
# an optional oracle is an unresolved import there even though the skip means
# it never runs.
cue = pytest.importorskip(
    "cuequivariance", reason="needs cuequivariance as a second oracle"
)

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


@pytest.mark.parametrize(
    ("l1", "l2", "l3"),
    [
        (l1, l2, l3)
        for l1 in range(4)
        for l2 in range(4)
        for l3 in range(abs(l1 - l2), min(l1 + l2, 3) + 1)
    ],
)
def test_the_3j_tables_agree_up_to_a_scale_and_a_sign(l1, l2, l3):
    """What the docstring above relies on: the entries differ through the path
    selection, not through a different real basis underneath."""
    (theirs,) = np.asarray(
        cue.O3.clebsch_gordan(
            cue.O3(l1, (-1) ** l1),
            cue.O3(l2, (-1) ** l2),
            cue.O3(l3, (-1) ** (l1 + l2)),
        ),
        dtype=np.float64,
    )
    theirs = theirs / np.linalg.norm(theirs)
    mine = wigner_3j_real(l1, l2, l3)
    assert min(np.abs(mine - theirs).max(), np.abs(mine + theirs).max()) < 1e-12
