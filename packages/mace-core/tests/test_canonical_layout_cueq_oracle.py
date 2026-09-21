"""The canonical-to-backend weight map, derived against a real second basis.

`cuequivariance` is a **test-only** dependency. `mace_core` never imports it,
and `tests/architecture/test_no_vendor_deps.py` asserts that. It is here
because a map derived only against synthetic bases proves the algebra and not
the claim: the claim is that a checkpoint written in the canonical layout loads
into a backend that enumerates paths its own way.

**What the map absorbs, stated because the block sizes below are otherwise
over-read.** Two things differ between the two bases and the derivation cannot
separate them. One is the free choice this contract exists for: which of the
linearly dependent coupling paths each implementation keeps. The other is the
real spherical-harmonic convention, textbook `m = -l..+l` here against e3nn's,
which cuequivariance's O3 delegates to. Conjugating an equivariant tensor by a
rotation leaves it equivariant, so the two spans coincide and the map exists;
it is simply carrying both differences at once. So a block larger than one is
evidence that *something* is not a relabelling, not evidence about which of the
two caused it.

What is asserted, in order of what it buys: the two bases span the same space
at fp64, a weight vector carries across and comes back, and the block structure
is what it was measured to be. The third is a golden. A cuequivariance release
that changed its segmentation would move it, and that is exactly the event this
file exists to make loud rather than silent.
"""

import importlib.util

import numpy as np
import pytest
from mace_core.clebsch_gordan import reduced_symmetric_tensor_product_basis
from mace_core.kernels import derive_reorder

if importlib.util.find_spec("cuequivariance") is None:  # pragma: no cover
    pytest.skip("needs cuequivariance as a second basis", allow_module_level=True)

import cuequivariance as cue

pytestmark = pytest.mark.cueq

ATOL = 1e-12

#: ``(irreps_in, keep_ir, correlation)`` -> per output irrep, the block sizes
#: of the map, measured 2026-09-21 against cuequivariance 0.11.
MEASURED_BLOCKS = {
    ("0e+1o", "0e+1o", 3): {"0e": {1: 2}, "1o": {1: 2}},
    ("0e+1o+2e", "0e+1o", 3): {"0e": {1: 5}, "1o": {1: 3, 2: 1}},
    ("0e+1o+2e+3o", "0e+1o+2e", 3): {
        "0e": {1: 8},
        "1o": {1: 6, 2: 3},
        "2e": {1: 6, 2: 1, 3: 2},
    },
}


def _cueq_basis(irreps_in: str, correlation: int, keep_ir: str):
    """cuequivariance's own reduced basis, with the path axis brought to front."""
    basis = cue.reduced_symmetric_tensor_product_basis(
        cue.Irreps("O3", irreps_in),
        correlation,
        keep_ir=cue.Irreps("O3", keep_ir),
        layout=cue.ir_mul,
    )
    return {
        target: np.moveaxis(np.asarray(segment, dtype=np.float64), [-1, -2], [0, 1])
        for target, segment in zip(keep_ir.split("+"), basis.segments, strict=True)
    }


@pytest.mark.parametrize("case", sorted(MEASURED_BLOCKS))
def test_the_two_bases_span_the_same_space(case):
    """If they did not, no weight map would exist and a load would lie."""
    irreps_in, keep_ir, correlation = case
    theirs = _cueq_basis(irreps_in, correlation, keep_ir)
    mine = reduced_symmetric_tensor_product_basis(irreps_in, correlation, keep_ir)
    for target in theirs:
        reorder = derive_reorder(mine[target], theirs[target])
        assert reorder.residual < ATOL, target


@pytest.mark.parametrize("case", sorted(MEASURED_BLOCKS))
def test_a_weight_vector_carries_across_and_comes_back(case):
    """The only property a checkpoint actually needs: the same function."""
    irreps_in, keep_ir, correlation = case
    generator = np.random.default_rng(20250921)
    theirs = _cueq_basis(irreps_in, correlation, keep_ir)
    mine = reduced_symmetric_tensor_product_basis(irreps_in, correlation, keep_ir)
    for target, basis in mine.items():
        reorder = derive_reorder(basis, theirs[target])
        weights = generator.normal(size=(3, basis.shape[0], 4))
        carried = reorder.apply(weights)
        np.testing.assert_allclose(
            np.einsum("zam,a...->zm...", weights, basis),
            np.einsum("zam,a...->zm...", carried, theirs[target]),
            atol=ATOL,
        )
        np.testing.assert_allclose(reorder.inverse().apply(carried), weights, atol=ATOL)


@pytest.mark.parametrize("case", sorted(MEASURED_BLOCKS))
def test_the_block_structure_is_the_measured_one(case):
    irreps_in, keep_ir, correlation = case
    theirs = _cueq_basis(irreps_in, correlation, keep_ir)
    mine = reduced_symmetric_tensor_product_basis(irreps_in, correlation, keep_ir)
    measured = {
        target: derive_reorder(basis, theirs[target]).block_sizes()
        for target, basis in mine.items()
    }
    assert measured == MEASURED_BLOCKS[case]


def test_at_least_one_grid_point_is_not_a_scaled_permutation():
    """The guard against a vacuous suite.

    Every assertion above holds trivially if the two enumerations happen to
    agree everywhere. They do not, and this is what says so: were the whole
    grid to become 1x1 blocks, the map could be stored as a permutation and
    this contract would be over-engineered, which is a conclusion worth
    reaching deliberately rather than by the tests quietly going green.
    """
    irreps_in, keep_ir, correlation = "0e+1o+2e+3o", "0e+1o+2e", 3
    theirs = _cueq_basis(irreps_in, correlation, keep_ir)
    mine = reduced_symmetric_tensor_product_basis(irreps_in, correlation, keep_ir)
    reorders = [derive_reorder(mine[t], theirs[t]) for t in mine]
    assert not all(reorder.is_scaled_permutation for reorder in reorders)
    assert max(reorder.max_block_size for reorder in reorders) == 3
