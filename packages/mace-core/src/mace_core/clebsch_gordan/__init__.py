"""The reduced symmetric tensor-product basis, and the coefficients under it.

Imports neither ``e3nn`` nor ``cuequivariance``. The basis is model state, one
value for every device and every backend: nothing here branches on what happens
to be installed. That is the defect this package exists to remove, since on the
frozen tree a host without ``cuequivariance`` silently trains a different, more
heavily parametrized network for the same hyperparameters, 29 parameters
against 86 on the measured grid point.

The weight format, stated once
------------------------------

Three conventions are fixed here, and together they *are* the on-disk format of
the symmetric-contraction weights. Changing any of them changes every
checkpoint, so each is chosen deliberately and written down rather than left to
whatever a library happened to do.

**Order.** Paths are enumerated by coupling one input factor at a time,
outermost first, over the total order on
:class:`~mace_core.clebsch_gordan.irreps.Irrep`: ascending degree, even parity
before odd at equal degree. Input slices are taken in the order the declaration
writes them. The enumeration is a pure function of
``(irreps_in, correlation, keep_ir)``.

**Reduction.** The enumerated paths are linearly dependent, because the
symmetric product is invariant under permuting its factors. They are
symmetrized and then reduced by a modified Gram-Schmidt *in enumeration order*.
An SVD would give the same span with an arbitrary basis inside it and a sign
that moves between LAPACK builds, which is not a file format.

**Normalization.** Each surviving path carries unit Frobenius norm, with its
sign fixed so the first structurally non-zero entry is positive.

**Layout** is ``mul_ir``: the multiplicity index varies slowest.

The real basis
--------------

Components run ``m = -l .. +l`` and the real combinations are the textbook
ones; see :mod:`mace_core.clebsch_gordan.real_basis`. This is not e3nn's basis,
and the difference is not a relabelling: they agree up to a signed permutation
at l = 0 and l = 1, and at l = 2 the transformation mixes m = 0 with m = +2
through a rotation. The difference is gauge, since the weights multiplying the
basis are learned, and the converter absorbs it exactly as it already absorbs
the node embedding's factor of sqrt(num_elements).

Verified against the anchor at ``irreps_in=0e+1o+2e+3o``, correlation 3: 13
paths for ``0e``, 16 for ``1o``, 20 for ``2e``, so 29 for ``keep_ir=0e+1o``,
against 28, 58, 73 and 86 for the unreduced basis. Those are the numbers the
legacy cueq-only path produces.
"""

from mace_core.clebsch_gordan.coefficients import clebsch_gordan, wigner_3j_complex
from mace_core.clebsch_gordan.conversion import (
    from_canonical,
    full_to_reduced,
    ir_mul_to_mul_ir,
    mul_ir_to_ir_mul,
    reduced_to_full,
    to_canonical,
)
from mace_core.clebsch_gordan.irreps import Irrep, Irreps, IrrepsError
from mace_core.clebsch_gordan.real_basis import real_basis_change, wigner_3j_real
from mace_core.clebsch_gordan.reduced_basis import (
    full_symmetric_tensor_product_basis,
    path_count,
    reduced_symmetric_tensor_product_basis,
)

__all__ = [
    "Irrep",
    "Irreps",
    "IrrepsError",
    "clebsch_gordan",
    "from_canonical",
    "full_symmetric_tensor_product_basis",
    "full_to_reduced",
    "ir_mul_to_mul_ir",
    "mul_ir_to_ir_mul",
    "path_count",
    "real_basis_change",
    "reduced_symmetric_tensor_product_basis",
    "reduced_to_full",
    "to_canonical",
    "wigner_3j_complex",
    "wigner_3j_real",
]
