"""The reduced symmetric tensor-product basis, and the coefficients under it.

Imports neither ``e3nn`` nor ``cuequivariance``. The basis is model state, one
value for every device and every backend: nothing here branches on what happens
to be installed. That is the defect this package exists to remove, since on the
frozen tree a host without ``cuequivariance`` silently trains a different, more
heavily parametrized network for the same hyperparameters, 29 parameters
against 86 on the measured grid point.

The weight format, stated once
------------------------------

Four conventions are fixed here, and together they *are* the on-disk format of
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
sign fixed so the first structurally non-zero entry is positive. The norm is one
rule for every output irrep rather than ``sqrt(ir.dim)``, so weights on disk have
a comparable scale across irreps; the factor a given backend wants lives in its
converter, and an initializer that wants to match the legacy scale carries it
explicitly.

**Naming.** Every surviving path carries its
:class:`~mace_core.clebsch_gordan.reduced_basis.CouplingTree`, the sequence of
consumed input slices and running intermediate irreps that produced it. Order
and normalization alone do not pin the layout: the enumerated paths are
dependent, so *which* of them survives is a free choice, and two
implementations that resolve it differently span the same space with vectors no
reordering relates. Measured against ``cuequivariance``, four of the five grid
points the layout ticket uses agree up to a signed permutation and the fifth
needs two small dense blocks, of size 2 and 3, in the ``2e`` slot at body order
three. A label names a path wherever it sits, so a backend matches by name and
can say which trees it did not recognise instead of quietly reinterpreting the
weights.

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
    CouplingTree,
    full_path_labels,
    full_symmetric_tensor_product_basis,
    path_count,
    path_labels,
    reduced_symmetric_tensor_product_basis,
)

__all__ = [
    "CouplingTree",
    "Irrep",
    "Irreps",
    "IrrepsError",
    "clebsch_gordan",
    "from_canonical",
    "full_path_labels",
    "full_symmetric_tensor_product_basis",
    "full_to_reduced",
    "ir_mul_to_mul_ir",
    "mul_ir_to_ir_mul",
    "path_count",
    "path_labels",
    "real_basis_change",
    "reduced_symmetric_tensor_product_basis",
    "reduced_to_full",
    "to_canonical",
    "wigner_3j_complex",
    "wigner_3j_real",
]
