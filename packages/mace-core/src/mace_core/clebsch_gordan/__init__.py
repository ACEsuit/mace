"""The reduced symmetric tensor-product basis, and the coefficients under it.

Imports neither ``e3nn`` nor ``cuequivariance``. The basis is model state, one
value for every device and every backend: nothing here branches on what happens
to be installed. That is the defect this package exists to remove, since on the
frozen tree a host without ``cuequivariance`` silently trains a different, more
heavily parametrized network for the same hyperparameters (the anchor below).

The weight format, stated once
------------------------------

Five conventions are fixed here, and together they *are* the on-disk format of
the symmetric-contraction weights. Changing any of them changes every
checkpoint, so each is chosen deliberately and written down rather than left to
whatever a library happened to do. :mod:`~mace_core.clebsch_gordan.reduced_basis`
implements them.

**Order.** A path couples the input slices one at a time: the first step takes
one slice as it is, and each later step couples the running irrep with one more
slice into a new running irrep, the last of which is the output irrep. Slices
are counted in the order the declaration writes them, each copy of a
multiplied term its own slice. The paths of body order ``nu`` to one output
irrep are sorted by, in decreasing priority:

1. the running irrep after ``nu - 1`` steps, in the total order on
   :class:`~mace_core.clebsch_gordan.irreps.Irrep` (ascending degree, even
   parity before odd at equal degree);
2. their first ``nu - 1`` steps, by this same rule applied recursively;
3. the index of the slice coupled at the last step.

The enumeration is a pure function of ``(irreps_in, body order, output
irrep)``, with no dependence on dictionary iteration or floating-point
comparisons.

**Reduction.** The enumerated paths are linearly dependent, because the
symmetric product is invariant under permuting its factors. They are
symmetrized and then reduced by a modified Gram-Schmidt *in enumeration order*.
An SVD would give the same span with an arbitrary basis inside it and a sign
that moves between LAPACK builds, which is not a file format.

**Selection.** Order and normalization alone do not pin the layout: the
enumerated paths are dependent, so *which* of them survives is a free choice,
and two implementations that resolve it differently span the same space with
vectors no reordering relates. Here a path survives when it is independent of
the ones enumerated before it. Measured against ``cuequivariance``, four of the
five grid points the layout ticket uses agree up to a signed permutation and
the fifth needs two small dense blocks, of size 2 and 3, in the ``2e`` slot at
body order three. So every surviving path carries its
:class:`~mace_core.clebsch_gordan.reduced_basis.CouplingTree`, the sequence of
consumed slices and running irreps that produced it, in coupling order. A
label names a path wherever it sits, so a backend matches by name and can say
which trees it did not recognise instead of quietly reinterpreting the weights.

**Normalization.** Each surviving path carries unit Frobenius norm, with its
sign fixed so the first structurally non-zero entry is positive. The norm is one
rule for every output irrep rather than ``sqrt(ir.dim)``, so weights on disk have
a comparable scale across irreps; the factor a given backend wants lives in its
converter, and an initializer that wants to match the legacy scale carries it
explicitly.

**Layout** is ``mul_ir``: the multiplicity index varies slowest.

The 3j tables underneath are in the real basis stated in
:mod:`mace_core.clebsch_gordan.real_basis`, which also states how they relate
to e3nn's.

The anchor
----------

At ``irreps_in=0e+1o+2e+3o`` the reduced basis of each single body order
carries

====== ===== ===== ===== =====
output nu=1  nu=2  nu=3  sum
====== ===== ===== ===== =====
``0e``   1     4     8    13
``1o``   1     3    12    16
``2e``   1     5    14    20
====== ===== ===== ===== =====

against 1, 4, 23 (28), 1, 6, 51 (58) and 1, 7, 65 (73) for the unreduced basis.
A model of correlation 3 builds all three body orders, so it carries the sums:
29 paths for ``keep_ir=0e+1o`` against 86 unreduced. Those sums are the numbers
the legacy cueq-only path produces, and the gap is what used to open and close
with what was installed.
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
