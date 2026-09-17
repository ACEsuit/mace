"""The canonical weight layout, which is what makes one checkpoint portable.

One layout means one checkpoint loads into any backend. The frozen tree has the
opposite: each backend stores weights its own way, and five conversion command
line tools exist to move between them.

The canonical form is stated here once:

* **Layout** is ``mul_ir``: the multiplicity index varies slowest.
* **Symmetric-contraction weights** are one flat ``[Z, A, mul]`` array over the
  reduced Clebsch-Gordan basis, in the path order
  :mod:`mace_core.clebsch_gordan` pins. ``Z`` is the element, ``A`` the path,
  ``mul`` the channel.
* **Conversion happens only at the checkpoint boundary.** ``to_canonical`` and
  ``load_canonical`` are called when saving and loading, never in ``forward``.
  A backend that wants another layout permutes once at build time and keeps the
  permuted copy.

The reference backend holds the canonical form directly, so for it both
functions are views.
"""

from __future__ import annotations

__all__ = ["CANONICAL_LAYOUT", "KERNEL_SPEC_VERSION", "canonical_weight_shape"]

#: The version of this contract. A backend records it, and a checkpoint carries
#: it, so a format change is a loud mismatch rather than a silent misread.
KERNEL_SPEC_VERSION = "1.0"

#: The one layout a checkpoint is ever written in.
CANONICAL_LAYOUT = "mul_ir"


def canonical_weight_shape(
    num_elements: int, path_count: int, num_features: int
) -> tuple[int, int, int]:
    """The shape of a symmetric-contraction weight array, ``[Z, A, mul]``.

    The legacy nested per-``(irrep_out, body order)`` tensors are contiguous
    slices of this one along the path axis, in the pinned order, so splitting
    and joining them is free.
    """
    return (num_elements, path_count, num_features)
