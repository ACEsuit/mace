"""Reading a trained legacy block's weights into the rewrite's canonical layout.

The frozen tree stores an equivariant linear as e3nn does, and the rewrite
stores it differently. Both hold the same map; what differs is the order the
numbers are written in and where a normalization is applied.

Two differences, both measured against a trained anchor rather than read off
documentation:

**Order.** e3nn walks its instructions input-term major and lays each one out as
a ``[multiplicity_in, multiplicity_out]`` block. The rewrite enumerates output
copies major and input copies minor, over copies rather than terms. The map
between them is a transpose of each block followed by a re-walk.

**Normalization.** e3nn keeps the raw weight and multiplies by a per-instruction
factor inside the forward, ``1 / sqrt(fan_in)`` where the fan-in is the total
input multiplicity reaching that output irrep. Measured on the anchor: 1/sqrt(16)
where one term feeds an output, 1/sqrt(32) where two do and 1/sqrt(48) where
three do. The rewrite's linear has no hidden factor, so the conversion **folds
it into the weights**. That is the right side to put it on: an op with a silent
scale is an op whose stored weights do not mean what they say.
"""

from __future__ import annotations

import numpy as np

from mace_core.clebsch_gordan.irreps import Irreps

__all__ = ["linear_weights_to_canonical"]


def linear_weights_to_canonical(legacy_linear, irreps_in: str, irreps_out: str):
    """One ``o3.Linear``'s weights, in the rewrite's order and with no hidden scale.

    Args:
        legacy_linear: The trained module. Its ``weight`` and ``instructions``
            are read; nothing about it is mutated.
        irreps_in: Its input declaration.
        irreps_out: Its output declaration.

    Returns:
        A flat ``float64`` array in the order the rewrite's linear expects.
    """
    source = Irreps.parse(irreps_in)
    target = Irreps.parse(irreps_out)

    blocks: dict[tuple[int, int], np.ndarray] = {}
    offset = 0
    flat = legacy_linear.weight.detach().numpy().astype(float)
    for instruction in legacy_linear.instructions:
        rows, columns = instruction.path_shape
        span = rows * columns
        blocks[(instruction.i_in, instruction.i_out)] = (
            flat[offset : offset + span].reshape(rows, columns)
            * instruction.path_weight
        )
        offset += span
    if offset != flat.size:
        raise ValueError(
            f"the instructions account for {offset} weights and the module "
            f"holds {flat.size}. The layout assumed here does not match this "
            f"module."
        )

    ordered = []
    for out_index, (out_multiplicity, out_irrep) in enumerate(target.terms):
        for out_copy in range(out_multiplicity):
            for in_index, (in_multiplicity, in_irrep) in enumerate(source.terms):
                if in_irrep != out_irrep:
                    continue
                block = blocks[(in_index, out_index)]
                ordered.extend(
                    float(block[in_copy, out_copy])
                    for in_copy in range(in_multiplicity)
                )
    return np.asarray(ordered, dtype=float)
