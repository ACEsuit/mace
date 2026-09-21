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

from fm00_projection import project_weights

from mace_core.clebsch_gordan.irreps import Irreps

__all__ = [
    "contraction_weights_to_canonical",
    "energy_constants_to_canonical",
    "fully_connected_tp_weights_to_canonical",
    "linear_weights_to_canonical",
]


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


def fully_connected_tp_weights_to_canonical(
    legacy_tp, irreps_in1: str, irreps_out: str, num_scalars: int
):
    """The skip connection's weights, in the rewrite's order and unscaled.

    The frozen tree's skip is a fully connected tensor product against the
    element attributes, which are scalars, so it is a per-element linear map.
    It stores one ``[multiplicity_in, elements, multiplicity_out]`` block per
    instruction; the rewrite stores ``[elements, plan]`` with the plan running
    output copies major.

    The scale works out simpler than either side states it. e3nn's
    per-instruction factor is ``sqrt(dim_out / (mul_in1 * mul_in2))`` and the
    coupling of an irrep with a scalar carries ``1 / sqrt(dim_out)``, so the
    output dimension cancels and what is left is ``1 / sqrt(mul_in1 * mul_in2)``.
    It is read off the module rather than rebuilt from that identity, and a test
    pins the two against each other.

    Args:
        legacy_tp: The trained module. Read, never mutated.
        irreps_in1: Its first input declaration.
        irreps_out: Its output declaration.
        num_scalars: How many element attributes, the second input's width.

    Returns:
        ``[num_scalars, plan_size]`` of ``float64``.
    """
    source = Irreps.parse(irreps_in1)
    target = Irreps.parse(irreps_out)

    blocks: dict[tuple[int, int], np.ndarray] = {}
    offset = 0
    flat = legacy_tp.weight.detach().numpy().astype(float)
    for instruction in legacy_tp.instructions:
        first, second, out = instruction.path_shape
        span = first * second * out
        scale = instruction.path_weight / np.sqrt(
            target.terms[instruction.i_out][1].dimension
        )
        blocks[(instruction.i_in1, instruction.i_out)] = (
            flat[offset : offset + span].reshape(first, second, out) * scale
        )
        offset += span
    if offset != flat.size:
        raise ValueError(
            f"the instructions account for {offset} weights and the module "
            f"holds {flat.size}."
        )

    columns = []
    for out_index, (out_multiplicity, out_irrep) in enumerate(target.terms):
        for out_copy in range(out_multiplicity):
            for in_index, (in_multiplicity, in_irrep) in enumerate(source.terms):
                if in_irrep != out_irrep:
                    continue
                block = blocks[(in_index, out_index)]
                for in_copy in range(in_multiplicity):
                    columns.append(block[in_copy, :, out_copy])
    return np.stack(columns, axis=1).astype(float)


def contraction_weights_to_canonical(
    legacy_contraction, irreps_in: str, target: str, correlation: int
):
    """One symmetric contraction's weights, per body order and reduced.

    The frozen tree stores the highest body order first, under its own name,
    and counts down through a list; the rewrite stores them ascending. It also
    keeps them against the full basis, which travels with the module as its
    coupling tables, so each order is projected onto the reduced basis on the
    way across.

    The evaluation differs and the function does not. The frozen tree folds the
    orders into a Horner cascade, multiplying a running result by the features
    once per step, while the rewrite sums each order separately. Measured on
    the anchor, the two agree to 1.3e-15: the cascade is a cheaper way to
    evaluate the same polynomial, not a different one.

    Args:
        legacy_contraction: The trained module. Read, never mutated.
        irreps_in: One channel's input declaration.
        target: The output irrep this contraction reads out.
        correlation: The highest body order.

    Returns:
        One ``[elements, reduced_paths, channels]`` array per body order,
        ascending.
    """
    by_order = {correlation: legacy_contraction.weights_max.detach().numpy()}
    for offset, weight in enumerate(legacy_contraction.weights):
        by_order[correlation - 1 - offset] = weight.detach().numpy()
    if sorted(by_order) != list(range(1, correlation + 1)):
        raise ValueError(
            f"this contraction carries body orders {sorted(by_order)} and a "
            f"correlation of {correlation} needs {list(range(1, correlation + 1))}."
        )

    converted = []
    for order in range(1, correlation + 1):
        basis = getattr(legacy_contraction, f"U_matrix_{order}").numpy()
        converted.append(
            project_weights(
                by_order[order].astype(float), irreps_in, order, target, source=basis
            )
        )
    return converted


def energy_constants_to_canonical(legacy_model, heads=("default",)):
    """The isolated-atom energies and the scale and shift, as they stand.

    A straight transfer, with no refit and no conversion: the frozen tree holds
    the same numbers in the same units. It is here because the transfer has to
    happen and because doing it by reflection at load time is how the three
    different defaults in the frozen tree leak into a converted model.

    Args:
        legacy_model: The trained model. Read, never mutated.
        heads: The head names to file the energies under.

    Returns:
        ``(values, scale, shift)`` where ``values`` is the mapping
        :class:`~mace_core.elements.ResolvedE0s` takes, and the other two are
        one float per head.
    """
    table = legacy_model.atomic_energies_fn.atomic_energies.detach().numpy()
    if table.ndim == 1:
        table = table[None, :]
    numbers = [int(z) for z in legacy_model.atomic_numbers.tolist()]
    if table.shape != (len(heads), len(numbers)):
        raise ValueError(
            f"the isolated-atom table is {table.shape} and {len(heads)} head(s) "
            f"over {len(numbers)} element(s) needs {(len(heads), len(numbers))}."
        )
    values = {
        head: {z: float(table[index][position]) for position, z in enumerate(numbers)}
        for index, head in enumerate(heads)
    }

    shift_block = getattr(legacy_model, "scale_shift", None)
    if shift_block is None:
        # The plain model has no scale-shift block at all: it is the identity.
        return values, (1.0,) * len(heads), (0.0,) * len(heads)
    scale = shift_block.scale.detach().reshape(-1).tolist()
    shift = shift_block.shift.detach().reshape(-1).tolist()
    return values, tuple(float(v) for v in scale), tuple(float(v) for v in shift)
