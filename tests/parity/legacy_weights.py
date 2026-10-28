"""Carrying legacy weights into the reference backend's canonical layout.

Shared by the parity suites that copy a frozen module's weights into a v1 one
and compare the two. Each map is derived from the legacy module's own
instruction list rather than from the reference's code, so a reference that
stored its weights in another order fails the comparison instead of agreeing
with itself.
"""

import itertools

import numpy as np
import torch
from e3nn import o3

from tests.golden.harness import tolerance

CLOSED_FORM_ATOL = tolerance("closed_form_fp64").atol


def canonical_linear_index(irreps_in: o3.Irreps, irreps_out: o3.Irreps) -> dict:
    """``(i_out, copy_out, i_in, copy_in) -> canonical position``.

    The canonical order as the layout states it: output copies outermost, and
    within one output copy the matching input copies in declaration order.
    """
    index, position = {}, 0
    for i_out, (multiplicity_out, irrep_out) in enumerate(irreps_out):
        for copy_out in range(multiplicity_out):
            for i_in, (multiplicity_in, irrep_in) in enumerate(irreps_in):
                if irrep_in != irrep_out:
                    continue
                for copy_in in range(multiplicity_in):
                    index[(i_out, copy_out, i_in, copy_in)] = position
                    position += 1
    return index


def linear_permutation(legacy: o3.Linear) -> torch.Tensor:
    """For each flat legacy weight, the canonical weight it is."""
    canonical = canonical_linear_index(legacy.irreps_in, legacy.irreps_out)
    order = []
    for instruction in legacy.instructions:
        if instruction.i_in == -1:
            continue
        multiplicity_in, multiplicity_out = instruction.path_shape
        for copy_in in range(multiplicity_in):
            for copy_out in range(multiplicity_out):
                order.append(
                    canonical[(instruction.i_out, copy_out, instruction.i_in, copy_in)]
                )
    return torch.tensor(order)


def skip_permutation(legacy: o3.FullyConnectedTensorProduct) -> torch.Tensor:
    """For each flat legacy weight, its flat position in ``[scalar, weight]``."""
    canonical = canonical_linear_index(legacy.irreps_in1, legacy.irreps_out)
    per_scalar = len(canonical)
    scalar_offsets = np.cumsum([0] + [mul for mul, _ in legacy.irreps_in2])
    order = []
    for instruction in legacy.instructions:
        multiplicity_1, multiplicity_2, multiplicity_out = instruction.path_shape
        for copy_1 in range(multiplicity_1):
            for copy_2 in range(multiplicity_2):
                scalar = int(scalar_offsets[instruction.i_in2]) + copy_2
                for copy_out in range(multiplicity_out):
                    weight = canonical[
                        (instruction.i_out, copy_out, instruction.i_in1, copy_1)
                    ]
                    order.append(scalar * per_scalar + weight)
    return torch.tensor(order)


def legacy_contraction_blocks(
    contraction,
) -> list[tuple[torch.Tensor, torch.nn.Parameter, int]]:
    """``(coupling matrix, weight, body order)`` for every legacy block."""
    blocks = [
        (
            contraction.U_tensors(contraction.correlation),
            contraction.weights_max,
            contraction.correlation,
        )
    ]
    for position, weight in enumerate(contraction.weights):
        order = contraction.correlation - 1 - position
        blocks.append((contraction.U_tensors(order), weight, order))
    return blocks


def symmetric_part(tensor: np.ndarray, order: int) -> np.ndarray:
    """Average over permutations of the ``order`` input axes after axis 0."""
    total = np.zeros_like(tensor)
    permutations = list(itertools.permutations(range(1, order + 1)))
    for permutation in permutations:
        total += np.transpose(tensor, (0, *permutation))
    return total / len(permutations)


def legacy_to_reference_map(
    coupling: torch.Tensor, basis: torch.Tensor, order: int
) -> np.ndarray:
    """``M[p, q]`` with ``sum_q M[p, q] basis_q = sym(coupling_p)``.

    Legacy contracts its coupling matrices with a symmetric power of the
    features, so only their symmetric part acts. The reference basis is
    symmetric already, so a legacy weight ``w_p`` is the reference weight
    ``sum_p M[p, q] w_p``.
    """
    legacy = coupling.detach().numpy()
    if legacy.ndim == order + 1:  # a scalar output carries no output axis
        legacy = legacy[None]
    legacy = np.moveaxis(legacy, -1, 0)
    targets = np.stack([symmetric_part(path, order).reshape(-1) for path in legacy])
    flat_basis = basis.detach().numpy().reshape(basis.shape[0], -1)
    solution, *_ = np.linalg.lstsq(flat_basis.T, targets.T, rcond=None)
    np.testing.assert_allclose(
        flat_basis.T @ solution, targets.T, atol=CLOSED_FORM_ATOL
    )
    return solution.T
