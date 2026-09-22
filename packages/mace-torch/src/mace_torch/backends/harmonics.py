"""Real spherical harmonics, built from this project's own coefficients.

Not a table and not a library. Each degree is the previous one tensored with
the ``l = 1`` harmonics and projected back with
:func:`mace_core.clebsch_gordan.real_basis.wigner_3j_real`, which is the same
function the symmetric contraction is built from.

That is the point of doing it this way. The harmonics and the contraction have
to agree on a convention or the model is wrong, and the symptom is numerical
rather than an error. Deriving one from the other makes disagreement
impossible instead of making it testable.

Convention: components run ``m = -l .. +l``, so the ``l = 1`` block of a unit
direction is ``(y, z, x)``. Each degree is scaled to unit norm, which is well
defined because the sum of the squares over ``m`` is the same for every
direction.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import torch
from mace_core.clebsch_gordan.real_basis import wigner_3j_real
from torch import Tensor

__all__ = ["spherical_harmonics"]


@cache
def _recursion(degree: int) -> np.ndarray:
    """The coefficients taking ``(Y_{l-1}, Y_1)`` to ``Y_l``, already scaled.

    The scale is fixed once, by evaluating on a single direction, because the
    norm does not depend on the direction.
    """
    coupling = wigner_3j_real(degree, degree - 1, 1)
    if degree == 1:
        return coupling
    direction = np.array([0.31, -0.57, 0.76])
    direction = direction / np.linalg.norm(direction)
    lower = _evaluate_numpy(direction[None, :], degree - 1)[-1][0]
    first = direction[[1, 2, 0]]
    raw = np.einsum("oab,a,b->o", coupling, lower, first)
    return coupling / np.linalg.norm(raw)


def _evaluate_numpy(directions: np.ndarray, lmax: int) -> list[np.ndarray]:
    """Every degree up to ``lmax`` on unit directions, in numpy.

    Used to fix the recursion's scale; the torch path below mirrors it.
    """
    blocks = [np.ones((directions.shape[0], 1))]
    if lmax == 0:
        return blocks
    first = directions[:, [1, 2, 0]]
    blocks.append(first)
    for degree in range(2, lmax + 1):
        blocks.append(np.einsum("oab,na,nb->no", _recursion(degree), blocks[-1], first))
    return blocks


def spherical_harmonics(
    directions: Tensor, lmax: int, normalize: bool = True
) -> Tensor:
    """Real spherical harmonics of ``directions``, concatenated over degrees.

    Args:
        directions: ``[n, 3]``, Cartesian. Normalized first unless
            ``normalize`` is false, in which case they are taken as already
            unit.
        lmax: The highest degree.
        normalize: Whether to normalize the input directions.

    Returns:
        ``[n, (lmax + 1) ** 2]``, degree 0 first, each degree's components
        running ``m = -l .. +l``.
    """
    if normalize:
        directions = directions / directions.norm(dim=-1, keepdim=True)
    blocks = [directions.new_ones((directions.shape[0], 1))]
    if lmax >= 1:
        first = directions[:, [1, 2, 0]]
        blocks.append(first)
        for degree in range(2, lmax + 1):
            coupling = torch.as_tensor(
                _recursion(degree), dtype=directions.dtype, device=directions.device
            )
            blocks.append(torch.einsum("oab,na,nb->no", coupling, blocks[-1], first))
    return torch.cat(blocks, dim=-1)
