"""Radial bases and the cutoff, as closed forms.

Reference-only ops: cheap enough that a backend is not required to provide its
own, and a backend that does must produce these functions rather than its own.

The cutoff is a separate factor rather than folded into each basis, because
every basis uses the same one and folding it in is how two of them end up
subtly different.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["polynomial_cutoff", "radial_basis"]


def polynomial_cutoff(lengths: Tensor, cutoff: float, power: int = 6) -> Tensor:
    """A smooth envelope going to zero at ``cutoff``, with zero derivative there.

    Args:
        lengths: ``[n_edges, 1]``, in Angstrom.
        cutoff: The radius, in Angstrom.
        power: The polynomial order. Higher keeps the envelope flatter for
            longer before it falls.

    Zero derivative at the cutoff is the property that matters: without it a
    force is discontinuous as an atom crosses the radius, which shows up as
    energy drift in a simulation rather than as an error.
    """
    scaled = lengths / cutoff
    envelope = (
        1.0
        - ((power + 1.0) * (power + 2.0) / 2.0) * scaled.pow(power)
        + power * (power + 2.0) * scaled.pow(power + 1)
        - (power * (power + 1.0) / 2.0) * scaled.pow(power + 2)
    )
    return envelope * (scaled < 1.0)


def radial_basis(
    lengths: Tensor,
    kind: str,
    num_basis: int,
    cutoff: float,
) -> Tensor:
    """Embed edge lengths into ``num_basis`` functions.

    Args:
        lengths: ``[n_edges, 1]``, in Angstrom.
        kind: ``"bessel"``, ``"gaussian"`` or ``"chebyshev"``, lowercase.
        num_basis: How many functions.
        cutoff: The radius, in Angstrom.

    Returns:
        ``[n_edges, num_basis]``, already multiplied by the cutoff envelope.

    Raises:
        ValueError: On an unknown kind, naming the ones there are.
    """
    if kind == "bessel":
        orders = torch.arange(
            1, num_basis + 1, dtype=lengths.dtype, device=lengths.device
        )
        values = (
            math.sqrt(2.0 / cutoff)
            * torch.sin(orders * math.pi * lengths / cutoff)
            / lengths
        )
    elif kind == "gaussian":
        centres = torch.linspace(
            0.0, cutoff, num_basis, dtype=lengths.dtype, device=lengths.device
        )
        width = cutoff / max(num_basis - 1, 1)
        values = torch.exp(-((lengths - centres) ** 2) / (2.0 * width**2))
    elif kind == "chebyshev":
        orders = torch.arange(0, num_basis, dtype=lengths.dtype, device=lengths.device)
        folded = torch.clamp(2.0 * lengths / cutoff - 1.0, -1.0, 1.0)
        values = torch.cos(orders * torch.acos(folded))
    else:
        raise ValueError(
            f"{kind!r} is not a radial basis this backend builds. The kinds "
            f"are 'bessel', 'gaussian' and 'chebyshev', lowercase."
        )
    return values * polynomial_cutoff(lengths, cutoff)
