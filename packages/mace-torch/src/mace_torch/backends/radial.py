"""Radial bases and the cutoff envelope, as closed forms.

Reference-only ops: cheap enough that a backend is not required to provide its
own, and a backend that does must reproduce these functions rather than its own.

The cutoff is a separate factor rather than folded into each basis, because
every basis uses the same one and folding it in is how two of them end up
subtly different.

Conventions shared by every class:

* distances are in Angstrom and enter as a column tensor ``[n_edges, 1]``;
* buffers are created in the ``dtype`` passed at construction, and in
  ``torch.get_default_dtype()`` when none is, so a module built under
  ``float64`` computes in ``float64``. The reference backend always passes the
  descriptor's precision rather than leaning on the process default.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "BesselBasis",
    "ChebyshevBasis",
    "GaussianBasis",
    "PolynomialCutoff",
    "polynomial_envelope",
]


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------


class BesselBasis(torch.nn.Module):
    """Spherical Bessel functions of order zero, equation (7) of the MACE paper.

    ``f_n(r) = sqrt(2 / r_max) * sin(n * pi * r / r_max) / r`` for ``n = 1..num_basis``.

    The frequencies ``n * pi / r_max`` are a buffer, or a parameter when
    ``trainable`` is set.
    """

    frequencies: torch.Tensor
    r_max: torch.Tensor
    prefactor: torch.Tensor

    def __init__(
        self,
        r_max: float,
        num_basis: int = 8,
        trainable: bool = False,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        frequencies = (
            math.pi
            / r_max
            * torch.linspace(start=1.0, end=num_basis, steps=num_basis, dtype=dtype)
        )
        if trainable:
            self.frequencies = torch.nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)
        self.register_buffer("r_max", torch.tensor(r_max, dtype=dtype))
        self.register_buffer(
            "prefactor", torch.tensor(math.sqrt(2.0 / r_max), dtype=dtype)
        )

    @property
    def num_basis(self) -> int:
        return len(self.frequencies)

    def forward(self, edge_lengths: torch.Tensor) -> torch.Tensor:
        """``[n_edges, 1]`` distances in Angstrom -> ``[n_edges, num_basis]``.

        ``sin(w r) / r`` is ``0 / 0`` at ``r = 0``, where its limit is ``w``. A
        zero-length edge (a padding edge in a static-shape batch) takes that
        limit through a masked division, so the value and every derivative
        order stay finite there; for ``r > 0`` the arithmetic is unchanged.
        ``torch.sinc`` is not used: its second derivative is NaN at zero, and
        force training differentiates twice through the basis.
        """
        is_zero = edge_lengths == 0.0
        safe_lengths = torch.where(is_zero, torch.ones_like(edge_lengths), edge_lengths)
        return self.prefactor * torch.where(
            is_zero,
            self.frequencies,
            torch.sin(self.frequencies * edge_lengths) / safe_lengths,
        )

    def extra_repr(self) -> str:
        return (
            f"r_max={self.r_max.item()}, num_basis={self.num_basis}, "
            f"trainable={self.frequencies.requires_grad}"
        )


class ChebyshevBasis(torch.nn.Module):
    """Chebyshev polynomials ``T_n`` of the raw input, ``--radial_type chebyshev``.

    ``T_0 = 1``, ``T_1 = x``, ``T_n = 2 x T_{n-1} - T_{n-2}``, evaluated by the
    three-term recurrence so the basis is differentiable to any order. Without
    the constant term the orders are ``1..num_basis``; with it, ``0..num_basis-1``.

    Both legacy classes collapse onto this one: the ``--radial_type chebyshev``
    basis is the default, and so is the magnetic family's moment basis, which
    legacy built as a second class with ``include_constant=False``. Nobody in
    the legacy tree uses the constant term.

    The input is not mapped into ``[-1, 1]``: legacy accepted and stored an
    ``r_max`` it never used, so a model trained with this basis sees the
    divergent ``cosh`` branch beyond 1 Angstrom. That is pinned by test.
    """

    def __init__(self, num_basis: int = 8, include_constant: bool = False):
        super().__init__()
        self.num_basis = num_basis
        self.include_constant = include_constant

    def forward(self, edge_lengths: torch.Tensor) -> torch.Tensor:
        """``[n_edges, 1]`` -> ``[n_edges, num_basis]``.

        The magnetic family passes a transformed moment length in place of an
        edge length; the recurrence does not care which.
        """
        highest_order = self.num_basis - 1 if self.include_constant else self.num_basis
        previous = torch.ones_like(edge_lengths)
        current = edge_lengths
        polynomials = [previous, current]
        for _ in range(2, highest_order + 1):
            previous, current = current, 2.0 * edge_lengths * current - previous
            polynomials.append(current)
        first_order = 0 if self.include_constant else 1
        return torch.cat(
            polynomials[first_order : first_order + self.num_basis], dim=-1
        )

    def extra_repr(self) -> str:
        return f"num_basis={self.num_basis}, include_constant={self.include_constant}"


class GaussianBasis(torch.nn.Module):
    """Gaussians on evenly spaced centres in ``[0, r_max]``, ``--radial_type gaussian``.

    ``g_k(r) = exp(-0.5 * ((r - c_k) / w)^2)`` with centres
    ``c_k = linspace(0, r_max, num_basis)`` and width ``w = r_max / (num_basis - 1)``,
    folded into one coefficient ``-0.5 / w^2``. Never zero: only the cutoff envelope
    makes a long edge vanish.
    """

    centers: torch.Tensor

    def __init__(
        self,
        r_max: float,
        num_basis: int = 128,
        trainable: bool = False,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        centers = torch.linspace(
            start=0.0,
            end=r_max,
            steps=num_basis,
            dtype=dtype or torch.get_default_dtype(),
        )
        if trainable:
            self.centers = torch.nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)
        self.exponent_coefficient = -0.5 / (r_max / (num_basis - 1)) ** 2
        self.num_basis = num_basis

    def forward(self, edge_lengths: torch.Tensor) -> torch.Tensor:
        """``[n_edges, 1]`` -> ``[n_edges, num_basis]``."""
        offsets = edge_lengths - self.centers
        return torch.exp(self.exponent_coefficient * torch.pow(offsets, 2))

    def extra_repr(self) -> str:
        return f"num_basis={self.num_basis}, trainable={self.centers.requires_grad}"


# ---------------------------------------------------------------------------
# Cutoff envelope
# ---------------------------------------------------------------------------


def polynomial_envelope(
    edge_lengths: torch.Tensor, r_max: torch.Tensor, polynomial_order: int
) -> torch.Tensor:
    """The smooth envelope ``u(r)`` that is 1 at ``r = 0`` and 0 with two vanishing
    derivatives at ``r = r_max``, and exactly zero beyond.

    With ``x = r / r_max`` and ``p`` the order::

        u = 1 - (p+1)(p+2)/2 * x^p + p(p+2) * x^(p+1) - p(p+1)/2 * x^(p+2)

    multiplied by the mask ``r < r_max``. The mask is what makes the padded-batch
    contract hold: a self-loop edge shifted by ``2 * r_max`` contributes exactly
    zero, not a small number. ``r_max`` may be a tensor broadcasting against
    ``edge_lengths``, which is how the ZBL term uses a per-pair radius.
    """
    order = float(polynomial_order)
    scaled = edge_lengths / r_max
    envelope = (
        1.0
        - ((order + 1.0) * (order + 2.0) / 2.0) * torch.pow(scaled, order)
        + order * (order + 2.0) * torch.pow(scaled, order + 1.0)
        - (order * (order + 1.0) / 2.0) * torch.pow(scaled, order + 2.0)
    )
    return envelope * (edge_lengths < r_max)


class PolynomialCutoff(torch.nn.Module):
    """The polynomial envelope as a module, sized by ``--num_cutoff_basis``."""

    r_max: torch.Tensor

    def __init__(
        self, r_max: float, polynomial_order: int = 6, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.polynomial_order = int(polynomial_order)
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=dtype or torch.get_default_dtype())
        )

    def forward(self, edge_lengths: torch.Tensor) -> torch.Tensor:
        """Same shape as the input; values in ``[0, 1]``."""
        return polynomial_envelope(edge_lengths, self.r_max, self.polynomial_order)

    def extra_repr(self) -> str:
        return f"polynomial_order={self.polynomial_order}, r_max={self.r_max.item()}"
