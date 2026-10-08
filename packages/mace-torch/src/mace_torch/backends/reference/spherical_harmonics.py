"""Real spherical harmonics in the legacy e3nn convention, in plain torch.

This is the reference implementation of the ``SphericalHarmonicsOp``: closed
form, written for clarity, differentiable to any order by construction because
it is nothing but products and sums of the input components. A backend may
override it with an accelerated kernel; this one is what such a kernel is
measured against.

**Convention.** The output reproduces
``e3nn.o3.SphericalHarmonics(lmax, normalize=True, normalization="component")``
exactly, which the rest of the model assumes:

* the input vector is normalised to the unit sphere first (a zero vector maps
  to zero and then to the ``l = 0`` component alone);
* ``"component"`` normalisation: each ``l`` block has squared norm ``2l + 1`` on
  the unit sphere;
* the blocks are concatenated ``l = 0, 1, ..., lmax`` and, inside a block, run
  over ``m = -l, ..., l``; the ``l = 1`` block is ``(x, y, z)`` up to the factor
  ``sqrt(3)``.

That last point makes e3nn's harmonics the textbook real harmonics with **y as
the polar axis**, and no Condon-Shortley phase. The textbook functions, with
polar axis ``z'``, are evaluated at ``(x', y', z') = (z, x, y)``: index the
input with :data:`E3NN_AXIS_ORDER`. Negative ``m`` carries ``sin(|m| phi)``,
positive ``m`` carries ``cos(m phi)``, each with the Schmidt semi-normalisation
``sqrt((2 - delta_m0) (l-|m|)! / (l+|m|)!)`` so that the ``l`` block has unit
norm before the ``sqrt(2l + 1)`` component factor.

The recurrences are written in homogeneous form (``r^2`` where the textbook has
``1``), so on unit input they are the textbook functions and off it they are
the regular solid harmonics, which is what keeps a zero vector finite.
"""

from __future__ import annotations

import math

import torch

__all__ = ["E3NN_AXIS_ORDER", "SphericalHarmonics", "spherical_harmonics"]

#: The permutation that takes ``(x, y, z)`` to the textbook ``(x', y', z')`` with
#: ``z' = y`` the polar axis. The same ``[2, 0, 1]`` develop applies before
#: handing vectors to sphericart, whose output is in the textbook convention.
E3NN_AXIS_ORDER = (2, 0, 1)


def _double_factorial(n: int) -> float:
    """``n!! = n (n-2) (n-4) ...`` with ``(-1)!! = 1``."""
    result = 1.0
    while n > 1:
        result *= n
        n -= 2
    return result


def _schmidt_factor(degree: int, order: int) -> float:
    """``sqrt((2 - delta_m0) (l - m)! / (l + m)!)`` for ``m >= 0``."""
    two_or_one = 1.0 if order == 0 else 2.0
    return math.sqrt(
        two_or_one * math.factorial(degree - order) / math.factorial(degree + order)
    )


def spherical_harmonics(
    vectors: torch.Tensor, lmax: int, normalize: bool = True
) -> torch.Tensor:
    """Real spherical harmonics ``Y^l_m`` of ``vectors`` for ``l = 0..lmax``.

    Args:
        vectors: ``[..., 3]`` Cartesian vectors, any unit; ``(x, y, z)`` order.
        lmax: highest degree. The output has ``(lmax + 1)^2`` components.
        normalize: project onto the unit sphere first. This is what legacy
            does on every edge vector. With ``False`` the return is the
            regular solid harmonics, ``r^l Y^l_m`` of the direction.

    Returns:
        ``[..., (lmax + 1)^2]`` in the convention described in the module
        docstring, same dtype as ``vectors``.
    """
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    if vectors.shape[-1] != 3:
        raise ValueError(
            f"vectors must have 3 Cartesian components on the last axis, got shape "
            f"{tuple(vectors.shape)}"
        )
    if normalize:
        # forward 0 for a zero vector rather than nan; e3nn's eps of 1e-12
        vectors = torch.nn.functional.normalize(vectors, dim=-1)

    x_polar_frame = vectors[..., E3NN_AXIS_ORDER[0]]
    y_polar_frame = vectors[..., E3NN_AXIS_ORDER[1]]
    z_polar_axis = vectors[..., E3NN_AXIS_ORDER[2]]
    radius_squared = x_polar_frame**2 + y_polar_frame**2 + z_polar_axis**2

    # rho^m cos(m phi) and rho^m sin(m phi) as polynomials in (x', y'), m = 0..lmax
    cosines = [torch.ones_like(x_polar_frame)]
    sines = [torch.zeros_like(x_polar_frame)]
    for order in range(1, lmax + 1):
        cosines.append(
            x_polar_frame * cosines[order - 1] - y_polar_frame * sines[order - 1]
        )
        sines.append(
            x_polar_frame * sines[order - 1] + y_polar_frame * cosines[order - 1]
        )

    # legendre[l][m] = P_l^m(z') / rho^m, a homogeneous polynomial of degree l - m
    # in (x', y', z'), by the standard three-term recurrence in l at fixed m
    legendre: list[list[torch.Tensor]] = [[] for _ in range(lmax + 1)]
    for order in range(lmax + 1):
        diagonal = torch.full_like(z_polar_axis, _double_factorial(2 * order - 1))
        legendre[order].append(diagonal)
        if order + 1 <= lmax:
            legendre[order + 1].append((2 * order + 1) * z_polar_axis * diagonal)
        for degree in range(order + 2, lmax + 1):
            previous = legendre[degree - 1][order]
            before_previous = legendre[degree - 2][order]
            legendre[degree].append(
                (
                    (2 * degree - 1) * z_polar_axis * previous
                    - (degree + order - 1) * radius_squared * before_previous
                )
                / (degree - order)
            )

    components: list[torch.Tensor] = []
    for degree in range(lmax + 1):
        component_factor = math.sqrt(2 * degree + 1)
        for order in range(degree, 0, -1):  # m = -l .. -1: the sine terms
            factor = component_factor * _schmidt_factor(degree, order)
            components.append(factor * legendre[degree][order] * sines[order])
        components.append(component_factor * legendre[degree][0])  # m = 0
        for order in range(1, degree + 1):  # m = 1 .. l: the cosine terms
            factor = component_factor * _schmidt_factor(degree, order)
            components.append(factor * legendre[degree][order] * cosines[order])
    return torch.stack(components, dim=-1)


class SphericalHarmonics(torch.nn.Module):
    """:func:`spherical_harmonics` as a module holding ``lmax``."""

    def __init__(self, lmax: int, normalize: bool = True):
        super().__init__()
        if lmax < 0:
            raise ValueError(f"lmax must be non-negative, got {lmax}")
        self.lmax = lmax
        self.normalize = normalize

    @property
    def output_dim(self) -> int:
        return (self.lmax + 1) ** 2

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """``[..., 3]`` -> ``[..., (lmax + 1)^2]``."""
        return spherical_harmonics(vectors, self.lmax, normalize=self.normalize)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(lmax={self.lmax}, normalize={self.normalize})"
        )
