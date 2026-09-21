"""How many symmetric paths there are, counted without any Clebsch-Gordan.

The converter projects a trained model's symmetric-contraction weights from the
full basis onto the reduced one. That projection is only meaningful if the
reduced basis has the right *rank*: too few directions and the projection
silently throws away part of the model, too many and it is not reduced at all.

Checking that rank against the basis the projection itself uses would be
self-validating. So this counts the same number a completely different way,
from the characters of O(3), and touches no tensor of coupling coefficients at
all: not the rewrite's, not the frozen tree's, not e3nn's.

The multiplicity of an irrep in a representation is the inner product of their
characters, and the character of a symmetric power follows from the cycle index
of the symmetric group:

    chi_Sym1(g) = chi(g)
    chi_Sym2(g) = ( chi(g)^2 + chi(g^2) ) / 2
    chi_Sym3(g) = ( chi(g)^3 + 3 chi(g) chi(g^2) + 2 chi(g^3) ) / 6

For O(3) the group splits into proper rotations and their products with the
inversion, so the integral is the average of the two halves, with the target's
parity weighting the improper one. An improper element squares to a proper one
and cubes back to an improper one, which is the only fiddly part.
"""

from __future__ import annotations

import numpy as np

__all__ = ["symmetric_multiplicity"]

#: Rotation angles for the class integral over SO(3), and the class measure
#: `(1 - cos t) / pi`. The endpoints are dropped because the character has a
#: removable singularity at zero and the measure vanishes there anyway.
_ANGLES = np.linspace(0.0, np.pi, 400_001)[1:-1]
_MEASURE = (1.0 - np.cos(_ANGLES)) / np.pi


def _character(degree: int, angle: np.ndarray) -> np.ndarray:
    """The SO(3) character of degree `l` at rotation angle `t`."""
    return np.sin((degree + 0.5) * angle) / np.sin(angle / 2.0)


def _representation_character(
    terms: list[tuple[int, int]], angle: np.ndarray, improper: bool
) -> np.ndarray:
    """The character of a sum of irreps, on a proper or improper element."""
    total = np.zeros_like(angle)
    for degree, parity in terms:
        total = total + (parity if improper else 1) * _character(degree, angle)
    return total


def _symmetric_power_character(
    terms: list[tuple[int, int]], order: int, angle: np.ndarray, improper: bool
) -> np.ndarray:
    """The character of `Sym^order` of the representation."""
    first = _representation_character(terms, angle, improper)
    # g squared is proper whichever g was; g cubed keeps g's own parity.
    second = _representation_character(terms, (2 * angle) % (2 * np.pi), False)
    third = _representation_character(terms, (3 * angle) % (2 * np.pi), improper)
    if order == 1:
        return first
    if order == 2:
        return 0.5 * (first**2 + second)
    if order == 3:
        return (first**3 + 3 * first * second + 2 * third) / 6.0
    raise ValueError(
        f"the symmetric-power character is spelled out here for orders 1 to 3, "
        f"and {order} was asked for. Add its cycle-index term rather than "
        f"extrapolating."
    )


def symmetric_multiplicity(
    terms: list[tuple[int, int]], order: int, degree_out: int, parity_out: int
) -> float:
    """How many times an irrep appears in the symmetric power of a rep.

    Args:
        terms: The input representation, as ``(degree, parity)`` pairs with
            parity ``+1`` for even and ``-1`` for odd.
        order: The body order, the power the representation is raised to.
        degree_out: The target irrep's degree.
        parity_out: The target irrep's parity.

    Returns:
        The multiplicity, which is an integer up to the quadrature error. It is
        returned as a float on purpose: rounding here would hide a derivation
        that had come out at 4.5.
    """
    character = _character(degree_out, _ANGLES) * _MEASURE
    proper = np.trapezoid(
        _symmetric_power_character(terms, order, _ANGLES, False) * character, _ANGLES
    )
    improper = np.trapezoid(
        _symmetric_power_character(terms, order, _ANGLES, True) * character, _ANGLES
    )
    return float(0.5 * (proper + parity_out * improper))
