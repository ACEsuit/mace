"""The irreps string grammar, and nothing beyond it.

An observable declares the spherical-tensor shape of its values as a string:
``"0e"`` for a scalar, ``"1o"`` for a polar vector, ``"1e"`` for an axial one,
``"0e+2e"`` for a symmetric rank-2 tensor, ``"128x0e+128x1o+128x2e"`` for a
block of hidden features. This module parses and validates that string.

It is grammar only. No tensor products, no Clebsch-Gordan coefficients, no
simplification or sorting of terms: the algebra lands in its own module with
the reduced basis, and a half-implementation here would be the version every
later caller had to work around. What a caller gets from this module is the
guarantee that a declaration is well formed, the terms it names, and the
dimension they add up to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "IRREPS_GRAMMAR",
    "IrrepTerm",
    "IrrepsGrammarError",
    "irreps_dimension",
    "parse_irreps",
]

#: Stated in every error this module raises, because an error that says only
#: "invalid" leaves the reader to guess between four plausible spellings.
IRREPS_GRAMMAR = (
    "a '+'-separated sum of terms, each written '<l><parity>' or "
    "'<multiplicity>x<l><parity>', where <l> is a non-negative integer and "
    "<parity> is 'e' (even) or 'o' (odd). Examples: '0e' (a scalar), '1o' (a "
    "polar vector), '1e' (an axial vector), '0e+2e' (a symmetric rank-2 "
    "tensor), '128x0e+128x1o+128x2e'."
)

_TERM = re.compile(r"^(?:(\d+)x)?(\d+)([eo])$")


class IrrepsGrammarError(ValueError):
    """A declaration that is not a well-formed irreps string."""


@dataclass(frozen=True)
class IrrepTerm:
    """One ``<multiplicity>x<l><parity>`` term of a declaration.

    Attributes:
        multiplicity: How many copies of the irrep the term declares.
        degree: The rotation order ``l``. A value of ``l`` spans ``2l + 1``
            components.
        parity: ``"e"`` or ``"o"``, the behaviour under inversion. The
            distinction is load-bearing rather than decorative: a force is
            ``1o`` and a magnetic moment is ``1e``, and a model that confuses
            them is wrong under inversion while looking right under rotation.
    """

    multiplicity: int
    degree: int
    parity: str

    @property
    def dimension(self) -> int:
        """The number of components this term contributes."""
        return self.multiplicity * (2 * self.degree + 1)

    def __str__(self) -> str:
        return f"{self.multiplicity}x{self.degree}{self.parity}"


def parse_irreps(text: str, *, observable: str | None = None) -> tuple[IrrepTerm, ...]:
    """Parse an irreps declaration into its terms.

    Args:
        text: The declaration, for example ``"128x0e+128x1o"``.
        observable: The observable the declaration belongs to. Named in the
            error message, because a validation failure reported without it
            tells the reader which grammar was violated and not which of their
            declarations violated it.

    Returns:
        The terms, in the order they were written. The order is preserved
        rather than sorted: it is the layout of the values themselves.

    Raises:
        IrrepsGrammarError: If the declaration is empty or any term is
            malformed.
    """
    where = f"observable {observable!r}: " if observable else ""
    if not text or not text.strip():
        raise IrrepsGrammarError(
            f"{where}the irreps declaration is empty. Expected {IRREPS_GRAMMAR}"
        )
    terms = []
    for piece in text.split("+"):
        match = _TERM.match(piece.strip())
        if match is None:
            raise IrrepsGrammarError(
                f"{where}{piece.strip()!r} is not a valid irreps term in "
                f"{text!r}. Expected {IRREPS_GRAMMAR}"
            )
        multiplicity, degree, parity = match.groups()
        terms.append(
            IrrepTerm(
                multiplicity=1 if multiplicity is None else int(multiplicity),
                degree=int(degree),
                parity=parity,
            )
        )
    return tuple(terms)


def irreps_dimension(text: str, *, observable: str | None = None) -> int:
    """The total number of components a declaration spans."""
    return sum(term.dimension for term in parse_irreps(text, observable=observable))
