"""Validate observable irreps strings and compute their dimensions.

Examples: ``0e`` scalar, ``1o`` polar vector, ``1e`` axial vector, ``0e+2e``
symmetric rank-2 tensor, ``128x0e+128x1o`` hidden features. Terms retain their
written order. Observable declarations require canonical spelling without
leading zeros; the algebra parser retains its more permissive spelling.
"""

from __future__ import annotations

from dataclasses import dataclass

from mace_core._irreps import parse_terms

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
    "'<multiplicity>x<l><parity>', where <multiplicity> is a positive integer, "
    "<l> is a non-negative integer, both are written in ASCII digits with no "
    "leading zeros, and <parity> is 'e' (even) or 'o' (odd). Examples: '0e' (a "
    "scalar), '1o' (a "
    "polar vector), '1e' (an axial vector), '0e+2e' (a symmetric rank-2 "
    "tensor), '128x0e+128x1o+128x2e'."
)

_PARITIES = ("e", "o")


class IrrepsGrammarError(ValueError):
    """A declaration that is not a well-formed irreps string."""


@dataclass(frozen=True)
class IrrepTerm:
    """One ``<multiplicity>x<l><parity>`` term.

    Attributes:
        multiplicity: Positive integer number of copies; bools are rejected.
        degree: Non-negative integer rotation order ``l``, spanning
            ``2l + 1`` components.
        parity: ``"e"`` (even) or ``"o"`` (odd) under inversion.

    Raises:
        IrrepsGrammarError: A field violates these rules, including direct
            construction.
    """

    multiplicity: int
    degree: int
    parity: str

    def __post_init__(self) -> None:
        # `bool` is a subclass of `int`, so `True` would otherwise pass as 1.
        if type(self.multiplicity) is not int or self.multiplicity < 1:
            raise IrrepsGrammarError(
                f"an irreps term has multiplicity {self.multiplicity!r}. The "
                f"multiplicity is a positive integer: a term with none spans "
                f"no components."
            )
        if type(self.degree) is not int or self.degree < 0:
            raise IrrepsGrammarError(
                f"an irreps term has degree {self.degree!r}. The degree l is a "
                f"non-negative integer."
            )
        if self.parity not in _PARITIES:
            raise IrrepsGrammarError(
                f"an irreps term has parity {self.parity!r}. The parity is "
                f"'e' (even) or 'o' (odd)."
            )

    @property
    def dimension(self) -> int:
        """The number of components this term contributes."""
        return self.multiplicity * (2 * self.degree + 1)

    def __str__(self) -> str:
        return f"{self.multiplicity}x{self.degree}{self.parity}"


def parse_irreps(text: str, *, observable: str | None = None) -> tuple[IrrepTerm, ...]:
    """Parse a declaration, preserving its term order.

    Args:
        text: Declaration such as ``"128x0e+128x1o"``.
        observable: Optional observable name included in errors.

    Raises:
        IrrepsGrammarError: The declaration is empty or a term is malformed.
    """
    where = f"observable {observable!r}: " if observable else ""
    try:
        terms = parse_terms(text, canonical=True)
    except ValueError as failure:
        raise IrrepsGrammarError(
            f"{where}{failure} Expected {IRREPS_GRAMMAR}"
        ) from failure
    return tuple(IrrepTerm(*term) for term in terms)


def irreps_dimension(text: str, *, observable: str | None = None) -> int:
    """The total number of components a declaration spans."""
    return sum(term.dimension for term in parse_irreps(text, observable=observable))
