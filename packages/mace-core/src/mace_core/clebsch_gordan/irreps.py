"""Irreducible representations of O(3), and the algebra over them.

No e3nn. The parsing grammar is the project's ``<mul>x<l><parity>`` one; what
this module adds on top is the part that is algebra rather than syntax: a total
order, dimensions, and the selection rules for a tensor product.

**The total order is a decision, not an inheritance.** It is the order the
coupling paths are enumerated in, which is to say it is the order the weights
sit in on disk. It is defined here as ascending degree, and even parity before
odd at equal degree, because that is the textbook reading order and because a
convention the project can state in one line is worth more than one it has to
measure out of a library.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

__all__ = ["IRREPS_GRAMMAR", "Irrep", "Irreps"]

IRREPS_GRAMMAR = (
    "a '+'-separated sum of terms, each '<l><parity>' or "
    "'<multiplicity>x<l><parity>', where <l> is a non-negative integer and "
    "<parity> is 'e' (even) or 'o' (odd). For example '0e', '1o', '0e+2e', "
    "'128x0e+128x1o'."
)

_TERM = re.compile(r"^(?:(\d+)x)?(\d+)([eo])$")


@dataclass(frozen=True, order=False)
class Irrep:
    """One irreducible representation, of degree ``degree`` and parity ``parity``.

    Attributes:
        degree: The rotation order ``l``. Spans ``2l + 1`` components.
        parity: ``+1`` for even under inversion, ``-1`` for odd.
    """

    degree: int
    parity: int

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise ValueError(f"degree must be non-negative, got {self.degree}")
        if self.parity not in (1, -1):
            raise ValueError(f"parity must be +1 or -1, got {self.parity}")

    @property
    def dimension(self) -> int:
        return 2 * self.degree + 1

    def __str__(self) -> str:
        return f"{self.degree}{'e' if self.parity == 1 else 'o'}"

    def __lt__(self, other: Irrep) -> bool:
        """Ascending degree, even before odd at equal degree.

        This order is the path order and therefore the weight order on disk.
        """
        return (self.degree, -self.parity) < (other.degree, -other.parity)

    def couple(self, other: Irrep) -> Iterator[Irrep]:
        """Every irrep the tensor product with ``other`` contains, in order.

        The triangle inequality on the degrees, and parities multiply. Each
        appears exactly once, which is what makes O(3) simply reducible.
        """
        for degree in range(
            abs(self.degree - other.degree), self.degree + other.degree + 1
        ):
            yield Irrep(degree, self.parity * other.parity)


class IrrepsError(ValueError):
    """A string that is not a well-formed irreps declaration."""


@dataclass(frozen=True)
class Irreps:
    """A direct sum of irreps with multiplicities, in the order written.

    The order is preserved rather than sorted: it is the layout of the values
    themselves, and sorting it would silently move every weight.
    """

    terms: tuple[tuple[int, Irrep], ...]

    @classmethod
    def parse(cls, text: str) -> Irreps:
        """Parse a declaration such as ``"128x0e+128x1o"``.

        Raises:
            IrrepsError: If the declaration is empty or a term is malformed.
                The message quotes the offending term and states the grammar.
        """
        if not text or not text.strip():
            raise IrrepsError(f"the declaration is empty. Expected {IRREPS_GRAMMAR}")
        terms = []
        for piece in text.split("+"):
            match = _TERM.match(piece.strip())
            if match is None:
                raise IrrepsError(
                    f"{piece.strip()!r} is not a valid term in {text!r}. "
                    f"Expected {IRREPS_GRAMMAR}"
                )
            multiplicity, degree, parity = match.groups()
            terms.append(
                (
                    1 if multiplicity is None else int(multiplicity),
                    Irrep(int(degree), 1 if parity == "e" else -1),
                )
            )
        return cls(tuple(terms))

    @property
    def dimension(self) -> int:
        """The total number of components."""
        return sum(mul * ir.dimension for mul, ir in self.terms)

    def slices(self) -> Iterator[tuple[slice, Irrep]]:
        """Each term's slice of the flat value vector, in ``mul_ir`` layout.

        ``mul_ir`` means the multiplicity index varies slowest: a ``2x1o`` term
        is two contiguous blocks of three, not three interleaved pairs. It is
        the canonical layout, and the only one this package stores.
        """
        start = 0
        for mul, ir in self.terms:
            for _ in range(mul):
                yield slice(start, start + ir.dimension), ir
                start += ir.dimension

    def __iter__(self) -> Iterator[tuple[int, Irrep]]:
        return iter(self.terms)

    def __contains__(self, ir: Irrep) -> bool:
        return any(term_ir == ir for _, term_ir in self.terms)

    def __str__(self) -> str:
        return "+".join(f"{mul}x{ir}" for mul, ir in self.terms)
