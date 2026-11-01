"""O(3) irreps: parsing, dimensions, ordering and tensor-product selection rules.

Irreps sort by ascending degree, even before odd at equal degree. Coupling
paths use this order, so it determines the weight layout on disk.
"""

# API and notation inspired by e3nn.o3.Irrep/Irreps (e3nn authors, MIT):
# https://github.com/e3nn/e3nn/blob/0.4.4/e3nn/o3/_irreps.py
# MACE defines its own validation and ordering; see the module docstring.

from __future__ import annotations

import numbers
import re
from collections.abc import Iterator
from dataclasses import dataclass

from mace_core._irreps import parse_terms

__all__ = ["IRREPS_GRAMMAR", "Irrep", "Irreps", "IrrepsError"]

IRREPS_GRAMMAR = (
    "a '+'-separated sum of terms, each '<l><parity>' or "
    "'<multiplicity>x<l><parity>', where <l> is a non-negative integer and "
    "<parity> is 'e' (even) or 'o' (odd). For example '0e', '1o', '0e+2e', "
    "'128x0e+128x1o'."
)

_SINGLE = re.compile(r"(\d+)([eo])", re.ASCII)


class IrrepsError(ValueError):
    """A value that is not a well-formed irrep or irreps declaration."""


def _is_integer(value: object) -> bool:
    """An integer, and not a bool: ``True`` would otherwise pass as degree 1."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


@dataclass(frozen=True, order=False)
class Irrep:
    """One irreducible representation, of degree ``degree`` and parity ``parity``.

    Attributes:
        degree: The rotation order ``l``, a non-negative integer. Spans
            ``2l + 1`` components.
        parity: The integer ``+1`` for even under inversion, ``-1`` for odd.

    Raises:
        IrrepsError: If ``degree`` is not a non-negative integer or ``parity``
            is not ``+1`` or ``-1``. Bools and floats are refused rather than
            coerced, so ``Irrep(1.5, 1)`` and ``Irrep(True, 1)`` both raise.
    """

    degree: int
    parity: int

    def __post_init__(self) -> None:
        if not _is_integer(self.degree) or self.degree < 0:
            raise IrrepsError(
                f"degree must be a non-negative integer, got {self.degree!r}. "
                f"Pass the rotation order l, for example Irrep(2, 1)."
            )
        if not _is_integer(self.parity) or self.parity not in (1, -1):
            raise IrrepsError(
                f"parity must be the integer 1 (even) or -1 (odd), got {self.parity!r}."
            )
        object.__setattr__(self, "degree", int(self.degree))
        object.__setattr__(self, "parity", int(self.parity))

    @classmethod
    def parse(cls, text: str) -> Irrep:
        """Parse one irrep written ``<l><parity>``, such as ``"1o"``.

        Stricter than :meth:`Irreps.parse` on purpose: a multiplicity, a
        ``+`` or surrounding whitespace means the caller passed a declaration
        where one irrep was expected, and dropping the rest would silently
        answer a different question.

        Raises:
            IrrepsError: If ``text`` is anything else. The message quotes it.
        """
        match = _SINGLE.fullmatch(text) if isinstance(text, str) else None
        if match is None:
            raise IrrepsError(
                f"{text!r} is not a single irrep. Expected '<l><parity>' with "
                f"no multiplicity, no '+' and no spaces, for example '0e' or '1o'."
            )
        degree, parity = match.groups()
        return cls(int(degree), 1 if parity == "e" else -1)

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


@dataclass(frozen=True)
class Irreps:
    """A direct sum of irreps with multiplicities, in the order written.

    The order is preserved rather than sorted: it is the layout of the values
    themselves, and sorting it would silently move every weight.

    Attributes:
        terms: ``(multiplicity, irrep)`` pairs, at least one, each multiplicity
            an integer of at least 1.

    Raises:
        IrrepsError: If there are no terms, a term is not a
            ``(multiplicity, Irrep)`` pair, or a multiplicity is not a positive
            integer. A term with multiplicity 0 is refused rather than kept,
            because it would still answer ``ir in irreps`` with yes while
            contributing no components.
    """

    terms: tuple[tuple[int, Irrep], ...]

    def __post_init__(self) -> None:
        terms = tuple(tuple(term) for term in self.terms)
        if not terms:
            raise IrrepsError(f"the declaration is empty. Expected {IRREPS_GRAMMAR}")
        for term in terms:
            if len(term) != 2 or not isinstance(term[1], Irrep):
                raise IrrepsError(
                    f"{term!r} is not a (multiplicity, Irrep) pair, for example "
                    f"(16, Irrep(0, 1))."
                )
            multiplicity, irrep = term
            if not _is_integer(multiplicity) or multiplicity < 1:
                raise IrrepsError(
                    f"the multiplicity of {irrep} is {multiplicity!r}, and it must "
                    f"be an integer of at least 1. Leave the term out instead of "
                    f"declaring it with no copies."
                )
        object.__setattr__(
            self, "terms", tuple((int(mul), irrep) for mul, irrep in terms)
        )

    @classmethod
    def parse(cls, text: str) -> Irreps:
        """Parse a declaration such as ``"128x0e+128x1o"``.

        Raises:
            IrrepsError: If the declaration is empty, a term is malformed, or a
                term has multiplicity 0. The message quotes the offending term.
        """
        try:
            terms = parse_terms(text)
        except ValueError as failure:
            raise IrrepsError(f"{failure} Expected {IRREPS_GRAMMAR}") from failure
        return cls(
            tuple(
                (copies, Irrep(degree, 1 if parity == "e" else -1))
                for copies, degree, parity in terms
            )
        )

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
