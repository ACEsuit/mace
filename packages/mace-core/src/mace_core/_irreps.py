"""Irreps syntax shared by the algebra and observable declarations."""

from __future__ import annotations

import re

_TERM = re.compile(r"(?:(\d+)x)?(\d+)([eo])", re.ASCII)


def parse_terms(
    text: str, *, canonical: bool = False
) -> tuple[tuple[int, int, str], ...]:
    """Preserve term order; canonical spelling additionally forbids leading zeros."""
    if not text or not text.strip():
        raise ValueError("the irreps declaration is empty.")
    terms = []
    for piece in text.split("+"):
        match = _TERM.fullmatch(piece.strip())
        if match is None:
            raise ValueError(
                f"{piece.strip()!r} is not a valid irreps term in {text!r}."
            )
        multiplicity, degree, parity = match.groups()
        copies = 1 if multiplicity is None else int(multiplicity)
        if copies == 0:
            raise ValueError(
                f"{piece.strip()!r} declares no copies; "
                "use a multiplicity of at least 1."
            )
        if canonical and any(
            number is not None and len(number) > 1 and number.startswith("0")
            for number in (multiplicity, degree)
        ):
            raise ValueError(f"{piece.strip()!r} contains leading zeros.")
        terms.append((copies, int(degree), parity))
    return tuple(terms)
