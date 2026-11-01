"""The grouped declaration of the node features.

Between ops the node features are grouped by irrep: one block per term, each
holding that term's channels. How a block is laid out inside is the backend's
choice, and :mod:`mace_torch.backends.layout` is what reads it.
"""

from __future__ import annotations

from mace_core.clebsch_gordan.irreps import Irreps

__all__ = ["expanded_irreps", "term_widths"]


def expanded_irreps(hidden: str, num_features: int) -> str:
    """The declaration of ``num_features`` channels of ``hidden``, grouped:
    one term per term of ``hidden``, with its multiplicity times the channels.
    """
    parsed = Irreps.parse(hidden)
    return "+".join(f"{num_features * mul}x{ir}" for mul, ir in parsed.terms)


def term_widths(hidden: str) -> list[int]:
    """One channel's width of each term of a declaration, in order."""
    return [mul * ir.dimension for mul, ir in Irreps.parse(hidden).terms]
