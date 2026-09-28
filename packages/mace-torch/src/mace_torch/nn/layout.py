"""Where a component sits in a flat node-feature vector.

Between ops the node features are grouped by irrep: one block per term, each
holding that term's channels. A kernel that works channel by channel wants
``[n, channel, component]`` instead. These say what the grouped declaration is
and move values between the two.
"""

from __future__ import annotations

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from torch import Tensor

__all__ = [
    "channel_major",
    "expanded_irreps",
    "grouped",
    "term_widths",
]


def expanded_irreps(hidden: str, num_features: int) -> str:
    """The declaration of ``num_features`` channels of ``hidden``, grouped:
    one term per term of ``hidden``, with its multiplicity times the channels.
    """
    parsed = Irreps.parse(hidden)
    return "+".join(f"{num_features * mul}x{ir}" for mul, ir in parsed.terms)


def term_widths(hidden: str) -> list[int]:
    """One channel's width of each term of a declaration, in order."""
    return [mul * ir.dimension for mul, ir in Irreps.parse(hidden).terms]


def channel_major(features: Tensor, widths: list[int], num_features: int) -> Tensor:
    """Grouped ``[n, num_features * dim]`` in, ``[n, num_features, dim]`` out.

    ``widths`` are one channel's width of each block: :func:`term_widths` for a
    declaration, the path widths for a convolution's output. Within a block the
    grouped layout is already channel-major, so each block is a view and only
    the join copies. Its backward is a split, rather than the accumulating
    scatter a gather by index differentiates into.
    """
    nodes = features.shape[0]
    pieces, offset = [], 0
    for width in widths:
        span = num_features * width
        pieces.append(
            features[:, offset : offset + span].reshape(nodes, num_features, width)
        )
        offset += span
    return torch.cat(pieces, dim=-1)


def grouped(channels: Tensor, widths: list[int]) -> Tensor:
    """``[n, num_features, dim]`` in, grouped ``[n, num_features * dim]`` out.

    The inverse of :func:`channel_major`: one block per term, or per path,
    each holding that block's channels.
    """
    nodes = channels.shape[0]
    pieces = torch.split(channels, widths, dim=-1)
    return torch.cat(
        [piece.reshape(nodes, piece.shape[1] * piece.shape[2]) for piece in pieces],
        dim=-1,
    )
