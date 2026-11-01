"""The two arrangements of channelled node features, and the moves between them.

Between ops the node features are **grouped**: the canonical ``mul_ir`` over the
declaration expanded to ``num_features`` channels, one contiguous block per
term. That is the layout a checkpoint holds and a user is handed, and the one
every op of the model reads and writes.

The reference kernels compute **channel by channel**, ``[n, num_features,
dim]`` with one channel's whole declaration along the last axis. They move
between the two here, at the backend boundary, which is the only place a
layout move is allowed: the model never permutes its features.

Within one term of multiplicity ``m``, copy ``k`` of channel ``c`` is copy
``c * m + k`` of the grouped term.
"""

from __future__ import annotations

from functools import cache

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from torch import Tensor

__all__ = ["expanded_irreps", "grouped", "per_channel", "term_widths"]


def term_widths(irreps: str) -> tuple[int, ...]:
    """One channel's width of each term of a declaration, in order.

    Called when an op is built, so that nothing is parsed in ``forward``.
    """
    return tuple(mul * ir.dimension for mul, ir in Irreps.parse(irreps).terms)


def per_channel(features: Tensor, widths: tuple[int, ...], num_features: int) -> Tensor:
    """Grouped ``[n, num_features * sum(widths)]`` to
    ``[n, num_features, sum(widths)]``, one channel per row of the middle axis."""
    blocks = features.split([num_features * width for width in widths], dim=-1)
    return _join(
        [
            block.reshape(features.shape[0], num_features, width)
            for block, width in zip(blocks, widths, strict=True)
        ]
    )


def grouped(features: Tensor, widths: tuple[int, ...]) -> Tensor:
    """``[n, num_features, sum(widths)]`` to grouped
    ``[n, num_features * sum(widths)]``, the inverse of :func:`per_channel`."""
    blocks = features.split(list(widths), dim=-1)
    return _join(
        [
            block.reshape(features.shape[0], features.shape[1] * width)
            for block, width in zip(blocks, widths, strict=True)
        ]
    )


def _join(blocks: list[Tensor]) -> Tensor:
    """The blocks along the last axis, without a copy when there is only one."""
    if len(blocks) == 1:
        return blocks[0]
    return torch.cat(blocks, dim=-1)


def expanded_irreps(irreps: str, num_features: int) -> str:
    """The grouped declaration of ``num_features`` channels of ``irreps``.

    One term per term of ``irreps``, its multiplicity times the channel count,
    in the same order. ``"0e+1o"`` at 8 channels is ``"8x0e+8x1o"``.
    """
    return "+".join(
        f"{num_features * mul}x{ir}" for mul, ir in Irreps.parse(irreps).terms
    )


Terms = tuple[tuple[int, int], ...]


@cache
def terms(irreps: str) -> Terms:
    """Multiplicity and component width per canonical term, parsed at build."""
    return tuple((mul, ir.dimension) for mul, ir in Irreps.parse(irreps).terms)


def blocks(features: Tensor, declaration: Terms) -> list[Tensor]:
    """Canonical ``[nodes, dim]`` as ``[nodes, multiplicity, components]`` views."""
    nodes = features.shape[0]
    pieces, offset = [], 0
    for mul, dimension in declaration:
        pieces.append(
            features[:, offset : offset + mul * dimension].reshape(
                nodes, mul, dimension
            )
        )
        offset += mul * dimension
    return pieces


def join(pieces: list[Tensor]) -> Tensor:
    """Canonical blocks flattened and joined, preserving channel order."""
    return torch.cat(
        [
            block.reshape(block.shape[0], block.shape[1] * block.shape[2])
            for block in pieces
        ],
        dim=-1,
    )
