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

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from torch import Tensor

__all__ = ["grouped", "per_channel", "term_widths"]


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
    return _join([block.reshape(features.shape[0], -1) for block in blocks])


def _join(blocks: list[Tensor]) -> Tensor:
    """The blocks along the last axis, without a copy when there is only one."""
    if len(blocks) == 1:
        return blocks[0]
    return torch.cat(blocks, dim=-1)
