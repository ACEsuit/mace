"""Where a component sits in a flat node-feature vector.

The node features are held as ``[n, channel, component]``. Read as one flat
vector that is channel-major, while an equivariant linear map wants its input
grouped by irrep. These two say what that regrouping is, and they live here
rather than beside a readout because both the readouts above and the input
streams below need them.
"""

from __future__ import annotations

import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from torch import Tensor

__all__ = [
    "channel_layout_index",
    "channel_major",
    "expanded_irreps",
    "grouped",
    "inverse_layout_index",
    "path_layout_index",
]


def expanded_irreps(hidden: str, num_features: int) -> str:
    """The declaration of ``num_features`` channels of ``hidden``, grouped.

    The node features are held as ``[n, channel, component]``. Read as one flat
    vector that is channel-major, and an equivariant linear map wants its input
    grouped by irrep instead. This is the grouped declaration;
    :func:`channel_layout_index` is the permutation that gets there.
    """
    parsed = Irreps.parse(hidden)
    return "+".join(f"{num_features * mul}x{ir}" for mul, ir in parsed.terms)


def channel_layout_index(hidden: str, num_features: int) -> np.ndarray:
    """Where each entry of the grouped layout reads from the channel-major one.

    Args:
        hidden: One channel's declaration.
        num_features: How many channels.

    Returns:
        ``[dim]`` of int64, to be used as ``flat[..., index]``.
    """
    parsed = Irreps.parse(hidden)
    width = parsed.dimension
    index = np.empty(num_features * width, dtype=np.int64)
    target = 0
    source_offset = 0
    for mul, ir in parsed.terms:
        span = ir.dimension
        for channel in range(num_features):
            for copy in range(mul):
                start = channel * width + source_offset + copy * span
                index[target : target + span] = np.arange(start, start + span)
                target += span
        source_offset += mul * span
    return index


def inverse_layout_index(hidden: str, num_features: int) -> np.ndarray:
    """The permutation back from the grouped layout to the channel-major one.

    An input stream is mapped into the grouped layout by an equivariant linear,
    and has to be written back into the channel-major features it is added to.
    """
    forward = channel_layout_index(hidden, num_features)
    inverse = np.empty_like(forward)
    inverse[forward] = np.arange(len(forward), dtype=forward.dtype)
    return inverse


def path_layout_index(paths, num_features: int) -> np.ndarray:
    """From the convolution's channel-major output to the linear's grouped one.

    The convolution returns ``[nodes, channel, component]``, which read flat is
    channel-major. The linear after it wants one contiguous block per path,
    each holding that path's channels. This is the permutation, as
    ``flat[..., index]``.

    Args:
        paths: The convolution's paths, in their pinned order.
        num_features: The channel count.
    """
    per_channel = sum(path.irrep.dimension for path in paths)
    index = np.empty(num_features * per_channel, dtype=np.int64)
    target = 0
    source_offset = 0
    for path in paths:
        span = path.irrep.dimension
        for channel in range(num_features):
            start = channel * per_channel + source_offset
            index[target : target + span] = np.arange(start, start + span)
            target += span
        source_offset += span
    return index


def _term_widths(hidden: str) -> list[int]:
    return [mul * ir.dimension for mul, ir in Irreps.parse(hidden).terms]


def channel_major(features: Tensor, hidden: str, num_features: int) -> Tensor:
    """Grouped ``[n, num_features * dim]`` in, ``[n, num_features, dim]`` out.

    The same map as ``features[..., inverse_layout_index(...)]``, written as one
    view per term and a concatenation. Within a term the grouped layout is
    already channel-major, so each term is a view and only the join copies. Its
    backward is a split rather than the accumulating scatter an index gather
    differentiates into, which is the reason for writing it this way.
    """
    nodes = features.shape[0]
    pieces, offset = [], 0
    for width in _term_widths(hidden):
        span = num_features * width
        pieces.append(
            features[:, offset : offset + span].reshape(nodes, num_features, width)
        )
        offset += span
    return torch.cat(pieces, dim=-1)


def grouped(channels: Tensor, hidden: str) -> Tensor:
    """``[n, num_features, dim]`` in, grouped ``[n, num_features * dim]`` out.

    The inverse of :func:`channel_major`, the same map as
    ``flat[..., channel_layout_index(...)]``.
    """
    nodes = channels.shape[0]
    pieces = torch.split(channels, _term_widths(hidden), dim=-1)
    return torch.cat(
        [piece.reshape(nodes, piece.shape[1] * piece.shape[2]) for piece in pieces],
        dim=-1,
    )
