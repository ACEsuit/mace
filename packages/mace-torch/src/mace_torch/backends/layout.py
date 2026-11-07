"""Activation views in the layout selected once for the complete model chain.

Within each irrep, mul_ir stores [multiplicity, components] and ir_mul stores
[components, multiplicity]. Checkpoint weights and public feature outputs use
mul_ir. Operators consume and produce the chain's negotiated activation layout.
"""

# Layout views and negotiation follow reforge/perf-integration (8dee319d).

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import ActivationLayout
from torch import Tensor

__all__ = [
    "CANONICAL",
    "Layout",
    "Terms",
    "expanded_irreps",
    "grouped",
    "layout_of",
    "per_channel",
    "term_widths",
]


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
    return CANONICAL.blocks(features, declaration)


def join(pieces: list[Tensor]) -> Tensor:
    """Canonical blocks flattened and joined, preserving channel order."""
    return CANONICAL.join(pieces)


@dataclass(frozen=True)
class Layout:
    """An immutable activation layout and its term-wise views."""

    name: ActivationLayout

    def __post_init__(self):
        if self.name not in {"mul_ir", "ir_mul"}:
            raise ValueError(f"unknown activation layout {self.name!r}")

    @property
    def is_canonical(self) -> bool:
        return self.name == "mul_ir"

    @staticmethod
    def terms(irreps: str) -> Terms:
        return terms(irreps)

    def blocks(self, features: Tensor, declaration: Terms) -> list[Tensor]:
        """Views [nodes, multiplicity, components], with shared storage."""
        pieces, offset = [], 0
        for mul, dimension in declaration:
            block = features[:, offset : offset + mul * dimension]
            if self.is_canonical:
                block = block.reshape(features.shape[0], mul, dimension)
            else:
                block = block.reshape(features.shape[0], dimension, mul).transpose(1, 2)
            pieces.append(block)
            offset += mul * dimension
        return pieces

    def join(self, pieces: list[Tensor]) -> Tensor:
        """Flatten term-wise [nodes, multiplicity, components] in this layout."""
        if not self.is_canonical:
            pieces = [piece.transpose(1, 2) for piece in pieces]
        return _join(
            [
                piece.reshape(piece.shape[0], piece.shape[1] * piece.shape[2])
                for piece in pieces
            ]
        )

    def channel_major(
        self, features: Tensor, declaration: Terms, channels: int
    ) -> Tensor:
        """Return [nodes, channels, single_channel_dimension] for reference math."""
        expanded = tuple((channels * mul, dimension) for mul, dimension in declaration)
        return _join(
            [
                block.reshape(features.shape[0], channels, mul * dimension)
                for block, (mul, dimension) in zip(
                    self.blocks(features, expanded), declaration, strict=True
                )
            ]
        )

    def grouped(self, features: Tensor, declaration: Terms) -> Tensor:
        """Return flat chain-layout features from channel-major reference math."""
        widths = [mul * dimension for mul, dimension in declaration]
        pieces = features.split(widths, dim=-1)
        return self.join(
            [
                piece.reshape(features.shape[0], features.shape[1] * mul, dimension)
                for piece, (mul, dimension) in zip(pieces, declaration, strict=True)
            ]
        )

    def positions(self, irreps: str) -> np.ndarray:
        """Index of each canonical component in this layout, resolved at build."""
        return _positions(irreps, self.name)

    def from_canonical(self, features: Tensor, declaration: Terms) -> Tensor:
        if self.is_canonical or all(mul == 1 or dim == 1 for mul, dim in declaration):
            return features
        return self.join(CANONICAL.blocks(features, declaration))

    def to_canonical(self, features: Tensor, declaration: Terms) -> Tensor:
        if self.is_canonical or all(mul == 1 or dim == 1 for mul, dim in declaration):
            return features
        return CANONICAL.join(self.blocks(features, declaration))


@cache
def _positions(irreps: str, name: ActivationLayout) -> np.ndarray:
    result, offset = [], 0
    for mul, dimension in terms(irreps):
        indices = np.arange(mul * dimension)
        if name == "ir_mul":
            indices = indices.reshape(dimension, mul).T.reshape(-1)
        result.extend(offset + indices)
        offset += mul * dimension
    positions = np.asarray(result, dtype=np.int64)
    positions.flags.writeable = False
    return positions


CANONICAL = Layout("mul_ir")


def layout_of(backend: Any) -> Layout:
    """The layout resolved by the chain builder, canonical for standalone ops."""
    return getattr(backend, "layout", CANONICAL)
