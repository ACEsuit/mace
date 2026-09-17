"""Node and edge embeddings: the element embedding and the radial embedding.

Both are the first thing a MACE model does with a graph. The node embedding
maps each atom's element to a vector of scalar channels; the radial embedding
maps each edge's length to a vector of radial features and the smooth cutoff
envelope that goes with them.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from mace_torch.nn.radial import (
    AgnesiTransform,
    BesselBasis,
    ChebyshevBasis,
    GaussianBasis,
    PolynomialCutoff,
    SoftTransform,
)

__all__ = [
    "DistanceTransformKind",
    "LinearNodeEmbeddingBlock",
    "RadialBasisKind",
    "RadialEmbeddingBlock",
]

RadialBasisKind = Literal["bessel", "gaussian", "chebyshev"]
DistanceTransformKind = Literal["none", "agnesi", "soft"]


class LinearNodeEmbeddingBlock(torch.nn.Module):
    """One-hot element attributes -> scalar node channels, a plain weight matrix.

    The element attributes are scalars (``l = 0``), so the equivariant linear
    layer legacy used reduces to ``node_attributes @ weight`` with no bias. The
    weight is initialised ``N(0, 1 / num_elements)``, which is the distribution
    an equivariant linear layer with ``N(0, 1)`` weights and ``1 / sqrt(fan_in)``
    path normalisation produces; a legacy weight ``W`` maps to
    ``W / sqrt(num_elements)`` here.
    """

    def __init__(self, num_elements: int, num_channels: int):
        super().__init__()
        self.num_elements = num_elements
        self.num_channels = num_channels
        self.weight = torch.nn.Parameter(
            torch.randn(num_elements, num_channels, dtype=torch.get_default_dtype())
            / math.sqrt(num_elements)
        )

    def forward(self, node_attributes: torch.Tensor) -> torch.Tensor:
        """``[n_nodes, num_elements]`` one-hot -> ``[n_nodes, num_channels]``."""
        return node_attributes @ self.weight

    def extra_repr(self) -> str:
        return f"num_elements={self.num_elements}, num_channels={self.num_channels}"


class RadialEmbeddingBlock(torch.nn.Module):
    """The radial basis and the cutoff envelope of every edge, with an optional
    distance transform.

    The forward runs three steps in a fixed order that is load-bearing:

    1. the cutoff envelope is computed on the **raw** edge lengths;
    2. the distance transform, if configured, is applied to the lengths;
    3. the basis is evaluated on the (transformed) lengths.

    The two results are returned side by side, never multiplied: where the
    envelope enters is the consumer's decision, not this block's.
    """

    def __init__(
        self,
        r_max: float,
        num_basis: int,
        num_polynomial_cutoff: int,
        radial_basis: RadialBasisKind = "bessel",
        distance_transform: DistanceTransformKind = "none",
    ):
        super().__init__()
        self.basis: torch.nn.Module
        if radial_basis == "bessel":
            self.basis = BesselBasis(r_max=r_max, num_basis=num_basis)
        elif radial_basis == "gaussian":
            self.basis = GaussianBasis(r_max=r_max, num_basis=num_basis)
        elif radial_basis == "chebyshev":
            self.basis = ChebyshevBasis(num_basis=num_basis)
        else:
            raise ValueError(
                f"radial_basis={radial_basis!r} is not one of "
                f"'bessel', 'gaussian', 'chebyshev'"
            )
        self.distance_transform: torch.nn.Module | None
        if distance_transform == "none":
            self.distance_transform = None
        elif distance_transform == "agnesi":
            self.distance_transform = AgnesiTransform()
        elif distance_transform == "soft":
            self.distance_transform = SoftTransform()
        else:
            raise ValueError(
                f"distance_transform={distance_transform!r} is not one of "
                f"'none', 'agnesi', 'soft'"
            )
        self.cutoff = PolynomialCutoff(
            r_max=r_max, polynomial_order=num_polynomial_cutoff
        )
        self.num_basis = num_basis

    def forward(
        self,
        edge_lengths: torch.Tensor,
        node_atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``edge_lengths`` is ``[n_edges, 1]`` in Angstrom.

        Returns ``(edge_radial_features, edge_cutoff)``: the basis on the
        (transformed) lengths, ``[n_edges, num_basis]``, and the envelope on
        the raw lengths, ``[n_edges, 1]``.
        """
        edge_cutoff = self.cutoff(edge_lengths)  # on the raw lengths, always
        if self.distance_transform is not None:
            edge_lengths = self.distance_transform(
                edge_lengths, node_atomic_numbers, edge_index
            )
        edge_radial_features = self.basis(edge_lengths)
        return edge_radial_features, edge_cutoff
