"""Node and edge embeddings: the element embedding and the radial embedding.

Both are the first thing a MACE model does with a graph. The node embedding
maps each atom's element to a vector of scalar channels; the radial embedding
maps each edge's length to a vector of radial features multiplied by a smooth
cutoff envelope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    "RadialEmbedding",
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_elements={self.num_elements}, "
            f"num_channels={self.num_channels})"
        )


@dataclass(frozen=True)
class RadialEmbedding:
    """The radial embedding of every edge.

    ``edge_radial_features`` is ``[n_edges, num_basis]``. When the block applied
    the cutoff, it is already the product with the envelope and ``edge_cutoff``
    is ``None``. Otherwise the features are the bare basis and ``edge_cutoff``
    (``[n_edges, 1]``) is the envelope the consumer multiplies in later; the
    product of the two is bit-for-bit the applied branch.
    """

    edge_radial_features: torch.Tensor
    edge_cutoff: torch.Tensor | None


class RadialEmbeddingBlock(torch.nn.Module):
    """Basis x cutoff for every edge, with an optional distance transform.

    The forward runs three steps in a fixed order that is load-bearing:

    1. the cutoff envelope is computed on the **raw** edge lengths;
    2. the distance transform, if configured, is applied to the lengths;
    3. the basis is evaluated on the (transformed) lengths.

    With ``apply_cutoff`` the block returns ``basis * cutoff``; otherwise it
    returns the bare basis and the envelope side by side, for a consumer that
    applies the envelope itself (``--apply_cutoff False``).
    """

    def __init__(
        self,
        r_max: float,
        num_basis: int,
        num_polynomial_cutoff: int,
        radial_basis: RadialBasisKind = "bessel",
        distance_transform: DistanceTransformKind = "none",
        apply_cutoff: bool = True,
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
        self.apply_cutoff = apply_cutoff

    def forward(
        self,
        edge_lengths: torch.Tensor,
        node_atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> RadialEmbedding:
        """``edge_lengths`` is ``[n_edges, 1]`` in Angstrom.

        See :class:`RadialEmbedding` for the two return shapes.
        """
        edge_cutoff = self.cutoff(edge_lengths)  # on the raw lengths, always
        if self.distance_transform is not None:
            edge_lengths = self.distance_transform(
                edge_lengths, node_atomic_numbers, edge_index
            )
        edge_radial_features = self.basis(edge_lengths)
        if not self.apply_cutoff:
            return RadialEmbedding(edge_radial_features, edge_cutoff)
        return RadialEmbedding(edge_radial_features * edge_cutoff, None)
