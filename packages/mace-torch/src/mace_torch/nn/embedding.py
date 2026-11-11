"""Node and edge embeddings: the element embedding and the radial embedding.

Both are the first thing a MACE model does with a graph. The node embedding
maps each atom's element to a vector of scalar channels; the radial embedding
maps each edge's length to a vector of radial features and the smooth cutoff
envelope that goes with them.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from mace_core.config import (
    AgnesiTransformConfig,
    BesselBasisConfig,
    ChebyshevBasisConfig,
    DistanceTransformConfig,
    GaussianBasisConfig,
    NoDistanceTransformConfig,
    PolynomialCutoffConfig,
    RadialBasisConfig,
    SoftTransformConfig,
)
from mace_core.kernels.precision import Precision
from mace_core.kernels.specs import RadialBasisSpec

from mace_torch.nn.radial import (
    AgnesiTransform,
    SoftTransform,
)

__all__ = [
    "DistanceTransform",
    "LinearNodeEmbeddingBlock",
    "RadialEmbeddingBlock",
    "build_distance_transform",
    "radial_basis_spec",
]

DistanceTransform = AgnesiTransform | SoftTransform


class LinearNodeEmbeddingBlock(torch.nn.Module):
    """One-hot element attributes -> scalar node channels, a plain weight matrix.

    The element attributes are scalars (``l = 0``), so the equivariant linear
    layer legacy used reduces to ``node_attributes @ weight / sqrt(num_elements)``
    with no bias.
    """

    def __init__(self, num_elements: int, num_channels: int):
        super().__init__()
        self.num_elements = num_elements
        self.num_channels = num_channels
        self.normalisation = 1.0 / math.sqrt(num_elements)
        self.weight = torch.nn.Parameter(
            torch.randn(num_elements, num_channels, dtype=torch.get_default_dtype())
        )

    def forward(self, node_attributes: torch.Tensor) -> torch.Tensor:
        """``[n_nodes, num_elements]`` one-hot -> ``[n_nodes, num_channels]``."""
        return (node_attributes @ self.weight) * self.normalisation

    def extra_repr(self) -> str:
        return f"num_elements={self.num_elements}, num_channels={self.num_channels}"


class RadialEmbeddingBlock(torch.nn.Module):
    """The radial basis and the cutoff envelope of every edge, with an optional
    distance transform.

    The basis and its envelope are one op, the one the backend built from a
    :class:`~mace_core.kernels.specs.RadialBasisSpec`; the block adds the
    distance transform, which the model holds, and hands the envelope on to the
    interaction blocks.

    The forward runs three steps in a fixed order that is load-bearing:

    1. the cutoff envelope is computed on the **raw** edge lengths;
    2. the distance transform, if there is one, is applied to the lengths;
    3. the basis is evaluated on the (transformed) lengths.

    Both results are returned, always: the radial features ``[n_edges, num_basis]``
    and the envelope ``[n_edges, 1]``. With ``apply_cutoff`` (legacy's
    ``--apply_cutoff``, the default) the features are already ``basis * cutoff``;
    without it they are the bare basis and the consumer multiplies the envelope
    in later, after its radial MLP.
    """

    def __init__(
        self,
        radial_basis: torch.nn.Module,
        distance_transform: DistanceTransform | None,
    ):
        super().__init__()
        self.radial = radial_basis
        self.distance_transform = distance_transform
        self.kernel_spec = cast(RadialBasisSpec, radial_basis.kernel_spec)

    @property
    def num_basis(self) -> int:
        return self.kernel_spec.num_basis

    @property
    def apply_cutoff(self) -> bool:
        return self.kernel_spec.apply_cutoff

    @property
    def cutoff(self) -> torch.nn.Module:
        """The envelope, which the forward evaluates on the raw lengths."""
        return cast(torch.nn.Module, self.radial.cutoff)

    def forward(
        self,
        edge_lengths: torch.Tensor,
        node_atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``edge_lengths`` is ``[n_edges, 1]`` in Angstrom.

        Returns ``(edge_radial_features, edge_cutoff)``: the basis on the
        (transformed) lengths, ``[n_edges, num_basis]``, multiplied by the
        envelope when ``apply_cutoff`` is set, and the envelope on the raw
        lengths, ``[n_edges, 1]``.
        """
        edge_cutoff = self.cutoff(edge_lengths)  # on the raw lengths, always
        if self.distance_transform is not None:
            edge_lengths = self.distance_transform(
                edge_lengths, node_atomic_numbers, edge_index
            )
        edge_radial_features = self.radial(edge_lengths, envelope=edge_cutoff)
        return edge_radial_features, edge_cutoff

    def extra_repr(self) -> str:
        return f"apply_cutoff={self.apply_cutoff}"


# ---------------------------------------------------------------------------
# Config section -> specification or module
# ---------------------------------------------------------------------------


def radial_basis_spec(
    radial_basis: RadialBasisConfig,
    cutoff: PolynomialCutoffConfig,
    r_max: float,
    *,
    apply_cutoff: bool = True,
    precision: Precision = "float64",
) -> RadialBasisSpec:
    """The kernel specification of the basis and envelope the sections describe.

    ``r_max`` is in Angstrom and sets both; the Chebyshev basis does not use it.
    Every option of the basis section reaches the specification, so a backend
    sees the whole radial embedding it is asked to build.
    """
    return RadialBasisSpec(
        kind=radial_basis.kind,
        num_basis=radial_basis.num_basis,
        cutoff=r_max,
        cutoff_order=cutoff.polynomial_order,
        trainable=(
            radial_basis.trainable
            if isinstance(radial_basis, BesselBasisConfig | GaussianBasisConfig)
            else False
        ),
        include_constant=(
            radial_basis.include_constant
            if isinstance(radial_basis, ChebyshevBasisConfig)
            else False
        ),
        apply_cutoff=apply_cutoff,
        precision=precision,
    )


def build_distance_transform(
    config: DistanceTransformConfig,
) -> DistanceTransform | None:
    """The transform a config section describes; ``None`` for the kind ``none``."""
    match config:
        case NoDistanceTransformConfig():
            return None
        case AgnesiTransformConfig():
            return AgnesiTransform(
                exponent_q=config.exponent_q,
                exponent_p=config.exponent_p,
                amplitude=config.amplitude,
                trainable=config.trainable,
            )
        case SoftTransformConfig():
            return SoftTransform(steepness=config.steepness, trainable=config.trainable)
        case _:
            raise TypeError(
                f"no distance transform is built from {type(config).__name__}"
            )
