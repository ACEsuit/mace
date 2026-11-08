"""Radial edge features and their cutoff envelope, built from model settings."""

from __future__ import annotations

from typing import Any

import torch
from mace_core.config import (
    AgnesiTransformConfig,
    DistanceTransformConfig,
    ModelConfig,
    NoDistanceTransformConfig,
    SoftTransformConfig,
)
from mace_core.kernels.descriptors import RadialBasisDescriptor
from mace_core.kernels.precision import Precision

from mace_torch.backends.reference import ReferenceBackend
from mace_torch.nn.radial import (
    AgnesiTransform,
    SoftTransform,
)

__all__ = [
    "DistanceTransform",
    "RadialEmbeddingBlock",
    "build_distance_transform",
    "build_radial_embedding",
]

DistanceTransform = AgnesiTransform | SoftTransform


class RadialEmbeddingBlock(torch.nn.Module):
    """The radial basis and the cutoff envelope of every edge, with an optional
    distance transform.

    The backend supplies the radial operator and its cutoff. This block adds
    the distance transform and returns the envelope for the interaction layers.

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
        radial_basis: Any,
        distance_transform: DistanceTransform | None,
    ):
        super().__init__()
        self.radial = radial_basis
        self.distance_transform = distance_transform

    @property
    def num_basis(self) -> int:
        return self.radial.descriptor.num_basis

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
        edge_cutoff = self.radial.cutoff(edge_lengths)  # on the raw lengths, always
        if self.distance_transform is not None:
            edge_lengths = self.distance_transform(
                edge_lengths, node_atomic_numbers, edge_index
            )
        edge_radial_features = self.radial(edge_lengths, envelope=edge_cutoff)
        return edge_radial_features, edge_cutoff

    def extra_repr(self) -> str:
        return f"apply_cutoff={self.radial.descriptor.apply_cutoff}"

    def span_weights(self) -> tuple[torch.Tensor, ...]:
        """Live canonical edge-member tensors in the public span call order."""
        transform = (
            ()
            if self.distance_transform is None
            else self.distance_transform.span_weights()
        )
        return (*self.radial.span_weights(), *transform)

    def canonical_components(self) -> dict[str, torch.nn.Module | None]:
        """Stable checkpoint roles, independent of implementation attributes."""
        return {
            **self.radial.canonical_components(),
            "distance_transform": self.distance_transform,
        }


# ---------------------------------------------------------------------------
# Config section -> module
# ---------------------------------------------------------------------------


def build_radial_embedding(
    backend: Any,
    config: ModelConfig,
    precision: Precision,
) -> RadialEmbeddingBlock:
    """Build every basis through the backend, preserving all model options."""
    basis = config.radial_basis
    descriptor = RadialBasisDescriptor(
        **basis.model_dump(),
        cutoff=config.r_max,
        cutoff_order=config.cutoff.polynomial_order,
        apply_cutoff=config.apply_cutoff,
        precision=precision,
    )
    radial = backend.make_radial_basis(descriptor)
    if radial is None:
        radial = ReferenceBackend().make_radial_basis(descriptor)
    return RadialEmbeddingBlock(
        radial, build_distance_transform(config.distance_transform)
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
