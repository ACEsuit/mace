"""Closed, framework-free shapes accepted by the optional span factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    Descriptor,
    DistanceTransformDescriptor,
    RadialBasisDescriptor,
    RadialMLPDescriptor,
    SphericalHarmonicsDescriptor,
)

__all__ = ["EdgeConvolutionSpan"]


@dataclass(frozen=True)
class EdgeConvolutionSpan:
    """Basis/envelope, harmonics, radial network and receiver-summed convolution.

    The tuple order preserves the request made by the interaction blocks:
    basis, harmonics, optional transform, radial network, optional density,
    convolution. Element conditioning is described by the radial network.
    All members use the chain's activation layout. Other radial families can
    define their own shape and matcher over the same descriptor tuple protocol.
    """

    radial_basis: RadialBasisDescriptor
    spherical_harmonics: SphericalHarmonicsDescriptor
    radial_network: RadialMLPDescriptor
    convolution: ChannelwiseTPConvDescriptor
    distance_transform: DistanceTransformDescriptor | None = None
    density: RadialMLPDescriptor | None = None

    def __post_init__(self) -> None:
        for name, kind in (
            ("radial_basis", RadialBasisDescriptor),
            ("spherical_harmonics", SphericalHarmonicsDescriptor),
            ("radial_network", RadialMLPDescriptor),
            ("convolution", ChannelwiseTPConvDescriptor),
        ):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"{name} must be {kind.__name__}.")
        if self.distance_transform is not None and not isinstance(
            self.distance_transform, DistanceTransformDescriptor
        ):
            raise TypeError(
                "distance_transform must be DistanceTransformDescriptor or None."
            )
        if self.density is not None:
            if not isinstance(self.density, RadialMLPDescriptor):
                raise TypeError("density must be RadialMLPDescriptor or None.")
            if self.density.channels[-1] != 1:
                raise ValueError("density must produce one scalar per edge.")
            for name in ("num_elements", "element_embedding_width"):
                if getattr(self.density, name) != getattr(self.radial_network, name):
                    raise ValueError(
                        "density and radial network must share conditioning."
                    )
            if self.density.channels[0] != self.radial_network.channels[0]:
                raise ValueError("density and radial network must share input widths.")

    @property
    def descriptors(self) -> tuple[Descriptor, ...]:
        return (
            (self.radial_basis, self.spherical_harmonics)
            + (() if self.distance_transform is None else (self.distance_transform,))
            + (self.radial_network,)
            + (() if self.density is None else (self.density,))
            + (self.convolution,)
        )

    @classmethod
    def from_descriptors(
        cls, descriptors: tuple[Descriptor, ...]
    ) -> EdgeConvolutionSpan | None:
        """Match exactly this shape; never infer a missing or reordered member."""
        kinds = tuple(type(member) for member in descriptors)
        prefix = (RadialBasisDescriptor, SphericalHarmonicsDescriptor)
        if kinds[:2] != prefix:
            return None
        offset = 2
        transform = None
        if kinds[offset : offset + 1] == (DistanceTransformDescriptor,):
            transform = cast(DistanceTransformDescriptor, descriptors[offset])
            offset += 1
        suffix = kinds[offset:]
        if suffix not in (
            (RadialMLPDescriptor, ChannelwiseTPConvDescriptor),
            (RadialMLPDescriptor, RadialMLPDescriptor, ChannelwiseTPConvDescriptor),
        ):
            return None
        try:
            return cls(
                radial_basis=cast(RadialBasisDescriptor, descriptors[0]),
                spherical_harmonics=cast(SphericalHarmonicsDescriptor, descriptors[1]),
                radial_network=cast(RadialMLPDescriptor, descriptors[offset]),
                convolution=cast(ChannelwiseTPConvDescriptor, descriptors[-1]),
                distance_transform=transform,
                density=cast(RadialMLPDescriptor, descriptors[offset + 1])
                if len(suffix) == 3
                else None,
            )
        except ValueError:
            # Two networks in the tuple need not have this shape's semantics.
            return None
