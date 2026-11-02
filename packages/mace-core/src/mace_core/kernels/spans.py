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

__all__ = ["EdgeConvolutionSpan", "supports_span_contract"]


def supports_span_contract(version: str) -> bool:
    """The stateless call contract starts at kernel spec 1.2."""
    parts = version.split(".")
    return (
        len(parts) == 2
        and all(part.isdecimal() for part in parts)
        and int(parts[0]) == 1
        and int(parts[1]) >= 2
    )


@dataclass(frozen=True)
class EdgeConvolutionSpan:
    """Basis/envelope, harmonics, radial network and receiver-summed convolution.

    The tuple order preserves the request made by the interaction blocks:
    basis, harmonics, optional transform, radial network, convolution. Extra
    members (density networks or element embeddings) are different shapes and
    are declined. All members use the chain's activation layout.
    """

    radial_basis: RadialBasisDescriptor
    spherical_harmonics: SphericalHarmonicsDescriptor
    radial_network: RadialMLPDescriptor
    convolution: ChannelwiseTPConvDescriptor
    distance_transform: DistanceTransformDescriptor | None = None

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

    @property
    def descriptors(self) -> tuple[Descriptor, ...]:
        return (
            (self.radial_basis, self.spherical_harmonics)
            + (() if self.distance_transform is None else (self.distance_transform,))
            + (self.radial_network, self.convolution)
        )

    @classmethod
    def from_descriptors(
        cls, descriptors: tuple[Descriptor, ...]
    ) -> EdgeConvolutionSpan | None:
        """Match exactly this shape; never infer a missing or reordered member."""
        kinds = tuple(type(member) for member in descriptors)
        prefix = (RadialBasisDescriptor, SphericalHarmonicsDescriptor)
        suffix = (RadialMLPDescriptor, ChannelwiseTPConvDescriptor)
        if kinds not in (
            prefix + suffix,
            (*prefix, DistanceTransformDescriptor, *suffix),
        ):
            return None
        return cls(
            radial_basis=cast(RadialBasisDescriptor, descriptors[0]),
            spherical_harmonics=cast(SphericalHarmonicsDescriptor, descriptors[1]),
            radial_network=cast(RadialMLPDescriptor, descriptors[-2]),
            convolution=cast(ChannelwiseTPConvDescriptor, descriptors[-1]),
            distance_transform=cast(DistanceTransformDescriptor, descriptors[2])
            if len(descriptors) == 5
            else None,
        )
