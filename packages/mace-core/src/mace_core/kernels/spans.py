"""Closed, framework-free shapes accepted by the optional span factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from mace_core.kernels.specs import (
    ChannelwiseTPConvSpec,
    DistanceTransformSpec,
    KernelSpec,
    RadialBasisSpec,
    RadialMLPSpec,
    SphericalHarmonicsSpec,
)

__all__ = ["EdgeConvolutionSpan"]


@dataclass(frozen=True)
class EdgeConvolutionSpan:
    """Basis/envelope, harmonics, radial network and receiver-summed convolution.

    The tuple order preserves the request made by the interaction blocks:
    basis, harmonics, optional transform, radial network, optional density,
    convolution. Element conditioning is described by the radial network.
    All members use the chain's activation layout. Other radial families can
    define their own shape and matcher over the same tuple of kernel specifications.
    """

    radial_basis: RadialBasisSpec
    spherical_harmonics: SphericalHarmonicsSpec
    radial_network: RadialMLPSpec
    convolution: ChannelwiseTPConvSpec
    distance_transform: DistanceTransformSpec | None = None
    density: RadialMLPSpec | None = None

    def __post_init__(self) -> None:
        for name, kind in (
            ("radial_basis", RadialBasisSpec),
            ("spherical_harmonics", SphericalHarmonicsSpec),
            ("radial_network", RadialMLPSpec),
            ("convolution", ChannelwiseTPConvSpec),
        ):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"{name} must be {kind.__name__}.")
        if self.distance_transform is not None and not isinstance(
            self.distance_transform, DistanceTransformSpec
        ):
            raise TypeError("distance_transform must be DistanceTransformSpec or None.")
        if self.density is not None:
            if not isinstance(self.density, RadialMLPSpec):
                raise TypeError("density must be RadialMLPSpec or None.")
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
    def kernel_specs(self) -> tuple[KernelSpec, ...]:
        return (
            (self.radial_basis, self.spherical_harmonics)
            + (() if self.distance_transform is None else (self.distance_transform,))
            + (self.radial_network,)
            + (() if self.density is None else (self.density,))
            + (self.convolution,)
        )

    @classmethod
    def from_specs(
        cls, kernel_specs: tuple[KernelSpec, ...]
    ) -> EdgeConvolutionSpan | None:
        """Match exactly this shape; never infer a missing or reordered member."""
        kinds = tuple(type(member) for member in kernel_specs)
        prefix = (RadialBasisSpec, SphericalHarmonicsSpec)
        if kinds[:2] != prefix:
            return None
        offset = 2
        transform = None
        if kinds[offset : offset + 1] == (DistanceTransformSpec,):
            transform = cast(DistanceTransformSpec, kernel_specs[offset])
            offset += 1
        suffix = kinds[offset:]
        if suffix not in (
            (RadialMLPSpec, ChannelwiseTPConvSpec),
            (RadialMLPSpec, RadialMLPSpec, ChannelwiseTPConvSpec),
        ):
            return None
        try:
            return cls(
                radial_basis=cast(RadialBasisSpec, kernel_specs[0]),
                spherical_harmonics=cast(SphericalHarmonicsSpec, kernel_specs[1]),
                radial_network=cast(RadialMLPSpec, kernel_specs[offset]),
                convolution=cast(ChannelwiseTPConvSpec, kernel_specs[-1]),
                distance_transform=transform,
                density=cast(RadialMLPSpec, kernel_specs[offset + 1])
                if len(suffix) == 3
                else None,
            )
        except ValueError:
            # Two networks in the tuple need not have this shape's semantics.
            return None
