"""What an op is, before any backend has agreed to build it.

A descriptor is the complete statement of one operation's shape: enough for a
backend to answer whether it can build it, and enough to build it. It carries no
weights and no tensors, only the numbers and the irreps strings, so it is
hashable and can be compared, cached and written into a checkpoint.

Descriptors are frozen. A backend is handed one at model build time, returns an
op, and the descriptor is never consulted again: nothing resolves in ``forward``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.precision import Precision

__all__ = [
    "ChannelwiseTPConvDescriptor",
    "Descriptor",
    "FullyConnectedTPDescriptor",
    "LinearDescriptor",
    "RadialBasisDescriptor",
    "SegmentReduceDescriptor",
    "SphericalHarmonicsDescriptor",
    "SymmetricContractionDescriptor",
]


@dataclass(frozen=True)
class Descriptor:
    """What every op descriptor carries.

    Attributes:
        precision: The dtype the op computes in, by name.
    """

    precision: Precision = "float64"

    @property
    def weight_numel(self) -> int:
        """How many trainable scalars the op owns.

        Zero for the ops whose weights come from outside, and for the ones that
        have none. A backend uses it to size its own storage, and the checkpoint
        format uses it to know how much to read.
        """
        return 0


@dataclass(frozen=True)
class LinearDescriptor(Descriptor):
    """An equivariant linear map between two irreps declarations.

    Attributes:
        irreps_in: The input declaration.
        irreps_out: The output declaration.
        has_bias: Whether the map carries a bias. First-class rather than a
            wrapper, because a scalar bias on the readout is part of published
            model architectures and a backend that silently drops it is wrong
            by a constant.
    """

    irreps_in: str = "0e"
    irreps_out: str = "0e"
    has_bias: bool = False

    @property
    def weight_numel(self) -> int:
        """One weight per matching (input, output) multiplicity pair.

        An equivariant linear map connects a term only to a term of the same
        irrep, so the count is the sum over matching irreps of the product of
        multiplicities. The bias adds one scalar per scalar output channel: only
        ``0e`` may carry a bias without breaking equivariance.
        """
        source = Irreps.parse(self.irreps_in)
        target = Irreps.parse(self.irreps_out)
        total = 0
        for out_mul, out_ir in target:
            for in_mul, in_ir in source:
                if in_ir == out_ir:
                    total += in_mul * out_mul
        if self.has_bias:
            total += sum(mul for mul, ir in target if ir.degree == 0 and ir.parity == 1)
        return total


@dataclass(frozen=True)
class ChannelwiseTPConvDescriptor(Descriptor):
    """The message-passing tensor product, channel by channel, over the edges.

    Attributes:
        irreps_node: The sender node features.
        irreps_edge: The edge attributes, normally spherical harmonics.
        irreps_out: The message irreps before the node-level reduction.
        num_radial: Width of the radial embedding whose MLP supplies the
            weights.

    The op is **always node-level**: it returns ``[n_nodes, ...]``, never
    ``[n_edges, ...]``. Whether the reduction over edges is fused into the
    kernel is the backend's business and not the model's, which is what removes
    the six ``hasattr(self, "conv_fusion")`` branches the frozen tree carries
    through its interaction blocks.
    """

    irreps_node: str = "0e"
    irreps_edge: str = "0e"
    irreps_out: str = "0e"
    num_radial: int = 8

    @property
    def weight_numel(self) -> int:
        """Zero. The weights come from the radial MLP, which is external."""
        return 0


@dataclass(frozen=True)
class SymmetricContractionDescriptor(Descriptor):
    """The many-body contraction over the reduced Clebsch-Gordan basis.

    Attributes:
        irreps_in: The node features being contracted.
        irreps_out: The output irreps to keep.
        correlation: The body order.
        num_elements: How many chemical elements carry their own weights.
        num_features: The channel width.
        basis: Which Clebsch-Gordan basis the weights are written against.
            ``"reduced"`` is what v1 persists. It is a recorded field rather
            than something inferred from what happens to be installed, which is
            the defect this contract exists to remove.
    """

    irreps_in: str = "0e"
    irreps_out: str = "0e"
    correlation: int = 1
    num_elements: int = 1
    num_features: int = 1
    basis: Literal["reduced", "full"] = "reduced"

    @property
    def path_count(self) -> int:
        """How many basis paths the weights multiply, summed over body orders.

        Computed from first principles by :mod:`mace_core.clebsch_gordan`, with
        no dependence on what is installed.
        """
        from mace_core.clebsch_gordan.reduced_basis import (
            full_symmetric_tensor_product_basis,
            reduced_symmetric_tensor_product_basis,
        )

        build = (
            reduced_symmetric_tensor_product_basis
            if self.basis == "reduced"
            else full_symmetric_tensor_product_basis
        )
        total = 0
        for order in range(1, self.correlation + 1):
            for array in build(self.irreps_in, order, self.irreps_out).values():
                total += int(array.shape[0])
        return total

    @property
    def weight_numel(self) -> int:
        """The canonical flat ``[Z, A, mul]`` count: elements, paths, channels."""
        return self.num_elements * self.path_count * self.num_features


@dataclass(frozen=True)
class FullyConnectedTPDescriptor(Descriptor):
    """The skip connection's tensor product against the element attributes.

    Attributes:
        irreps_in1: The node features.
        irreps_in2: The element attributes, which are scalars.
        irreps_out: What it produces.
    """

    irreps_in1: str = "0e"
    irreps_in2: str = "0e"
    irreps_out: str = "0e"

    @property
    def weight_numel(self) -> int:
        """Derived from the irreps, like every other descriptor's.

        The second input is scalars, so the product is one equivariant linear
        map per attribute: the matching multiplicity pairs between the first
        input and the output, times how many attributes there are. Taking it
        from the caller instead made it a number nobody checked, and a
        capability filter or a checkpoint sized against it would have been
        wrong by whatever the caller happened to pass.
        """
        source = Irreps.parse(self.irreps_in1)
        target = Irreps.parse(self.irreps_out)
        pairs = sum(
            in_mul * out_mul
            for out_mul, out_ir in target
            for in_mul, in_ir in source
            if in_ir == out_ir
        )
        return pairs * Irreps.parse(self.irreps_in2).dimension


@dataclass(frozen=True)
class SegmentReduceDescriptor(Descriptor):
    """A reduction of values into segments. No irreps semantics.

    Attributes:
        reduction: Which reduction. ``"sum"`` is what message passing and the
            energy total both need; the others exist because the same op serves
            the pair-repulsion and readout reductions.
        num_features: The width of each value.
    """

    reduction: Literal["sum", "mean", "max"] = "sum"
    num_features: int = 1


@dataclass(frozen=True)
class SphericalHarmonicsDescriptor(Descriptor):
    """Real spherical harmonics of the edge directions.

    Reference-only: a backend may decline it and the reference builds it, since
    it is a cheap closed form. The convention is the one
    :mod:`mace_core.clebsch_gordan.real_basis` states, and a backend that
    supplies its own must produce that convention rather than its own.
    """

    lmax: int = 0
    normalize: bool = True


@dataclass(frozen=True)
class RadialBasisDescriptor(Descriptor):
    """The radial embedding of the edge lengths.

    Attributes:
        kind: Which basis. Lowercase names, matching the configuration.
        num_basis: How many functions.
        cutoff: The cutoff radius, in Angstrom.
        extra: Basis-specific parameters, as a sorted tuple of pairs so the
            descriptor stays hashable.
    """

    kind: Literal["bessel", "gaussian", "chebyshev"] = "bessel"
    num_basis: int = 8
    cutoff: float = 5.0
    extra: tuple[tuple[str, float], ...] = field(default_factory=tuple)
