"""What an op is, before any backend has agreed to build it.

A descriptor is the complete statement of one operation's shape: enough for a
backend to answer whether it can build it, and enough to build it. It carries no
weights and no tensors, only the numbers and the irreps strings, so it is
hashable and can be compared, cached and written into a checkpoint.

Descriptors are frozen. A backend is handed one at model build time, returns an
op, and the descriptor is never consulted again: nothing resolves in ``forward``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, get_args

from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.canonical import CANONICAL_LAYOUT
from mace_core.kernels.paths import ConvolutionPath, channelwise_paths, path_irreps
from mace_core.kernels.precision import PRECISIONS, Precision

__all__ = [
    "CLEBSCH_GORDAN_BASES",
    "ChannelwiseTPConvDescriptor",
    "Descriptor",
    "FullyConnectedTPDescriptor",
    "LinearDescriptor",
    "RadialBasisDescriptor",
    "RadialMLPDescriptor",
    "SegmentReduceDescriptor",
    "SphericalHarmonicsDescriptor",
    "SymmetricContractionDescriptor",
]


ClebschGordanBasis = Literal["reduced", "full"]

#: The Clebsch-Gordan bases a symmetric contraction can be written against.
CLEBSCH_GORDAN_BASES: tuple[str, ...] = get_args(ClebschGordanBasis)


def _require_member(field_name: str, value: object, allowed: tuple) -> None:
    """Raise unless ``value`` is one of ``allowed``.

    A ``Literal`` annotation is a promise to the type checker only. A value
    read from a configuration file or a checkpoint reaches the constructor
    unchecked, and a typo would otherwise select whatever the code does in its
    fallback branch.
    """
    if value not in allowed:
        raise ValueError(
            f"{field_name}={value!r} is not one of {list(allowed)}. Spell it "
            f"exactly as listed; nothing is inferred from a near miss."
        )


def _highest_degree(*declarations: str) -> int:
    """The largest ``l`` any of the irreps declarations carries."""
    return max(
        (ir.degree for text in declarations for _, ir in Irreps.parse(text)),
        default=0,
    )


@dataclass(frozen=True)
class Descriptor:
    """What every op descriptor carries.

    Attributes:
        precision: The dtype the op computes in, by name.
        op: The factory this descriptor is for, as
            :attr:`~mace_core.kernels.capabilities.BackendCapabilities.ops`
            lists it. A class constant, not a field.
        layout: The layout of the features and weights the op reads and
            writes, or ``None`` for an op with no irreps semantics. A class
            constant, not a field: every op with irreps works in the canonical
            ``mul_ir``, since that is what a checkpoint holds.
    """

    op: ClassVar[str] = ""
    layout: ClassVar[str | None] = None

    precision: Precision = "float64"

    def __post_init__(self) -> None:
        _require_member("precision", self.precision, PRECISIONS)

    @property
    def highest_degree(self) -> int:
        """The largest rotation order ``l`` the op's declared irreps carry.

        What :attr:`~mace_core.kernels.capabilities.BackendCapabilities.max_lmax`
        is checked against. Zero for an op with no irreps.
        """
        return 0

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

    op: ClassVar[str] = "linear"
    layout: ClassVar[str | None] = CANONICAL_LAYOUT

    irreps_in: str = "0e"
    irreps_out: str = "0e"
    has_bias: bool = False

    @property
    def highest_degree(self) -> int:
        return _highest_degree(self.irreps_in, self.irreps_out)

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
        irreps_node: One channel's sender node features.
        irreps_edge: The edge attributes, normally spherical harmonics.
        irreps_out: Which output irreps the layer keeps. A coupling landing
            outside it is dropped.
        num_features: The channel count.

    **Every coupling is its own path, with its own output term**, in the order
    :func:`~mace_core.kernels.paths.channelwise_paths` pins, and
    :attr:`irreps_paths` is one channel's message declaration. Two couplings
    landing on one irrep are not summed: the linear after the convolution mixes
    them with independent weights, and summing first is a smaller model.

    The features are grouped by irrep, the canonical ``mul_ir`` over the
    expanded declaration: ``[n_nodes, num_features * dim_node]`` in, and
    ``[n_nodes, num_features * dim_paths]`` out, one block of ``num_features``
    channels per path. Copy ``k`` of a term with multiplicity ``m`` in channel
    ``c`` is copy ``c * m + k`` of the expanded term. How a backend arranges
    the values inside its kernel is its own business, and never a permute in
    the model.

    The weights are per edge, ``[n_edges, n_paths, num_features]``: the output
    of a radial MLP (:class:`RadialMLPDescriptor`) read path by path, with the
    channels varying fastest. A backend that fuses the two does so through the
    span factory, which is handed both descriptors.

    The op is **always node-level**: it returns ``[n_nodes, ...]``, never
    ``[n_edges, ...]``. Whether the reduction over edges is fused into the
    kernel is the backend's business and not the model's, which is what removes
    the six ``hasattr(self, "conv_fusion")`` branches the frozen tree carries
    through its interaction blocks.
    """

    op: ClassVar[str] = "channelwise_tp_conv"
    layout: ClassVar[str | None] = CANONICAL_LAYOUT

    irreps_node: str = "0e"
    irreps_edge: str = "0e"
    irreps_out: str = "0e"
    num_features: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_features < 1:
            raise ValueError(
                f"num_features={self.num_features} is not a channel count; it "
                f"is at least one."
            )
        for multiplicity, irrep in Irreps.parse(self.irreps_node):
            if multiplicity != 1:
                raise ValueError(
                    f"irreps_node={self.irreps_node!r} gives {irrep} a "
                    f"multiplicity of {multiplicity}. The radial weights are "
                    f"one per path and channel, and the channels are "
                    f"num_features, so write each irrep once and set "
                    f"num_features for the width."
                )

    @property
    def highest_degree(self) -> int:
        return _highest_degree(self.irreps_node, self.irreps_edge, self.irreps_out)

    @property
    def paths(self) -> tuple[ConvolutionPath, ...]:
        """Every coupling kept, in the order the radial weights are read."""
        return channelwise_paths(self.irreps_node, self.irreps_edge, self.irreps_out)

    @property
    def irreps_paths(self) -> str:
        """One channel's message: one term per path, repeated irreps apart."""
        return path_irreps(self.paths)

    @property
    def weight_numel(self) -> int:
        """Zero. The weights come from the radial MLP, which is its own op."""
        return 0


@dataclass(frozen=True)
class SymmetricContractionDescriptor(Descriptor):
    """The many-body contraction over the reduced Clebsch-Gordan basis.

    Attributes:
        irreps_in: One channel's node features being contracted.
        irreps_out: The output irreps to keep, one copy of each per channel.
            A multiplicity other than one is refused: the channels are the
            ``num_features`` axis, not a multiplicity in the irreps.
        correlation: The body order.
        num_elements: How many chemical elements carry their own weights.
        num_features: The channel width.
        basis: Which Clebsch-Gordan basis the weights are written against.
            ``"reduced"`` is what v1 persists. It is a recorded field rather
            than something inferred from what happens to be installed, which is
            the defect this contract exists to remove.

    The features are grouped by irrep, as the convolution's are:
    ``[n_nodes, num_features * dim_in]`` in and
    ``[n_nodes, num_features * dim_out]`` out, each the canonical ``mul_ir``
    over the declaration expanded to ``num_features`` channels.
    """

    op: ClassVar[str] = "symmetric_contraction"
    layout: ClassVar[str | None] = CANONICAL_LAYOUT

    irreps_in: str = "0e"
    irreps_out: str = "0e"
    correlation: int = 1
    num_elements: int = 1
    num_features: int = 1
    basis: ClebschGordanBasis = "reduced"

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_member("basis", self.basis, CLEBSCH_GORDAN_BASES)
        for multiplicity, irrep in Irreps.parse(self.irreps_out):
            if multiplicity != 1:
                raise ValueError(
                    f"irreps_out={self.irreps_out!r} gives {irrep} a "
                    f"multiplicity of {multiplicity}. The contraction's irreps "
                    f"are per channel and the channels are num_features, so "
                    f"write the output irrep once, as {irrep}, and set "
                    f"num_features for the width."
                )

    @property
    def highest_degree(self) -> int:
        return _highest_degree(self.irreps_in, self.irreps_out)

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

        build = {
            "reduced": reduced_symmetric_tensor_product_basis,
            "full": full_symmetric_tensor_product_basis,
        }[self.basis]
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

    op: ClassVar[str] = "fully_connected_tp"
    layout: ClassVar[str | None] = CANONICAL_LAYOUT

    irreps_in1: str = "0e"
    irreps_in2: str = "0e"
    irreps_out: str = "0e"

    @property
    def highest_degree(self) -> int:
        return _highest_degree(self.irreps_in1, self.irreps_in2, self.irreps_out)

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


Reduction = Literal["sum", "mean", "max"]


@dataclass(frozen=True)
class SegmentReduceDescriptor(Descriptor):
    """A reduction of values into segments. No irreps semantics.

    Attributes:
        reduction: Which reduction. ``"sum"`` is what message passing and the
            energy total both need; the others exist because the same op serves
            the pair-repulsion and readout reductions.
        num_features: The width of each value.
    """

    op: ClassVar[str] = "segment_reduce"

    reduction: Reduction = "sum"
    num_features: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_member("reduction", self.reduction, get_args(Reduction))


@dataclass(frozen=True)
class SphericalHarmonicsDescriptor(Descriptor):
    """Real spherical harmonics of the edge directions.

    Reference-only: a backend may decline it and the reference builds it, since
    it is a cheap closed form. The convention is e3nn's with
    ``normalization="component"``, the one legacy models were trained in: y is
    the polar axis, so the ``l = 1`` block is ``sqrt(3) (x, y, z)``, and each
    ``l`` block has squared norm ``2l + 1`` on the unit sphere. A backend that
    supplies its own must produce this convention rather than its own.

    These are the harmonics of :mod:`mace_core.clebsch_gordan.real_basis`
    evaluated at the permuted direction ``(z, x, y)``, scaled by
    ``sqrt(2l + 1)``. The permutation is cyclic and so a proper rotation, which
    is why the Clebsch-Gordan tables built in that basis couple these harmonics
    equivariantly without any change.
    """

    op: ClassVar[str] = "spherical_harmonics"

    lmax: int = 0
    normalize: bool = True

    @property
    def highest_degree(self) -> int:
        return self.lmax


RadialBasisKind = Literal["bessel", "gaussian", "chebyshev"]


@dataclass(frozen=True)
class RadialBasisDescriptor(Descriptor):
    """The radial embedding of the edge lengths.

    Attributes:
        kind: Which basis. Lowercase names, matching the configuration.
        num_basis: How many functions.
        cutoff: The cutoff radius, in Angstrom.
        cutoff_order: The order ``p`` of the polynomial cutoff envelope. The
            default is the envelope's own module default, not a model's: the
            legacy command line defaults ``--num_cutoff_basis`` to 5, so a
            model always passes the order it was configured with.
    """

    op: ClassVar[str] = "radial_basis"

    kind: RadialBasisKind = "bessel"
    num_basis: int = 8
    cutoff: float = 5.0
    cutoff_order: int = 6

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_member("kind", self.kind, get_args(RadialBasisKind))


RadialMLPKind = Literal["normalized_silu", "layer_norm_silu"]


@dataclass(frozen=True)
class RadialMLPDescriptor(Descriptor):
    """The dense network that turns a radial embedding into per-edge weights.

    One per interaction, feeding the convolution its path weights, and the same
    op with one output produces the learned density of the density blocks.

    Attributes:
        channels: The layer widths, input first and output last. The input is
            the radial embedding's width. No activation follows the last layer,
            so the output is unbounded.
        kind: Which of the two networks the frozen tree trains.
            ``"normalized_silu"`` is ``e3nn``'s ``FullyConnectedNet``, used by
            every interaction block but one: unbiased layers scaled by
            ``1 / sqrt(fan_in)``, and a SiLU scaled to unit second moment
            between them. ``"layer_norm_silu"`` is the nonlinear residual
            block's: biased layers with a layer norm and a SiLU between them.
            :mod:`~mace_core.kernels.canonical` states both exactly.
    """

    op: ClassVar[str] = "radial_mlp"

    channels: tuple[int, ...] = (8, 64, 64, 64)
    kind: RadialMLPKind = "normalized_silu"

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_member("kind", self.kind, get_args(RadialMLPKind))
        # A list from a configuration file would make the descriptor
        # unhashable, and a descriptor is a cache key.
        object.__setattr__(self, "channels", tuple(self.channels))
        if len(self.channels) < 2:
            raise ValueError(
                f"channels={self.channels!r} needs at least an input and an "
                f"output width."
            )
        if any(not isinstance(width, int) or width < 1 for width in self.channels):
            raise ValueError(
                f"channels={self.channels!r} has a width that is not a positive "
                f"integer."
            )

    @property
    def weight_numel(self) -> int:
        """Every matrix entry, plus the biases and the layer norms that the
        ``"layer_norm_silu"`` kind carries."""
        layers = list(zip(self.channels[:-1], self.channels[1:], strict=True))
        total = sum(fan_in * fan_out for fan_in, fan_out in layers)
        if self.kind == "layer_norm_silu":
            total += sum(fan_out for _, fan_out in layers)
            total += 2 * sum(self.channels[1:-1])
        return total
