"""The reference backend: plain torch, no e3nn, and the correctness oracle.

Mandatory and CPU-capable. Every other backend is checked against this one, so
it is written for clarity over speed: the symmetric contraction expands the
outer power rather than fusing anything, and the linear map builds a dense
matrix from its flat weights on every call. Both are wasteful and both are
obviously right, which is the trade a reference is for.

It holds the canonical weight layout directly, so ``to_canonical`` and
``load_canonical`` are views rather than conversions. That is the property that
makes one checkpoint loadable by any backend: the reference defines the format
by holding it.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import cast

import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.real_basis import wigner_3j_real
from mace_core.clebsch_gordan.reduced_basis import (
    full_symmetric_tensor_product_basis,
    reduced_symmetric_tensor_product_basis,
)
from mace_core.kernels.canonical import (
    LAYER_NORM_EPSILON,
    SILU_SECOND_MOMENT_SCALE,
    fully_connected_tp_path_normalization,
    linear_path_normalization,
    radial_mlp_initial_bound,
    radial_mlp_layer_normalization,
    symmetric_contraction_initial_scale,
)
from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    Descriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    RadialBasisDescriptor,
    RadialMLPDescriptor,
    SegmentReduceDescriptor,
    SphericalHarmonicsDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.protocol import DISPATCHED_OPS, REFERENCE_ONLY_OPS
from torch import Tensor, nn

from mace_torch.backends.harmonics import spherical_harmonics
from mace_torch.backends.layout import grouped, per_channel, term_widths
from mace_torch.backends.radial import (
    BesselBasis,
    ChebyshevBasis,
    GaussianBasis,
    PolynomialCutoff,
)
from mace_torch.kernels.ops import (
    channelwise_tp_conv,
    equivariant_linear,
    segment_sum,
    symmetric_contraction,
)

__all__ = [
    "TORCH_DTYPES",
    "ReferenceBackend",
    "ReferenceCapabilities",
    "ReferenceLayerNormSiluMLP",
    "ReferenceNormalizedSiluMLP",
]

#: The torch dtype of each precision name this backend computes in.
TORCH_DTYPES = {"float64": torch.float64, "float32": torch.float32}

#: The two bases, by the name a descriptor records. A lookup rather than a
#: conditional, so a name outside the set raises instead of selecting the
#: fallback branch.
_BASIS_BUILDERS = {
    "reduced": reduced_symmetric_tensor_product_basis,
    "full": full_symmetric_tensor_product_basis,
}


def _linear_plan(descriptor: LinearDescriptor):
    """Which weight writes to which entry of the dense matrix.

    One weight per matching multiplicity pair, repeated over the ``2l+1``
    components of its irrep. That repetition is the equivariance: a free matrix
    would mix components of one irrep into another.
    """
    source_irreps = Irreps.parse(descriptor.irreps_in)
    target_irreps = Irreps.parse(descriptor.irreps_out)
    rows, columns, sources = [], [], []
    weight = 0
    for out_slice, out_ir in target_irreps.slices():
        for in_slice, in_ir in source_irreps.slices():
            if in_ir != out_ir:
                continue
            for component in range(in_ir.dimension):
                rows.append(out_slice.start + component)
                columns.append(in_slice.start + component)
                sources.append(weight)
            weight += 1
    bias_rows = []
    if descriptor.has_bias:
        for out_slice, out_ir in target_irreps.slices():
            if out_ir.degree == 0 and out_ir.parity == 1:
                bias_rows.append(out_slice.start)
    return rows, columns, sources, weight, bias_rows


def _linear_path_normalizations(irreps_in: str, irreps_out: str) -> list[float]:
    """The forward factor of every weight of a linear plan, in plan order.

    One entry per weight, so the forward applies it as one multiplication. The
    order is `_linear_plan`'s: output copies outermost, and within one output
    copy the matching input copies in declaration order.
    """
    source = Irreps.parse(irreps_in)
    target = Irreps.parse(irreps_out)
    scales: list[float] = []
    for out_multiplicity, out_irrep in target:
        scale = linear_path_normalization(irreps_in, out_irrep)
        for _ in range(out_multiplicity):
            for in_multiplicity, in_irrep in source:
                if in_irrep != out_irrep:
                    continue
                scales.extend([scale] * in_multiplicity)
    return scales


def _skip_path_normalizations(descriptor: FullyConnectedTPDescriptor) -> list[float]:
    """The same, for the skip connection, whose fan-in counts both inputs.

    One factor per output irrep, shared by every path landing on it, which is
    how ``e3nn`` counts the fan-in.
    """
    source = Irreps.parse(descriptor.irreps_in1)
    target = Irreps.parse(descriptor.irreps_out)
    scales: list[float] = []
    for out_multiplicity, out_irrep in target:
        scale = fully_connected_tp_path_normalization(
            descriptor.irreps_in1, descriptor.irreps_in2, out_irrep
        )
        for _ in range(out_multiplicity):
            for in_multiplicity, in_irrep in source:
                if in_irrep != out_irrep:
                    continue
                scales.extend([scale] * in_multiplicity)
    return scales


def _draw(like: Tensor, seed: int) -> Tensor:
    """A standard normal shaped, typed and placed like ``like``.

    Its own generator rather than the global one: a model's weights must not
    depend on how many random numbers anything else drew first, which is what
    makes a run reproducible from its recorded seed.

    Drawn on the host and moved, never drawn on the device. A device generator
    seeded the same way produces different numbers, so drawing where the model
    happens to live would make a recorded seed rebuild a different model on a
    different machine. It is moved rather than left behind because a model that
    has been sent to a device has its buffers there, and multiplying a host
    tensor by one of them is a device mismatch.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(like.shape, generator=generator, dtype=torch.float64).to(
        device=like.device, dtype=like.dtype
    )


class ReferenceLinear(nn.Module):
    """An equivariant linear map with first-class bias.

    The weight is canonical and raw. The forward multiplies each weight by its
    ``1 / sqrt(fan_in)``, from
    :func:`~mace_core.kernels.canonical.linear_path_normalization`, before
    building the map, as ``e3nn`` does, so the gradient with respect to the
    stored weight is the frozen tree's. The bias is added unscaled.
    """

    row: Tensor
    column: Tensor
    source: Tensor
    bias_row: Tensor
    path_normalization: Tensor

    def __init__(self, descriptor: LinearDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        rows, columns, sources, count, bias_rows = _linear_plan(descriptor)
        dtype = TORCH_DTYPES[descriptor.precision]
        self.dim_out = Irreps.parse(descriptor.irreps_out).dimension
        self.register_buffer("row", torch.tensor(rows, dtype=torch.long))
        self.register_buffer("column", torch.tensor(columns, dtype=torch.long))
        self.register_buffer("source", torch.tensor(sources, dtype=torch.long))
        self.register_buffer("bias_row", torch.tensor(bias_rows, dtype=torch.long))
        self.weight = nn.Parameter(torch.zeros(count, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(len(bias_rows), dtype=dtype))
        self.register_buffer(
            "path_normalization",
            torch.tensor(
                _linear_path_normalizations(
                    descriptor.irreps_in, descriptor.irreps_out
                ),
                dtype=dtype,
            ),
            persistent=False,
        )

    def forward(self, features: Tensor) -> Tensor:
        return equivariant_linear(
            features,
            self.weight * self.path_normalization,
            self.row,
            self.column,
            self.source,
            self.bias,
            self.bias_row,
            self.dim_out,
        )

    def initialize_weights(self, seed: int) -> None:
        """A standard normal, unscaled, since the forward applies the fan-in.
        The bias starts at zero.

        A bias is an offset on the scalar outputs, and starting it anywhere
        other than zero would shift the model's energy before it has seen a
        structure. The frozen tree's linear does the same.
        """
        with torch.no_grad():
            self.weight.copy_(_draw(self.weight, seed))
            self.bias.zero_()

    def to_canonical(self) -> dict[str, Tensor]:
        """A view. The reference holds the canonical layout already."""
        return {"weight": self.weight.detach(), "bias": self.bias.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])
            self.bias.copy_(state["bias"])


class _ConstantTensors(nn.Module):
    """A device-following list of constant tables.

    Buffers move with the module and parameters do not fit, since these carry
    no gradient. A list of plain tensors would go stale the first time the
    model is moved to a device, so they are held as buffers and iterated in
    the order they were given.
    """

    def __init__(self, tensors: list[Tensor]) -> None:
        super().__init__()
        self.count = len(tensors)
        for position, tensor in enumerate(tensors):
            self.register_buffer(f"table_{position}", tensor, persistent=False)

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Tensor]:
        return iter(getattr(self, f"table_{p}") for p in range(self.count))


class ReferenceSymmetricContraction(nn.Module):
    """The many-body contraction, over the basis the descriptor records.

    One contraction per output irrep, concatenated on the component axis. They
    cannot share a stacked basis: each output irrep has its own component count,
    so stacking them would be joining arrays whose second axis differs. The
    frozen tree reaches the same shape by holding one `Contraction` per output
    irrep, and this is that, with the loop kept explicit.
    """

    def __init__(self, descriptor: SymmetricContractionDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        dtype = TORCH_DTYPES[descriptor.precision]
        if descriptor.basis not in _BASIS_BUILDERS:
            raise ValueError(
                f"basis={descriptor.basis!r} is not a Clebsch-Gordan basis this "
                f"backend builds. The bases are {sorted(_BASIS_BUILDERS)}."
            )
        build = _BASIS_BUILDERS[descriptor.basis]
        self.orders = descriptor.correlation
        self.targets = [str(ir) for _, ir in Irreps.parse(descriptor.irreps_out)]
        self.num_features = descriptor.num_features
        self.in_widths = term_widths(descriptor.irreps_in)
        self.out_widths = term_widths(descriptor.irreps_out)
        weights, bases = [], []
        for target in self.targets:
            group, tables = [], []
            for order in range(1, descriptor.correlation + 1):
                array = build(descriptor.irreps_in, order, target)[target]
                # The trailing extent is computed rather than inferred with
                # `-1`: an output irrep no path of this body order reaches has
                # zero paths, and numpy cannot infer a dimension of an empty
                # array. A `2e` output is exactly that at body order one.
                trailing = int(np.prod(array.shape[2:])) if array.ndim > 2 else 1
                flat = array.reshape(array.shape[0], array.shape[1], trailing)
                tables.append(torch.tensor(flat, dtype=dtype))
                group.append(
                    nn.Parameter(
                        torch.zeros(
                            descriptor.num_elements,
                            flat.shape[0],
                            descriptor.num_features,
                            dtype=dtype,
                        )
                    )
                )
            weights.extend(group)
            bases.append(_ConstantTensors(tables))
        self.weights = nn.ParameterList(weights)
        self.bases = nn.ModuleList(bases)

    def initialize_weights(self, seed: int) -> None:
        """A standard normal over the block's path count, the frozen tree's draw.

        One normal per element, path and channel, divided by the number of
        paths in its ``(output irrep, body order)`` block, as
        :func:`~mace_core.kernels.canonical.symmetric_contraction_initial_scale`
        states. The forward applies no factor. Each tensor of the flat list
        gets its own offset, or the body orders of one output irrep would
        start equal.
        """
        with torch.no_grad():
            for position, parameter in enumerate(self.weights):
                scale = symmetric_contraction_initial_scale(int(parameter.shape[1]))
                parameter.copy_(_draw(parameter, seed + position) * scale)

    def _group(self, position: int) -> list[Tensor]:
        """One output irrep's weights, by integer index.

        Flat storage with integer indexing rather than a slice of the
        `ParameterList`: slicing one goes through `slice.indices`, a C builtin
        that `torch.compile` cannot trace, and the break lands in the middle of
        the backbone rather than here.
        """
        base = position * self.orders
        return [self.weights[base + order] for order in range(self.orders)]

    def forward(self, features: Tensor, element: Tensor) -> Tensor:
        """Grouped features in and out.

        Args:
            features: ``[n_nodes, num_features * dim_in]``, grouped.
            element: ``[n_nodes]``, int64, which element each node is.

        Returns:
            ``[n_nodes, num_features * dim_out]``, grouped.
        """
        channels = per_channel(features, self.in_widths, self.num_features)
        # `nn.ModuleList` erases what it holds, so the element type has to be
        # said here. It is the one thing put into `self.bases`, two lines of
        # the constructor away.
        bases = cast("list[_ConstantTensors]", list(self.bases))
        pieces = [
            symmetric_contraction(
                channels, self._group(position), list(tables), element
            )
            for position, tables in enumerate(bases)
        ]
        return grouped(torch.cat(pieces, dim=-1), self.out_widths)

    def to_canonical(self) -> dict[str, Tensor]:
        """The flat ``[Z, A, mul]`` array, in the pinned path order.

        **Body order outermost, output irrep within it**, which is the order
        the path enumeration walks and therefore the order on disk. The weights
        are stored the other way round, output irrep outermost, because that is
        how the forward consumes them, so this reorders rather than simply
        concatenating. Emitting the storage order instead would write a file
        whose paths are the canonical ones permuted, and the round trip through
        this same object would not notice: it splits by the counts it wrote.
        """
        ordered = self._canonical_order()
        return {
            "weight": torch.cat([w.detach() for w in ordered], dim=1),
            "path_counts": torch.tensor([w.shape[1] for w in ordered]),
        }

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        counts = [int(n) for n in state["path_counts"]]
        pieces = torch.split(state["weight"], counts, dim=1)
        with torch.no_grad():
            for parameter, piece in zip(self._canonical_order(), pieces, strict=True):
                parameter.copy_(piece)

    def _canonical_order(self) -> list[Tensor]:
        """The weight tensors in the order the file has them.

        Body order ascending, and within one body order the output irreps in
        declaration order. `self.weights` is the transpose of this.
        """
        return [
            self.weights[position * self.orders + order]
            for order in range(self.orders)
            for position in range(len(self.targets))
        ]


def _term_offsets(irreps: Irreps) -> list[int]:
    """Where each term of a declaration starts in its flat vector."""
    offsets, start = [], 0
    for multiplicity, irrep in irreps.terms:
        offsets.append(start)
        start += multiplicity * irrep.dimension
    return offsets


def _coupling_coefficients(descriptor: ChannelwiseTPConvDescriptor) -> np.ndarray:
    """The Clebsch-Gordan coefficients of the message-passing product.

    One ``[dim_paths, dim_node, dim_edge]`` block per path, over one channel,
    stacked into ``[n_paths, dim_paths, dim_node, dim_edge]``. Each path is
    nonzero only in its own output term, so the kernel's sum over paths weights
    every coupling separately and writes it where the linear after it expects
    it. Dense because this is the reference; a backend with a real kernel
    keeps them sparse.

    Each path carries the factor ``sqrt(2 l_out + 1)`` on its unit-norm
    Wigner 3j symbol. That is the per-path weight of the frozen tree's
    ``e3nn`` tensor product, built with ``"component"`` irrep normalization,
    ``"element"`` path normalization and one ``uvu`` instruction per path:
    the fan-in of each instruction is the edge multiplicity, one, so the path
    weight is ``sqrt(2 l_out + 1)`` exactly.
    """
    node = Irreps.parse(descriptor.irreps_node)
    edge = Irreps.parse(descriptor.irreps_edge)
    paths = descriptor.paths
    dim_paths = sum(path.irrep.dimension for path in paths)
    node_offsets = _term_offsets(node)
    edge_offsets = _term_offsets(edge)
    coefficients = np.zeros((len(paths), dim_paths, node.dimension, edge.dimension))
    out_start = 0
    for index, path in enumerate(paths):
        node_irrep = node.terms[path.node_term][1]
        edge_irrep = edge.terms[path.edge_term][1]
        node_start = node_offsets[path.node_term]
        edge_start = edge_offsets[path.edge_term]
        coefficients[
            index,
            out_start : out_start + path.irrep.dimension,
            node_start : node_start + node_irrep.dimension,
            edge_start : edge_start + edge_irrep.dimension,
        ] = math.sqrt(path.irrep.dimension) * wigner_3j_real(
            path.irrep.degree, node_irrep.degree, edge_irrep.degree
        )
        out_start += path.irrep.dimension
    return coefficients


class ReferenceChannelwiseTPConv(nn.Module):
    """The message-passing tensor product. Node-level, always.

    Grouped features in and out, as every op of the chain reads them. The
    kernel computes channel by channel, so the features move to that
    arrangement and back here, at the backend boundary.
    """

    #: Annotated because `register_buffer` alone leaves it typed as a `Module`,
    #: and then reading its shape reads as subscripting a module.
    coefficients: Tensor

    def __init__(self, descriptor: ChannelwiseTPConvDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        dtype = TORCH_DTYPES[descriptor.precision]
        coefficients = _coupling_coefficients(descriptor)
        self.register_buffer(
            "coefficients", torch.tensor(coefficients, dtype=dtype), persistent=False
        )
        self.num_paths = len(descriptor.paths)
        self.num_features = descriptor.num_features
        self.node_widths = term_widths(descriptor.irreps_node)
        self.path_widths = term_widths(descriptor.irreps_paths)

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        radial_weights: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Grouped node features in, one grouped block per path out.

        Args:
            node_features: ``[n_nodes, num_features * dim_node]``, grouped.
            edge_attributes: ``[n_edges, dim_edge]``.
            radial_weights: ``[n_edges, n_paths, num_features]``.
            sender: ``[n_edges]``, int64.
            receiver: ``[n_edges]``, int64.
            num_nodes: The node count, as a plain int.
        """
        channels = per_channel(node_features, self.node_widths, self.num_features)
        message = channelwise_tp_conv(
            channels,
            edge_attributes,
            radial_weights,
            self.coefficients,
            sender,
            receiver,
            num_nodes,
        )
        return grouped(message, self.path_widths)


class ReferenceFullyConnectedTP(nn.Module):
    """The skip connection's tensor product against the element attributes.

    Only the case the models use is built: the second input is scalars, the
    element one-hots, so the product is a per-element linear map. The general
    case raises rather than returning something plausible, because a wrong skip
    connection is a wrong model that still trains.

    The weight is canonical and raw, ``[num_scalars, weights per scalar]``, and
    the forward multiplies it by the ``1 / sqrt(fan_in)`` of
    :func:`~mace_core.kernels.canonical.fully_connected_tp_path_normalization`,
    as ``e3nn`` does.
    """

    row: Tensor
    column: Tensor
    source: Tensor
    path_normalization: Tensor

    def __init__(self, descriptor: FullyConnectedTPDescriptor) -> None:
        super().__init__()
        second = Irreps.parse(descriptor.irreps_in2)
        if any(ir.degree != 0 or ir.parity != 1 for _, ir in second):
            raise NotImplementedError(
                f"the reference backend builds the fully connected tensor "
                f"product only against scalars, and {descriptor.irreps_in2!r} "
                f"is not. That is the case the models use, for the element "
                f"attributes; a general second input needs a backend that "
                f"declares it."
            )
        self.descriptor = descriptor
        dtype = TORCH_DTYPES[descriptor.precision]
        rows, columns, sources, count, _ = _linear_plan(
            LinearDescriptor(
                irreps_in=descriptor.irreps_in1,
                irreps_out=descriptor.irreps_out,
                precision=descriptor.precision,
            )
        )
        self.num_scalars = second.dimension
        self.dim_out = Irreps.parse(descriptor.irreps_out).dimension
        self.register_buffer("row", torch.tensor(rows, dtype=torch.long))
        self.register_buffer("column", torch.tensor(columns, dtype=torch.long))
        self.register_buffer("source", torch.tensor(sources, dtype=torch.long))
        self.weight = nn.Parameter(torch.zeros(self.num_scalars, count, dtype=dtype))
        self.register_buffer(
            "path_normalization",
            torch.tensor(_skip_path_normalizations(descriptor), dtype=dtype),
            persistent=False,
        )

    def forward(self, features: Tensor, attributes: Tensor) -> Tensor:
        weight = self.weight * self.path_normalization
        empty = features.new_zeros(0)
        empty_rows = self.row.new_zeros(0)
        # Zeros rather than `None`: with no scalar channels to weight by, the
        # sum is over an empty set and that is zero. Accumulating from `None`
        # made the declared return type a lie in exactly that case, and the
        # `None` would have travelled into the rest of the model.
        total = features.new_zeros(features.shape[0], self.dim_out)
        for scalar in range(self.num_scalars):
            mapped = equivariant_linear(
                features,
                weight[scalar],
                self.row,
                self.column,
                self.source,
                empty,
                empty_rows,
                self.dim_out,
            )
            total = total + mapped * attributes[:, scalar : scalar + 1]
        return total

    def initialize_weights(self, seed: int) -> None:
        """A standard normal, unscaled, since the forward applies the fan-in."""
        with torch.no_grad():
            self.weight.copy_(_draw(self.weight, seed))

    def to_canonical(self) -> dict[str, Tensor]:
        return {"weight": self.weight.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])


class ReferenceSegmentReduce(nn.Module):
    """A reduction into segments. Sum only, which is what the models use."""

    def __init__(self, descriptor: SegmentReduceDescriptor) -> None:
        super().__init__()
        if descriptor.reduction != "sum":
            raise NotImplementedError(
                f"the reference backend reduces by sum, not by "
                f"{descriptor.reduction!r}."
            )
        self.descriptor = descriptor

    def forward(self, values: Tensor, index: Tensor, num_segments: int) -> Tensor:
        return segment_sum(values, index, num_segments)


class ReferenceSphericalHarmonics(nn.Module):
    """Real spherical harmonics in this project's convention.

    Computed in the descriptor's precision whatever the input's, so the dtype
    is the one the model was built with and not whatever the caller had.
    """

    def __init__(self, descriptor: SphericalHarmonicsDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        self.dtype = TORCH_DTYPES[descriptor.precision]

    def forward(self, directions: Tensor) -> Tensor:
        return spherical_harmonics(
            directions.to(self.dtype), self.descriptor.lmax, self.descriptor.normalize
        )


class ReferenceRadialBasis(nn.Module):
    """The radial embedding, with the cutoff envelope already applied.

    Built and computed in the descriptor's precision, never in the process
    default dtype, and enveloped by a polynomial cutoff of the descriptor's
    ``cutoff_order``.
    """

    def __init__(self, descriptor: RadialBasisDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        self.dtype = TORCH_DTYPES[descriptor.precision]
        if descriptor.kind == "bessel":
            self.basis = BesselBasis(
                descriptor.cutoff, descriptor.num_basis, dtype=self.dtype
            )
        elif descriptor.kind == "gaussian":
            self.basis = GaussianBasis(
                descriptor.cutoff, descriptor.num_basis, dtype=self.dtype
            )
        elif descriptor.kind == "chebyshev":
            self.basis = ChebyshevBasis(descriptor.num_basis)
        else:
            raise ValueError(
                f"{descriptor.kind!r} is not a radial basis this backend builds. "
                f"The kinds are 'bessel', 'gaussian' and 'chebyshev', lowercase."
            )
        self.cutoff = PolynomialCutoff(
            descriptor.cutoff, descriptor.cutoff_order, dtype=self.dtype
        )

    def forward(self, lengths: Tensor) -> Tensor:
        """``[n_edges, 1]`` lengths in Angstrom -> ``[n_edges, num_basis]``."""
        lengths = lengths.to(self.dtype)
        return self.basis(lengths) * self.cutoff(lengths)


def _layer_widths(descriptor: RadialMLPDescriptor) -> list[tuple[int, int]]:
    """``(fan_in, fan_out)`` of every layer, input first."""
    return list(zip(descriptor.channels[:-1], descriptor.channels[1:], strict=True))


class ReferenceNormalizedSiluMLP(nn.Module):
    """The ``"normalized_silu"`` radial MLP: ``e3nn``'s ``FullyConnectedNet``.

    Each layer holds its raw ``[fan_in, fan_out]`` weight and multiplies it by
    ``1 / sqrt(fan_in)`` on every call, so the gradient with respect to the
    stored weight is the frozen tree's. Each hidden activation is a SiLU times
    :data:`~mace_core.kernels.canonical.SILU_SECOND_MOMENT_SCALE`.
    """

    layer_normalization: Tensor

    def __init__(self, descriptor: RadialMLPDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        dtype = TORCH_DTYPES[descriptor.precision]
        widths = _layer_widths(descriptor)
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(fan_in, fan_out, dtype=dtype))
                for fan_in, fan_out in widths
            ]
        )
        self.register_buffer(
            "layer_normalization",
            torch.tensor(
                [
                    radial_mlp_layer_normalization(descriptor.kind, fan_in)
                    for fan_in, _ in widths
                ],
                dtype=dtype,
            ),
            persistent=False,
        )

    def forward(self, radial: Tensor) -> Tensor:
        """``[n_edges, channels[0]]`` -> ``[n_edges, channels[-1]]``."""
        activated = radial
        last = len(self.weights) - 1
        for index, weight in enumerate(self.weights):
            activated = activated @ (weight * self.layer_normalization[index])
            if index != last:
                activated = SILU_SECOND_MOMENT_SCALE * torch.nn.functional.silu(
                    activated
                )
        return activated

    def initialize_weights(self, seed: int) -> None:
        """A standard normal per layer, unscaled, since the forward applies the
        fan-in. Each layer draws from ``seed`` plus its position."""
        with torch.no_grad():
            for index, weight in enumerate(self.weights):
                weight.copy_(_draw(weight, seed + index))

    def to_canonical(self) -> dict[str, Tensor]:
        """A view: ``layer_{i}``, one raw matrix per layer."""
        return {
            f"layer_{index}": weight.detach()
            for index, weight in enumerate(self.weights)
        }

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            for index, weight in enumerate(self.weights):
                weight.copy_(state[f"layer_{index}"])


class ReferenceLayerNormSiluMLP(nn.Module):
    """The ``"layer_norm_silu"`` radial MLP: the nonlinear residual block's.

    ``x @ W + b`` per layer, then a layer norm and a SiLU on every layer but the
    last. The weight is held ``[fan_in, fan_out]``, the transpose of the
    ``torch.nn.Linear`` the frozen tree holds, so that both kinds store their
    matrices the same way round.
    """

    def __init__(self, descriptor: RadialMLPDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        dtype = TORCH_DTYPES[descriptor.precision]
        widths = _layer_widths(descriptor)
        hidden = [fan_out for _, fan_out in widths[:-1]]
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(fan_in, fan_out, dtype=dtype))
                for fan_in, fan_out in widths
            ]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(fan_out, dtype=dtype)) for _, fan_out in widths]
        )
        self.norm_weights = nn.ParameterList(
            [nn.Parameter(torch.ones(width, dtype=dtype)) for width in hidden]
        )
        self.norm_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(width, dtype=dtype)) for width in hidden]
        )

    def forward(self, radial: Tensor) -> Tensor:
        """``[n_edges, channels[0]]`` -> ``[n_edges, channels[-1]]``."""
        activated = radial
        last = len(self.weights) - 1
        for index, weight in enumerate(self.weights):
            activated = activated @ weight + self.biases[index]
            if index != last:
                norm_weight = self.norm_weights[index]
                activated = torch.nn.functional.layer_norm(
                    activated,
                    norm_weight.shape,
                    norm_weight,
                    self.norm_biases[index],
                    LAYER_NORM_EPSILON,
                )
                activated = torch.nn.functional.silu(activated)
        return activated

    def initialize_weights(self, seed: int) -> None:
        """``torch.nn.Linear``'s draw and an identity layer norm.

        Weight and bias uniform on ``[-1 / sqrt(fan_in), 1 / sqrt(fan_in)]``,
        both from one generator seeded with ``seed`` plus the layer's position,
        drawn on the host in float64 for the reason :func:`_draw` gives.
        """
        with torch.no_grad():
            for index, (weight, bias) in enumerate(
                zip(self.weights, self.biases, strict=True)
            ):
                bound = radial_mlp_initial_bound(int(weight.shape[0]))
                generator = torch.Generator().manual_seed(seed + index)
                for parameter in (weight, bias):
                    unit = torch.rand(
                        parameter.shape, generator=generator, dtype=torch.float64
                    )
                    parameter.copy_(
                        ((2.0 * unit - 1.0) * bound).to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                    )
            for norm_weight, norm_bias in zip(
                self.norm_weights, self.norm_biases, strict=True
            ):
                norm_weight.fill_(1.0)
                norm_bias.zero_()

    def to_canonical(self) -> dict[str, Tensor]:
        """A view: ``layer_{i}`` and ``bias_{i}`` per layer, and
        ``norm_weight_{i}`` and ``norm_bias_{i}`` per hidden layer."""
        state = {}
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            state[f"layer_{index}"] = weight.detach()
            state[f"bias_{index}"] = bias.detach()
        for index, (norm_weight, norm_bias) in enumerate(
            zip(self.norm_weights, self.norm_biases, strict=True)
        ):
            state[f"norm_weight_{index}"] = norm_weight.detach()
            state[f"norm_bias_{index}"] = norm_bias.detach()
        return state

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            for index, (weight, bias) in enumerate(
                zip(self.weights, self.biases, strict=True)
            ):
                weight.copy_(state[f"layer_{index}"])
                bias.copy_(state[f"bias_{index}"])
            for index, (norm_weight, norm_bias) in enumerate(
                zip(self.norm_weights, self.norm_biases, strict=True)
            ):
                norm_weight.copy_(state[f"norm_weight_{index}"])
                norm_bias.copy_(state[f"norm_bias_{index}"])


#: The radial MLP classes, by the kind a descriptor records.
_RADIAL_MLPS: dict[
    str, type[ReferenceNormalizedSiluMLP] | type[ReferenceLayerNormSiluMLP]
] = {
    "normalized_silu": ReferenceNormalizedSiluMLP,
    "layer_norm_silu": ReferenceLayerNormSiluMLP,
}


class ReferenceCapabilities(BackendCapabilities):
    """The coarse fields, plus the two shapes the reference does not build.

    Stated here rather than only inside the op constructors, so that
    :meth:`supports` and the factories give the same answer: a model build
    that asks first and builds second must never be told yes and then fail.
    """

    def unsupported_reason(self, descriptor: Descriptor) -> str | None:
        reason = super().unsupported_reason(descriptor)
        if reason is not None:
            return reason
        if (
            isinstance(descriptor, SegmentReduceDescriptor)
            and descriptor.reduction != "sum"
        ):
            return (
                f"the reference reduces by 'sum' only, not by {descriptor.reduction!r}"
            )
        if isinstance(descriptor, FullyConnectedTPDescriptor) and any(
            irrep.degree != 0 or irrep.parity != 1
            for _, irrep in Irreps.parse(descriptor.irreps_in2)
        ):
            return (
                f"the reference builds the fully connected tensor product only "
                f"against scalars, and irreps_in2={descriptor.irreps_in2!r} is "
                f"not scalars"
            )
        return None


class ReferenceBackend:
    """Plain torch, every op, no e3nn. The oracle the others are checked against."""

    name = "reference"

    def capabilities(self) -> BackendCapabilities:
        return ReferenceCapabilities(
            ops=DISPATCHED_OPS | REFERENCE_ONLY_OPS,
            devices=frozenset({"cpu", "cuda"}),
            dtypes=frozenset({"float64", "float32"}),
            max_lmax=0,
            layouts=frozenset({"mul_ir"}),
            bases=frozenset({"reduced", "full"}),
            supports_double_backward=True,
        )

    def _check(self, descriptor) -> None:
        self.capabilities().require(descriptor, self.name)

    def make_linear(self, descriptor: LinearDescriptor) -> ReferenceLinear:
        self._check(descriptor)
        return ReferenceLinear(descriptor)

    def make_channelwise_tp_conv(
        self, descriptor: ChannelwiseTPConvDescriptor
    ) -> ReferenceChannelwiseTPConv:
        self._check(descriptor)
        return ReferenceChannelwiseTPConv(descriptor)

    def make_symmetric_contraction(
        self, descriptor: SymmetricContractionDescriptor
    ) -> ReferenceSymmetricContraction:
        self._check(descriptor)
        return ReferenceSymmetricContraction(descriptor)

    def make_fully_connected_tp(
        self, descriptor: FullyConnectedTPDescriptor
    ) -> ReferenceFullyConnectedTP:
        self._check(descriptor)
        return ReferenceFullyConnectedTP(descriptor)

    def make_segment_reduce(
        self, descriptor: SegmentReduceDescriptor
    ) -> ReferenceSegmentReduce:
        self._check(descriptor)
        return ReferenceSegmentReduce(descriptor)

    def make_spherical_harmonics(
        self, descriptor: SphericalHarmonicsDescriptor
    ) -> ReferenceSphericalHarmonics:
        self._check(descriptor)
        return ReferenceSphericalHarmonics(descriptor)

    def make_radial_basis(
        self, descriptor: RadialBasisDescriptor
    ) -> ReferenceRadialBasis:
        self._check(descriptor)
        return ReferenceRadialBasis(descriptor)

    def make_radial_mlp(
        self, descriptor: RadialMLPDescriptor
    ) -> ReferenceNormalizedSiluMLP | ReferenceLayerNormSiluMLP:
        self._check(descriptor)
        return _RADIAL_MLPS[descriptor.kind](descriptor)

    def make_interaction_layer(self, descriptors) -> None:
        """The reference fuses nothing. ``None`` means op by op, which is the
        normal answer and not a failure."""
        return None
