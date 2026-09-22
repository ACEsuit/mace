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

import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.real_basis import wigner_3j_real
from mace_core.clebsch_gordan.reduced_basis import (
    full_symmetric_tensor_product_basis,
    reduced_symmetric_tensor_product_basis,
)
from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    RadialBasisDescriptor,
    SegmentReduceDescriptor,
    SphericalHarmonicsDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.protocol import DISPATCHED_OPS, REFERENCE_ONLY_OPS
from torch import Tensor, nn

from mace_torch.backends.harmonics import spherical_harmonics
from mace_torch.backends.radial import radial_basis
from mace_torch.kernels.ops import (
    channelwise_tp_conv,
    equivariant_linear,
    segment_sum,
    symmetric_contraction,
)

__all__ = ["ReferenceBackend"]

_TORCH_DTYPE = {"float64": torch.float64, "float32": torch.float32}


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


class ReferenceLinear(nn.Module):
    """An equivariant linear map with first-class bias."""

    def __init__(self, descriptor: LinearDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        rows, columns, sources, count, bias_rows = _linear_plan(descriptor)
        dtype = _TORCH_DTYPE[descriptor.precision]
        self.dim_out = Irreps.parse(descriptor.irreps_out).dimension
        self.register_buffer("row", torch.tensor(rows, dtype=torch.long))
        self.register_buffer("column", torch.tensor(columns, dtype=torch.long))
        self.register_buffer("source", torch.tensor(sources, dtype=torch.long))
        self.register_buffer("bias_row", torch.tensor(bias_rows, dtype=torch.long))
        self.weight = nn.Parameter(torch.zeros(count, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(len(bias_rows), dtype=dtype))

    def forward(self, features: Tensor) -> Tensor:
        return equivariant_linear(
            features,
            self.weight,
            self.row,
            self.column,
            self.source,
            self.bias,
            self.bias_row,
            self.dim_out,
        )

    def to_canonical(self) -> dict[str, Tensor]:
        """A view. The reference holds the canonical layout already."""
        return {"weight": self.weight.detach(), "bias": self.bias.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])
            self.bias.copy_(state["bias"])


class ReferenceSymmetricContraction(nn.Module):
    """The many-body contraction, over the basis the descriptor records."""

    def __init__(self, descriptor: SymmetricContractionDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        dtype = _TORCH_DTYPE[descriptor.precision]
        build = (
            reduced_symmetric_tensor_product_basis
            if descriptor.basis == "reduced"
            else full_symmetric_tensor_product_basis
        )
        self.orders = descriptor.correlation
        weights = []
        for order in range(1, descriptor.correlation + 1):
            stacked = [
                array
                for array in build(
                    descriptor.irreps_in, order, descriptor.irreps_out
                ).values()
            ]
            basis = np.concatenate(stacked, axis=0) if stacked else np.zeros((0, 1, 1))
            flat = basis.reshape(basis.shape[0], basis.shape[1], -1)
            self.register_buffer(
                f"basis_{order}", torch.tensor(flat, dtype=dtype), persistent=False
            )
            weights.append(
                nn.Parameter(
                    torch.zeros(
                        descriptor.num_elements,
                        flat.shape[0],
                        descriptor.num_features,
                        dtype=dtype,
                    )
                )
            )
        self.weights = nn.ParameterList(weights)

    def _bases(self) -> list[Tensor]:
        return [getattr(self, f"basis_{order}") for order in range(1, self.orders + 1)]

    def forward(self, features: Tensor, element: Tensor) -> Tensor:
        return symmetric_contraction(
            features, list(self.weights), self._bases(), element
        )

    def to_canonical(self) -> dict[str, Tensor]:
        """The flat ``[Z, A, mul]`` array, joined over the body orders.

        The per-order tensors are contiguous slices of it in the pinned path
        order, so this is a concatenate rather than a conversion.
        """
        return {
            "weight": torch.cat([w.detach() for w in self.weights], dim=1),
            "path_counts": torch.tensor([w.shape[1] for w in self.weights]),
        }

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        counts = [int(n) for n in state["path_counts"]]
        pieces = torch.split(state["weight"], counts, dim=1)
        with torch.no_grad():
            for parameter, piece in zip(self.weights, pieces, strict=True):
                parameter.copy_(piece)


def _coupling_coefficients(descriptor: ChannelwiseTPConvDescriptor) -> np.ndarray:
    """The Clebsch-Gordan coefficients of the message-passing product.

    One path per ``(node irrep, edge irrep, output irrep)`` the selection rules
    allow, in the pinned irrep order, written into a dense
    ``[paths, dim_out, dim_in, dim_edge]`` array. Dense because this is the
    reference; a backend with a real kernel keeps them sparse.
    """
    node = Irreps.parse(descriptor.irreps_node)
    edge = Irreps.parse(descriptor.irreps_edge)
    target = Irreps.parse(descriptor.irreps_out)
    paths = []
    for out_slice, out_ir in target.slices():
        for in_slice, in_ir in node.slices():
            for edge_slice, edge_ir in edge.slices():
                if out_ir not in set(in_ir.couple(edge_ir)):
                    continue
                block = np.zeros((target.dimension, node.dimension, edge.dimension))
                block[out_slice, in_slice, edge_slice] = wigner_3j_real(
                    out_ir.degree, in_ir.degree, edge_ir.degree
                )
                paths.append(block)
    if not paths:
        return np.zeros((0, target.dimension, node.dimension, edge.dimension))
    return np.stack(paths)


class ReferenceChannelwiseTPConv(nn.Module):
    #: Annotated because `register_buffer` alone leaves it typed as a `Module`,
    #: and then reading its shape reads as subscripting a module.
    coefficients: Tensor

    """The message-passing tensor product. Node-level, always."""

    def __init__(self, descriptor: ChannelwiseTPConvDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        dtype = _TORCH_DTYPE[descriptor.precision]
        coefficients = _coupling_coefficients(descriptor)
        self.register_buffer(
            "coefficients", torch.tensor(coefficients, dtype=dtype), persistent=False
        )

    @property
    def num_paths(self) -> int:
        """How many weights the external radial MLP has to produce."""
        return int(self.coefficients.shape[0])

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        radial_weights: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> Tensor:
        return channelwise_tp_conv(
            node_features,
            edge_attributes,
            radial_weights,
            self.coefficients,
            sender,
            receiver,
            num_nodes,
        )


class ReferenceFullyConnectedTP(nn.Module):
    row: Tensor
    column: Tensor
    source: Tensor

    """The skip connection's tensor product against the element attributes.

    Only the case the models use is built: the second input is scalars, the
    element one-hots, so the product is a per-element linear map. The general
    case raises rather than returning something plausible, because a wrong skip
    connection is a wrong model that still trains.
    """

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
        dtype = _TORCH_DTYPE[descriptor.precision]
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

    def forward(self, features: Tensor, attributes: Tensor) -> Tensor:
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
                self.weight[scalar],
                self.row,
                self.column,
                self.source,
                empty,
                empty_rows,
                self.dim_out,
            )
            total = total + mapped * attributes[:, scalar : scalar + 1]
        return total

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
    """Real spherical harmonics in this project's convention."""

    def __init__(self, descriptor: SphericalHarmonicsDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor

    def forward(self, directions: Tensor) -> Tensor:
        return spherical_harmonics(
            directions, self.descriptor.lmax, self.descriptor.normalize
        )


class ReferenceRadialBasis(nn.Module):
    """The radial embedding, with the cutoff envelope already applied."""

    def __init__(self, descriptor: RadialBasisDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor

    def forward(self, lengths: Tensor) -> Tensor:
        return radial_basis(
            lengths,
            self.descriptor.kind,
            self.descriptor.num_basis,
            self.descriptor.cutoff,
        )


class ReferenceBackend:
    """Plain torch, every op, no e3nn. The oracle the others are checked against."""

    name = "reference"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
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

    def make_interaction_layer(self, descriptors) -> None:
        """The reference fuses nothing. ``None`` means op by op, which is the
        normal answer and not a failure."""
        return None
