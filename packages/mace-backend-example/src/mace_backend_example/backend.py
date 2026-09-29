"""The example backend: every dispatched op, in plain torch, on its own kernels.

Written the way a backend in its own wheel is written. It imports ``torch`` and
the public contract in ``mace_core``, and nothing of ``mace_torch``: the
coupling tables come from ``mace_core``'s Clebsch-Gordan basis and path
enumeration, the weight scales and path labels from its canonical layout, and
the kernels from :mod:`mace_backend_example.ops`.

What the contract asks of every op:

* **The canonical weights.** ``to_canonical`` and ``load_canonical`` read and
  write the layout ``mace_core.kernels.canonical`` pins, so a checkpoint any
  backend wrote loads here and one written here loads anywhere. This backend
  holds its weights in that layout already, so both are views.
* **The canonical draw.** ``initialize_weights`` draws a standard normal at the
  canonical scale of each weight.
* **Honest capabilities.** A descriptor this backend cannot compute is declined
  in :meth:`ExampleCapabilities.supports`, and asking for it anyway raises.

Features cross every op in the canonical ``mul_ir`` layout: each irrep term is
its copies one after another, and a term of ``C`` channels of ``mul`` copies
holds copy ``c * mul + u`` at position ``c``, ``u``. The two kernels that work
channel by channel see them channel-major, ``[N, C, one channel's
components]``.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.real_basis import wigner_3j_real
from mace_core.clebsch_gordan.reduced_basis import (
    full_symmetric_tensor_product_basis,
    reduced_symmetric_tensor_product_basis,
)
from mace_core.kernels.canonical import (
    KERNEL_SPEC_VERSION,
    contraction_path_labels,
    fully_connected_tp_weight_scale,
    linear_weight_scale,
)
from mace_core.kernels.capabilities import (
    BackendCapabilities,
    UnsupportedDescriptorError,
)
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    Descriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    SegmentReduceDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.paths import channelwise_paths
from mace_core.kernels.protocol import DISPATCHED_OPS
from torch import Tensor, nn

from mace_backend_example.ops import contraction, segment_sum, tp_conv

__all__ = ["ExampleBackend", "ExampleCapabilities"]

_DTYPES = {"float64": torch.float64, "float32": torch.float32}

#: ``(mul, d)`` of every term of a declaration.
Terms = tuple[tuple[int, int], ...]


def _terms(irreps: str) -> Terms:
    return tuple((mul, ir.dimension) for mul, ir in Irreps.parse(irreps).terms)


def _channel_major(features: Tensor, terms: Terms, channels: int) -> Tensor:
    """``[N, channels * dim]`` to ``[N, channels, dim]``, for one channel's
    ``terms``."""
    pieces, offset = [], 0
    for mul, dimension in terms:
        width = channels * mul * dimension
        block = features[:, offset : offset + width]
        pieces.append(block.reshape(features.shape[0], channels, mul * dimension))
        offset += width
    return torch.cat(pieces, dim=-1)


def _grouped(values: Tensor, terms: Terms) -> Tensor:
    """The inverse of :func:`_channel_major`."""
    nodes, channels = values.shape[0], values.shape[1]
    pieces, offset = [], 0
    for mul, dimension in terms:
        width = mul * dimension
        block = values[:, :, offset : offset + width]
        pieces.append(block.reshape(nodes, channels * width))
        offset += width
    return torch.cat(pieces, dim=-1)


def _draw(like: Tensor, seed: int) -> Tensor:
    """A standard normal like ``like``, from its own generator, on the host."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(like.shape, generator=generator, dtype=torch.float64).to(
        device=like.device, dtype=like.dtype
    )


# ---------------------------------------------------------------------------
# Linear maps
# ---------------------------------------------------------------------------


def _linear_entries(irreps_in: str, irreps_out: str):
    """Where each canonical weight of a linear map lands in its dense matrix,
    and at what scale it is drawn.

    One weight per pair of copies of the same irrep, output copies outermost
    and input copies in declaration order within, repeated over the irrep's
    components. That is the canonical order; the repetition is what makes the
    map equivariant.
    """
    source, target = Irreps.parse(irreps_in), Irreps.parse(irreps_out)
    rows, columns, weights, scales = [], [], [], []
    weight = 0
    for out_slice, out_ir in target.slices():
        for in_slice, in_ir in source.slices():
            if in_ir != out_ir:
                continue
            for component in range(in_ir.dimension):
                rows.append(out_slice.start + component)
                columns.append(in_slice.start + component)
                weights.append(weight)
            scales.append(linear_weight_scale(irreps_in, out_ir))
            weight += 1
    return rows, columns, weights, scales


class _DenseMap(nn.Module):
    """The index tables of one linear map, as buffers that follow the module."""

    row: Tensor
    column: Tensor
    source: Tensor

    def __init__(self, irreps_in: str, irreps_out: str) -> None:
        super().__init__()
        rows, columns, weights, scales = _linear_entries(irreps_in, irreps_out)
        self.shape = (
            Irreps.parse(irreps_out).dimension,
            Irreps.parse(irreps_in).dimension,
        )
        self.count = len(scales)
        self.scales = scales
        self.register_buffer(
            "row", torch.tensor(rows, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "column", torch.tensor(columns, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "source", torch.tensor(weights, dtype=torch.long), persistent=False
        )

    def matrix(self, weight: Tensor) -> Tensor:
        """The dense ``[dim_out, dim_in]`` matrix of these weights."""
        values = weight[self.source]
        return weight.new_zeros(self.shape).index_put((self.row, self.column), values)


class ExampleLinear(nn.Module):
    """An equivariant linear map, with a bias on the even scalar outputs."""

    bias_row: Tensor

    def __init__(self, descriptor: LinearDescriptor) -> None:
        super().__init__()
        dtype = _DTYPES[descriptor.precision]
        self.map = _DenseMap(descriptor.irreps_in, descriptor.irreps_out)
        bias_rows = [
            out_slice.start
            for out_slice, ir in Irreps.parse(descriptor.irreps_out).slices()
            if descriptor.has_bias and ir.degree == 0 and ir.parity == 1
        ]
        self.register_buffer(
            "bias_row", torch.tensor(bias_rows, dtype=torch.long), persistent=False
        )
        self.weight = nn.Parameter(torch.zeros(self.map.count, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(len(bias_rows), dtype=dtype))

    def forward(self, features: Tensor) -> Tensor:
        mapped = features @ self.map.matrix(self.weight).T
        return mapped.index_add(1, self.bias_row, self.bias.expand(mapped.shape[0], -1))

    def initialize_weights(self, seed: int) -> None:
        scale = torch.tensor(self.map.scales, dtype=self.weight.dtype)
        with torch.no_grad():
            self.weight.copy_(_draw(self.weight, seed) * scale.to(self.weight.device))
            self.bias.zero_()

    def to_canonical(self) -> dict[str, Tensor]:
        return {"weight": self.weight.detach(), "bias": self.bias.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])
            self.bias.copy_(state["bias"])


class ExampleFullyConnectedTP(nn.Module):
    """The skip connection against scalar attributes: one linear map per
    attribute, weighted by it."""

    def __init__(self, descriptor: FullyConnectedTPDescriptor) -> None:
        super().__init__()
        dtype = _DTYPES[descriptor.precision]
        self.map = _DenseMap(descriptor.irreps_in1, descriptor.irreps_out)
        self.num_scalars = Irreps.parse(descriptor.irreps_in2).dimension
        source = Irreps.parse(descriptor.irreps_in1)
        target = Irreps.parse(descriptor.irreps_out)
        self.scales = [
            fully_connected_tp_weight_scale(in_mul, self.num_scalars)
            for _, out_ir in target.slices()
            for in_mul, in_ir in source.terms
            if in_ir == out_ir
            for _ in range(in_mul)
        ]
        self.weight = nn.Parameter(
            torch.zeros(self.num_scalars, self.map.count, dtype=dtype)
        )

    def forward(self, features: Tensor, attributes: Tensor) -> Tensor:
        # One matrix per attribute, stacked, and each node mixes the mapped
        # features by its own attributes. Two products of two rather than one
        # of three: see `ops._edge_coupling` for why.
        matrices = torch.stack(
            [self.map.matrix(self.weight[s]) for s in range(self.num_scalars)]
        )
        mapped = torch.einsum("ni,soi->nso", features, matrices)
        return torch.einsum("ns,nso->no", attributes, mapped)

    def initialize_weights(self, seed: int) -> None:
        scale = torch.tensor(self.scales, dtype=self.weight.dtype)
        with torch.no_grad():
            self.weight.copy_(_draw(self.weight, seed) * scale.to(self.weight.device))

    def to_canonical(self) -> dict[str, Tensor]:
        return {"weight": self.weight.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])


# ---------------------------------------------------------------------------
# The convolution
# ---------------------------------------------------------------------------


def _coupling(descriptor: ChannelwiseTPConvDescriptor) -> tuple[np.ndarray, list[int]]:
    """``[O, I, J]`` coupling coefficients over every path, and the path of
    each output component.

    Each path writes its own block of output components, the real Wigner 3j
    symbol of its three degrees scaled by the square root of the output
    dimension, so each path's output has unit variance when its inputs do.
    """
    node = Irreps.parse(descriptor.irreps_node)
    edge = Irreps.parse(descriptor.irreps_edge)
    node_slices = list(node.slices())
    edge_slices = list(edge.slices())
    paths = channelwise_paths(
        descriptor.irreps_node, descriptor.irreps_edge, descriptor.irreps_out
    )
    width = sum(path.irrep.dimension for path in paths)
    coupling = np.zeros((width, node.dimension, edge.dimension))
    owner: list[int] = []
    offset = 0
    for number, path in enumerate(paths):
        in_slice, in_ir = node_slices[path.node_term]
        edge_slice, edge_ir = edge_slices[path.edge_term]
        rows = slice(offset, offset + path.irrep.dimension)
        coupling[rows, in_slice, edge_slice] = wigner_3j_real(
            path.irrep.degree, in_ir.degree, edge_ir.degree
        ) * np.sqrt(path.irrep.dimension)
        owner.extend([number] * path.irrep.dimension)
        offset += path.irrep.dimension
    return coupling, owner


class ExampleChannelwiseTPConv(nn.Module):
    """The message-passing tensor product, gathered and scattered per edge."""

    coupling: Tensor
    path: Tensor

    def __init__(self, descriptor: ChannelwiseTPConvDescriptor) -> None:
        super().__init__()
        dtype = _DTYPES[descriptor.precision]
        coupling, owner = _coupling(descriptor)
        self.register_buffer(
            "coupling", torch.tensor(coupling, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "path", torch.tensor(owner, dtype=torch.long), persistent=False
        )
        paths = channelwise_paths(
            descriptor.irreps_node, descriptor.irreps_edge, descriptor.irreps_out
        )
        self.num_features = descriptor.num_features
        self.node_terms = _terms(descriptor.irreps_node)
        self.path_terms = tuple((1, path.irrep.dimension) for path in paths)

    @property
    def num_paths(self) -> int:
        """How many weights the radial network produces per edge and channel."""
        return len(self.path_terms)

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        radial_weights: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> Tensor:
        node = _channel_major(node_features, self.node_terms, self.num_features)
        messages = tp_conv(
            node,
            edge_attributes,
            radial_weights,
            self.coupling,
            self.path,
            sender,
            receiver,
            num_nodes,
        )
        return _grouped(messages, self.path_terms)


# ---------------------------------------------------------------------------
# The symmetric contraction
# ---------------------------------------------------------------------------


def _symmetrized(array: np.ndarray, order: int) -> np.ndarray:
    """The basis averaged over every order of its input axes.

    The basis is symmetric already, to rounding; making it exactly so is what
    lets the backward differentiate a symmetric form of degree k as k times
    the form of degree k - 1.
    """
    if order < 2:
        return array
    permutations = list(itertools.permutations(range(2, 2 + order)))
    stacked = np.stack([np.transpose(array, (0, 1, *p)) for p in permutations])
    return stacked.mean(axis=0)


class ExampleSymmetricContraction(nn.Module):
    """The many-body contraction, one symmetric power at a time."""

    def __init__(self, descriptor: SymmetricContractionDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        dtype = _DTYPES[descriptor.precision]
        build = (
            reduced_symmetric_tensor_product_basis
            if descriptor.basis == "reduced"
            else full_symmetric_tensor_product_basis
        )
        dimension = Irreps.parse(descriptor.irreps_in).dimension
        outputs = [str(ir) for _, ir in Irreps.parse(descriptor.irreps_out)]
        self.widths = [Irreps.parse(ir).dimension for ir in outputs]
        self.orders = descriptor.correlation
        # Held in the canonical order, body order outermost and the output
        # irreps within it, which is what makes the canonical form a view.
        self.targets: list[int] = []
        weights, bases = [], []
        for order in range(1, descriptor.correlation + 1):
            for target, irrep in enumerate(outputs):
                array = build(descriptor.irreps_in, order, irrep)[irrep]
                paths = array.shape[0]
                shaped = array.reshape(paths, self.widths[target], *[dimension] * order)
                bases.append(
                    torch.tensor(_symmetrized(shaped, order), dtype=torch.float64).to(
                        dtype
                    )
                )
                weights.append(
                    nn.Parameter(
                        torch.zeros(
                            descriptor.num_elements,
                            paths,
                            descriptor.num_features,
                            dtype=dtype,
                        )
                    )
                )
                self.targets.append(target)
        self.weights = nn.ParameterList(weights)
        for position, basis in enumerate(bases):
            self.register_buffer(f"basis_{position}", basis, persistent=False)
        self.in_terms = _terms(descriptor.irreps_in)
        self.out_terms = _terms(descriptor.irreps_out)

    def _bases(self) -> list[Tensor]:
        return [getattr(self, f"basis_{p}") for p in range(len(self.targets))]

    def _weights(self) -> list[Tensor]:
        return [self.weights[p] for p in range(len(self.targets))]

    def forward(self, features: Tensor, element: Tensor) -> Tensor:
        channels = self.descriptor.num_features
        joined = contraction(
            _channel_major(features, self.in_terms, channels),
            self._weights(),
            self._bases(),
            element,
            self.targets,
            self.widths,
        )
        return _grouped(joined, self.out_terms)

    def initialize_weights(self, seed: int) -> None:
        """A standard normal per element, path and channel, unscaled, which is
        the canonical draw of the contraction."""
        with torch.no_grad():
            for position, parameter in enumerate(self._weights()):
                parameter.copy_(_draw(parameter, seed + position))

    def canonical_metadata(self) -> dict[str, object]:
        """What each weight on the path axis multiplies, by name."""
        return {
            "spec_version": KERNEL_SPEC_VERSION,
            "basis": self.descriptor.basis,
            "paths": list(
                contraction_path_labels(
                    self.descriptor.irreps_in,
                    self.descriptor.irreps_out,
                    self.descriptor.correlation,
                    basis=self.descriptor.basis,
                )
            ),
        }

    def to_canonical(self) -> dict[str, Tensor]:
        weights = self._weights()
        return {
            "weight": torch.cat([w.detach() for w in weights], dim=1),
            "path_counts": torch.tensor([w.shape[1] for w in weights]),
        }

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        counts = [int(n) for n in state["path_counts"]]
        pieces = torch.split(state["weight"], counts, dim=1)
        with torch.no_grad():
            for parameter, piece in zip(self._weights(), pieces, strict=True):
                parameter.copy_(piece)


class ExampleSegmentReduce(nn.Module):
    """A sum into segments."""

    def forward(self, values: Tensor, index: Tensor, num_segments: int) -> Tensor:
        return segment_sum(values, index, num_segments)


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


def _single_copies(irreps: str) -> bool:
    return all(mul == 1 for mul, _ in Irreps.parse(irreps).terms)


class ExampleCapabilities(BackendCapabilities):
    """The coarse fields, and the shapes these ops were written for."""

    def supports(self, descriptor: Descriptor) -> bool:
        if not super().supports(descriptor):
            return False
        if isinstance(descriptor, ChannelwiseTPConvDescriptor):
            # A path reads one node term and one edge term, and the coupling
            # is written for one copy of each.
            return _single_copies(descriptor.irreps_node) and _single_copies(
                descriptor.irreps_edge
            )
        if isinstance(descriptor, FullyConnectedTPDescriptor):
            return all(
                ir.degree == 0 and ir.parity == 1
                for _, ir in Irreps.parse(descriptor.irreps_in2).terms
            )
        if isinstance(descriptor, SegmentReduceDescriptor):
            return descriptor.reduction == "sum"
        return True


class ExampleBackend:
    """Plain torch on its own kernels, registered only through an entry point."""

    name = "example"

    def capabilities(self) -> ExampleCapabilities:
        return ExampleCapabilities(
            ops=DISPATCHED_OPS,
            devices=frozenset({"cpu", "cuda"}),
            dtypes=frozenset({"float64", "float32"}),
            layouts=frozenset({"mul_ir"}),
            activation_layouts=frozenset({"mul_ir"}),
            native_layout="mul_ir",
            bases=frozenset({"reduced", "full"}),
            supports_double_backward=True,
            spec_version=KERNEL_SPEC_VERSION,
        )

    def _check(self, descriptor: Descriptor) -> None:
        if not self.capabilities().supports(descriptor):
            raise UnsupportedDescriptorError(
                f"the example backend does not build {descriptor!r}. It builds "
                f"convolutions over single copies of each node and edge irrep, "
                f"skips against scalar attributes, and sums."
            )

    def make_linear(self, descriptor: LinearDescriptor) -> ExampleLinear:
        self._check(descriptor)
        return ExampleLinear(descriptor)

    def make_channelwise_tp_conv(
        self, descriptor: ChannelwiseTPConvDescriptor
    ) -> ExampleChannelwiseTPConv:
        self._check(descriptor)
        return ExampleChannelwiseTPConv(descriptor)

    def make_symmetric_contraction(
        self, descriptor: SymmetricContractionDescriptor
    ) -> ExampleSymmetricContraction:
        self._check(descriptor)
        return ExampleSymmetricContraction(descriptor)

    def make_fully_connected_tp(
        self, descriptor: FullyConnectedTPDescriptor
    ) -> ExampleFullyConnectedTP:
        self._check(descriptor)
        return ExampleFullyConnectedTP(descriptor)

    def make_segment_reduce(
        self, descriptor: SegmentReduceDescriptor
    ) -> ExampleSegmentReduce:
        self._check(descriptor)
        return ExampleSegmentReduce()

    def make_spherical_harmonics(self, descriptor: Descriptor) -> None:
        """Left to the reference, which is what ``None`` says."""
        return None

    def make_radial_basis(self, descriptor: Descriptor) -> None:
        """Left to the reference."""
        return None

    def make_interaction_layer(self, descriptors: Sequence[Descriptor]) -> None:
        """No fused span: every op is built on its own."""
        return None
