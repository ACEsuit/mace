"""The cuEquivariance backend: NVIDIA's kernels behind the kernel protocol.

It builds the four ops that have a cuEquivariance kernel: the linear map, the
fused convolution, the symmetric contraction and the skip connection. Every
other op, and every shape declined here, is built by the reference through
:class:`~mace_torch.backends.composite.CompositeBackend`.

**The layout is the chain's.** Each op is built with ``cue.mul_ir``, the
grouped layout every other op reads and writes, so nothing in the model
permutes around it. cuEquivariance's native layout is ``ir_mul``, and in
``mul_ir`` its ops transpose inside themselves. Measured on an A100 in float32
at 128 channels, 864 atoms and 76768 edges, ``mul_ir`` against ``ir_mul``: the
convolution 1.49 against 1.27 ms, the linear map 2.63 against 2.01 ms, the
contraction 5.34 against 4.65 ms. Moving the whole chain to ``ir_mul`` is a
change to every op and the reference, and is left for when that difference is
worth it.

**The weights are the canonical ones, mapped once.** Each op holds its
weights in cuEquivariance's own order and maps them to and from the canonical
form only at the checkpoint boundary:

* the linear map and the skip hold the frozen tree's ``e3nn`` order, one block
  per coupled pair of terms laid out ``[mul_in, (scalars,) mul_out]``, without
  the normalization the canonical form folds in. The map is a permutation and
  a per-weight scale;
* the convolution has no weights of its own. Its paths are in the pinned order
  and normalized alike, so the radial weights pass through unchanged;
* the symmetric contraction holds ``[Z, A, mul]`` over cuEquivariance's own
  reduced basis. The two bases span the same space and enumerate it
  differently, so the map is an ``A x A`` matrix applied per element and
  channel. It is derived once per basis shape by evaluating both bases on the
  same inputs, and cached. Measured, it is block diagonal with blocks of size
  one and two, and exact to rounding.

A fresh draw is the reference's draw, loaded through the canonical form, so a
seed builds the same model on either backend.

**What it declines.** A biased linear map (cuEquivariance's has no bias), a
convolution or contraction whose terms do not all have one copy per channel
(its kernels need a uniform multiplicity), a skip whose second input is not a
single block of scalars, and a contraction over the full basis.

With its compiled operations, cuEquivariance runs on CUDA only: some of them,
such as the layout transpose its convolution applies to its own output, have
no CPU kernel. Without them it runs its pure-torch path, correct and slow, on
the CPU. Neither is hidden: the capabilities list the one device that works,
and the backend says so when it is built. Without cuEquivariance at all,
importing this module fails, which is what the registry records.
"""

from __future__ import annotations

import logging
from functools import cache
from typing import Any

import cuequivariance as cue  # ty: ignore[unresolved-import]
import cuequivariance_torch as cuet  # ty: ignore[unresolved-import]
import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.capabilities import (
    BackendCapabilities,
    UnsupportedDescriptorError,
)
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    Descriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.paths import channelwise_paths
from torch import Tensor, nn

from mace_torch.backends.reference import ReferenceBackend
from mace_torch.nn.layout import expanded_irreps

__all__ = ["CuEqBackend", "compiled_operations_available"]

logger = logging.getLogger(__name__)

_DTYPES = {"float64": torch.float64, "float32": torch.float32}


def compiled_operations_available() -> bool:
    """Whether cuEquivariance's compiled operations import on this machine."""
    try:
        import cuequivariance_ops_torch  # noqa: F401  # ty: ignore[unresolved-import]
    except Exception:
        return False
    return True


def _irreps(text: str) -> cue.Irreps:
    return cue.Irreps("O3", text)


def _uniform(irreps: str) -> bool:
    """One copy of every term per channel, which is what the kernels need."""
    return all(multiplicity == 1 for multiplicity, _ in Irreps.parse(irreps).terms)


class CuEqCapabilities(BackendCapabilities):
    """The coarse fields, and the shapes the kernels take."""

    def supports(self, descriptor: Descriptor) -> bool:
        if not super().supports(descriptor):
            return False
        if isinstance(descriptor, LinearDescriptor):
            return not descriptor.has_bias
        if isinstance(descriptor, ChannelwiseTPConvDescriptor):
            return _uniform(descriptor.irreps_node)
        if isinstance(descriptor, SymmetricContractionDescriptor):
            return (
                descriptor.basis == "reduced"
                and _uniform(descriptor.irreps_in)
                and _uniform(descriptor.irreps_out)
            )
        if isinstance(descriptor, FullyConnectedTPDescriptor):
            terms = Irreps.parse(descriptor.irreps_in2).terms
            return (
                len(terms) == 1 and terms[0][1].degree == 0 and terms[0][1].parity == 1
            )
        return False


def _linear_order(irreps_in: str, irreps_out: str, scalars: int = 0) -> np.ndarray:
    """For each cuEquivariance weight, the canonical index it holds.

    cuEquivariance, like ``e3nn``, lays the weights out one block per coupled
    pair of terms, input term outermost, each block ``[mul_in, mul_out]``, or
    ``[mul_in, scalars, mul_out]`` for the skip. The canonical order has the
    output copy outermost and the input copies inside it, and the skip's
    scalars as a leading axis.
    """
    source = Irreps.parse(irreps_in).terms
    target = Irreps.parse(irreps_out).terms
    canonical: dict[tuple[int, int, int, int], int] = {}
    count = 0
    for o, (out_mul, out_ir) in enumerate(target):
        for v in range(out_mul):
            for i, (in_mul, in_ir) in enumerate(source):
                if in_ir != out_ir:
                    continue
                for u in range(in_mul):
                    canonical[i, o, u, v] = count
                    count += 1
    order = []
    for i, (in_mul, in_ir) in enumerate(source):
        for o, (out_mul, out_ir) in enumerate(target):
            if in_ir != out_ir:
                continue
            for u in range(in_mul):
                for scalar in range(max(scalars, 1)):
                    for v in range(out_mul):
                        offset = scalar * count if scalars else 0
                        order.append(offset + canonical[i, o, u, v])
    return np.asarray(order, dtype=np.int64)


class _PermutedWeights(nn.Module):
    """A cuEquivariance op whose flat weights are the canonical ones permuted
    and unscaled."""

    order: Tensor
    scale: Tensor

    def _setup(self, reference: Any, order: np.ndarray, operation: Any) -> None:
        self.operation = operation
        self.register_buffer("order", torch.tensor(order), persistent=False)
        self.register_buffer(
            "scale",
            reference.weight_scale.expand_as(reference.weight)
            .reshape(-1)[order]
            .clone(),
            persistent=False,
        )
        self._reference = [reference]

    def to_canonical(self) -> dict[str, Tensor]:
        weight = self.operation.weight.detach().reshape(-1) * self.scale
        canonical = torch.empty_like(weight)
        canonical[self.order] = weight
        reference = self._reference[0]
        return {"weight": canonical.reshape(reference.weight.shape)}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        flat = state["weight"].reshape(-1).to(self.scale)
        with torch.no_grad():
            self.operation.weight.copy_(
                (flat[self.order] / self.scale).reshape(self.operation.weight.shape)
            )

    def initialize_weights(self, seed: int) -> None:
        reference = self._reference[0]
        reference.initialize_weights(seed)
        self.load_canonical(reference.to_canonical())


class CuEqLinear(_PermutedWeights):
    """An equivariant linear map without a bias."""

    def __init__(self, descriptor: LinearDescriptor) -> None:
        super().__init__()
        dtype = _DTYPES[descriptor.precision]
        self.descriptor = descriptor
        reference = ReferenceBackend().make_linear(descriptor)
        operation = cuet.Linear(
            _irreps(descriptor.irreps_in),
            _irreps(descriptor.irreps_out),
            layout=cue.mul_ir,
            shared_weights=True,
            internal_weights=True,
            dtype=dtype,
            math_dtype=dtype,
        )
        self._setup(
            reference,
            _linear_order(descriptor.irreps_in, descriptor.irreps_out),
            operation,
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.operation(features)

    def to_canonical(self) -> dict[str, Tensor]:
        state = super().to_canonical()
        state["bias"] = self.operation.weight.new_zeros(0)
        return state


class CuEqFullyConnectedTP(_PermutedWeights):
    """The skip connection against a block of scalars."""

    def __init__(self, descriptor: FullyConnectedTPDescriptor) -> None:
        super().__init__()
        dtype = _DTYPES[descriptor.precision]
        self.descriptor = descriptor
        reference = ReferenceBackend().make_fully_connected_tp(descriptor)
        operation = cuet.FullyConnectedTensorProduct(
            _irreps(descriptor.irreps_in1),
            _irreps(descriptor.irreps_in2),
            _irreps(descriptor.irreps_out),
            layout=cue.mul_ir,
            shared_weights=True,
            internal_weights=True,
            dtype=dtype,
            math_dtype=dtype,
        )
        scalars = Irreps.parse(descriptor.irreps_in2).dimension
        self._setup(
            reference,
            _linear_order(descriptor.irreps_in1, descriptor.irreps_out, scalars),
            operation,
        )

    def forward(self, features: Tensor, attributes: Tensor) -> Tensor:
        return self.operation(features, attributes)


class CuEqChannelwiseTPConv(nn.Module):
    """The convolution, gather, product and scatter in one kernel call."""

    def __init__(self, descriptor: ChannelwiseTPConvDescriptor) -> None:
        super().__init__()
        dtype = _DTYPES[descriptor.precision]
        self.descriptor = descriptor
        paths = channelwise_paths(
            descriptor.irreps_node, descriptor.irreps_edge, descriptor.irreps_out
        )
        self.num_paths = len(paths)
        self.operation = cuet.ChannelWiseTensorProduct(
            _irreps(expanded_irreps(descriptor.irreps_node, descriptor.num_features)),
            _irreps(descriptor.irreps_edge),
            [ir for _, ir in _irreps(descriptor.irreps_out)],
            layout=cue.mul_ir,
            shared_weights=False,
            internal_weights=False,
            dtype=dtype,
            math_dtype=dtype,
        )
        expected = _irreps(
            "+".join(f"{descriptor.num_features}x{path.irrep}" for path in paths)
        )
        if self.operation.irreps_out != expected:
            raise UnsupportedDescriptorError(
                f"cuEquivariance orders the paths of {descriptor} as "
                f"{self.operation.irreps_out}, and the pinned order is {expected}"
            )

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        radial_weights: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> Tensor:
        return self.operation(
            node_features,
            edge_attributes,
            radial_weights.reshape(radial_weights.shape[0], -1),
            indices_1=sender,
            indices_out=receiver,
            size_out=num_nodes,
        )


@cache
def _contraction_map(irreps_in: str, irreps_out: str, correlation: int) -> np.ndarray:
    """``M`` with cuEquivariance's weights ``= M @`` the canonical ones.

    Both contractions are linear in their weights, so evaluating each basis
    direction of both on the same inputs gives two matrices whose columns span
    the same outputs; ``M`` solves one against the other. One channel and one
    element suffice: the map is the same for every channel and element.
    """
    descriptor = SymmetricContractionDescriptor(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        correlation=correlation,
        num_elements=1,
        num_features=1,
    )
    reference = ReferenceBackend().make_symmetric_contraction(descriptor)
    operation = cuet.SymmetricContraction(
        _irreps(irreps_in),
        _irreps(irreps_out),
        correlation,
        1,
        layout=cue.mul_ir,
        dtype=torch.float64,
        math_dtype=torch.float64,
        original_mace=False,
        method="naive",
    )
    paths = descriptor.path_count
    if operation.weight.shape[1] != paths:
        raise UnsupportedDescriptorError(
            f"cuEquivariance's basis for {descriptor} has "
            f"{operation.weight.shape[1]} paths and the canonical one {paths}"
        )
    generator = torch.Generator().manual_seed(0)
    samples = max(4 * paths, 32)
    features = torch.randn(
        samples,
        Irreps.parse(irreps_in).dimension,
        generator=generator,
        dtype=torch.float64,
    )
    element = torch.zeros(samples, dtype=torch.long)
    counts = reference.to_canonical()["path_counts"]
    columns_reference, columns_cueq = [], []
    with torch.no_grad():
        for path in range(paths):
            weight = torch.zeros(1, paths, 1, dtype=torch.float64)
            weight[0, path, 0] = 1.0
            reference.load_canonical({"weight": weight, "path_counts": counts})
            columns_reference.append(reference(features, element).reshape(-1))
            operation.weight.zero_()
            operation.weight[0, path, 0] = 1.0
            columns_cueq.append(operation(features, element).reshape(-1))
    canonical_outputs = torch.stack(columns_reference, dim=1)
    cueq_outputs = torch.stack(columns_cueq, dim=1)
    projection = torch.linalg.lstsq(cueq_outputs, canonical_outputs).solution
    residual = (cueq_outputs @ projection - canonical_outputs).abs().max().item()
    scale = canonical_outputs.abs().max().item()
    if residual > 1e-10 * max(scale, 1.0):
        raise UnsupportedDescriptorError(
            f"cuEquivariance's basis for {descriptor} does not span the "
            f"canonical one: residual {residual:.2e}"
        )
    return projection.numpy()


class CuEqSymmetricContraction(nn.Module):
    """The many-body contraction over cuEquivariance's reduced basis."""

    projection: Tensor
    inverse: Tensor

    def __init__(self, descriptor: SymmetricContractionDescriptor) -> None:
        super().__init__()
        dtype = _DTYPES[descriptor.precision]
        self.descriptor = descriptor
        features = descriptor.num_features
        self.operation = cuet.SymmetricContraction(
            _irreps(expanded_irreps(descriptor.irreps_in, features)),
            _irreps(expanded_irreps(descriptor.irreps_out, features)),
            descriptor.correlation,
            descriptor.num_elements,
            layout=cue.mul_ir,
            dtype=dtype,
            math_dtype=dtype,
            original_mace=False,
        )
        projection = _contraction_map(
            descriptor.irreps_in, descriptor.irreps_out, descriptor.correlation
        )
        self.register_buffer(
            "projection", torch.tensor(projection, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "inverse",
            torch.tensor(np.linalg.inv(projection), dtype=dtype),
            persistent=False,
        )
        self._reference = [ReferenceBackend().make_symmetric_contraction(descriptor)]
        self._path_counts = self._reference[0].to_canonical()["path_counts"]

    def forward(self, features: Tensor, element: Tensor) -> Tensor:
        return self.operation(features, element)

    def to_canonical(self) -> dict[str, Tensor]:
        weight = torch.einsum(
            "pq,zqc->zpc", self.inverse, self.operation.weight.detach()
        )
        return {"weight": weight, "path_counts": self._path_counts.clone()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        weight = state["weight"].to(self.projection)
        with torch.no_grad():
            self.operation.weight.copy_(
                torch.einsum("qp,zpc->zqc", self.projection, weight)
            )

    def initialize_weights(self, seed: int) -> None:
        reference = self._reference[0]
        reference.initialize_weights(seed)
        self.load_canonical(reference.to_canonical())


class CuEqBackend:
    """cuEquivariance for the four ops it has kernels for."""

    name = "cueq"

    def __init__(self) -> None:
        self.compiled = compiled_operations_available()
        if not self.compiled:
            logger.warning(
                "cuEquivariance's compiled operations did not import, so its "
                "ops run the pure-torch path: correct, and slow"
            )

    def capabilities(self) -> BackendCapabilities:
        return CuEqCapabilities(
            ops=frozenset(
                {
                    "linear",
                    "channelwise_tp_conv",
                    "symmetric_contraction",
                    "fully_connected_tp",
                }
            ),
            # With the compiled operations its ops run on CUDA only: some
            # have no CPU kernel. Without them, the pure-torch path runs
            # anywhere and is what a CPU host gets.
            devices=frozenset({"cuda"} if self.compiled else {"cpu"}),
            dtypes=frozenset({"float64", "float32"}),
            layouts=frozenset({"mul_ir", "ir_mul"}),
            bases=frozenset({"reduced"}),
            supports_double_backward=True,
        )

    def _check(self, descriptor: Descriptor) -> None:
        self.capabilities().require(descriptor, self.name)

    def make_linear(self, descriptor: LinearDescriptor) -> CuEqLinear:
        self._check(descriptor)
        return CuEqLinear(descriptor)

    def make_channelwise_tp_conv(
        self, descriptor: ChannelwiseTPConvDescriptor
    ) -> CuEqChannelwiseTPConv:
        self._check(descriptor)
        return CuEqChannelwiseTPConv(descriptor)

    def make_symmetric_contraction(
        self, descriptor: SymmetricContractionDescriptor
    ) -> CuEqSymmetricContraction:
        self._check(descriptor)
        return CuEqSymmetricContraction(descriptor)

    def make_fully_connected_tp(
        self, descriptor: FullyConnectedTPDescriptor
    ) -> CuEqFullyConnectedTP:
        self._check(descriptor)
        return CuEqFullyConnectedTP(descriptor)

    def make_interaction_layer(self, descriptors: tuple[Any, ...]) -> None:
        return None
