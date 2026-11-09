"""The cuEquivariance backend: NVIDIA's kernels behind the kernel protocol.

It builds the four ops that have a cuEquivariance kernel: the linear map, the
fused convolution, the symmetric contraction and the skip connection. Every
other op, and every shape declined here, is built by the reference through
:class:`~mace_torch.backends.composite.CompositeBackend`.

**The layout is the chain's.** Each op is built in the kernel specification's layout,
which is ``ir_mul``, cuEquivariance's native one, whenever cueq is the chosen
backend: the composite resolves the chain to it and the reference follows.
Built in ``mul_ir`` instead, its ops transpose inside themselves; measured on an
A100 at 1728 atoms and 128 channels, forward and backward in float64, the
linear map takes 3.27 ms in ``mul_ir`` and 1.22 in ``ir_mul``. In ``ir_mul``
the linear map and the skip use cuEquivariance's ``naive`` method, which is
what the frozen tree uses and what measured fastest there: the skip 1.91 ms
against 3.12 with the default method, in float64.

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
    UnsupportedKernelSpecError,
)
from mace_core.kernels.paths import channelwise_paths
from mace_core.kernels.reorder import ReorderError, WeightReorder, derive_reorder
from mace_core.kernels.specs import (
    ChannelwiseTPConvSpec,
    FullyConnectedTPSpec,
    KernelSpec,
    LinearSpec,
    SymmetricContractionSpec,
)
from torch import Tensor, nn

from mace_torch.backends.layout import expanded_irreps
from mace_torch.backends.reference import ReferenceBackend

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


def _layout(kernel_spec: KernelSpec) -> Any:
    """cuEquivariance's name for the kernel specification's feature layout."""
    return cue.ir_mul if kernel_spec.activation_layout == "ir_mul" else cue.mul_ir


def _method(kernel_spec: KernelSpec) -> str | None:
    """``naive`` in ``ir_mul``, the default otherwise: what measured fastest
    for the linear map and the skip in each, see the module docstring."""
    return "naive" if kernel_spec.activation_layout == "ir_mul" else None


class CuEqCapabilities(BackendCapabilities):
    """The coarse fields, and the shapes the kernels take."""

    def supports(self, kernel_spec: KernelSpec) -> bool:
        if not super().supports(kernel_spec):
            return False
        if isinstance(kernel_spec, LinearSpec):
            return not kernel_spec.has_bias
        if isinstance(kernel_spec, ChannelwiseTPConvSpec):
            return _uniform(kernel_spec.irreps_node)
        if isinstance(kernel_spec, SymmetricContractionSpec):
            return (
                kernel_spec.basis == "reduced"
                and _uniform(kernel_spec.irreps_in)
                and _uniform(kernel_spec.irreps_out)
            )
        if isinstance(kernel_spec, FullyConnectedTPSpec):
            terms = Irreps.parse(kernel_spec.irreps_in2).terms
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
            torch.ones_like(reference.weight).reshape(-1)[order].clone(),
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

    def __init__(self, kernel_spec: LinearSpec) -> None:
        super().__init__()
        dtype = _DTYPES[kernel_spec.precision]
        self.kernel_spec = kernel_spec
        reference = ReferenceBackend().make_linear(kernel_spec)
        operation = cuet.Linear(
            _irreps(kernel_spec.irreps_in),
            _irreps(kernel_spec.irreps_out),
            layout=_layout(kernel_spec),
            method=_method(kernel_spec),
            shared_weights=True,
            internal_weights=True,
            dtype=dtype,
            math_dtype=_DTYPES[kernel_spec.accumulate_floor],
        )
        self._setup(
            reference,
            _linear_order(kernel_spec.irreps_in, kernel_spec.irreps_out),
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

    def __init__(self, kernel_spec: FullyConnectedTPSpec) -> None:
        super().__init__()
        dtype = _DTYPES[kernel_spec.precision]
        self.kernel_spec = kernel_spec
        reference = ReferenceBackend().make_fully_connected_tp(kernel_spec)
        operation = cuet.FullyConnectedTensorProduct(
            _irreps(kernel_spec.irreps_in1),
            _irreps(kernel_spec.irreps_in2),
            _irreps(kernel_spec.irreps_out),
            layout=_layout(kernel_spec),
            method=_method(kernel_spec),
            shared_weights=True,
            internal_weights=True,
            dtype=dtype,
            math_dtype=_DTYPES[kernel_spec.accumulate_floor],
        )
        scalars = Irreps.parse(kernel_spec.irreps_in2).dimension
        self._setup(
            reference,
            _linear_order(kernel_spec.irreps_in1, kernel_spec.irreps_out, scalars),
            operation,
        )

    def forward(self, features: Tensor, attributes: Tensor) -> Tensor:
        return self.operation(features, attributes)


class CuEqChannelwiseTPConv(nn.Module):
    """The convolution, gather, product and scatter in one kernel call."""

    def __init__(self, kernel_spec: ChannelwiseTPConvSpec) -> None:
        super().__init__()
        dtype = _DTYPES[kernel_spec.precision]
        self.kernel_spec = kernel_spec
        paths = channelwise_paths(
            kernel_spec.irreps_node, kernel_spec.irreps_edge, kernel_spec.irreps_out
        )
        self.num_paths = len(paths)
        self.weight_width = self.num_paths * kernel_spec.num_features
        self.output_width = kernel_spec.num_features * sum(
            path.irrep.dimension for path in paths
        )
        self.operation = cuet.ChannelWiseTensorProduct(
            _irreps(expanded_irreps(kernel_spec.irreps_node, kernel_spec.num_features)),
            _irreps(kernel_spec.irreps_edge),
            [ir for _, ir in _irreps(kernel_spec.irreps_out)],
            layout=_layout(kernel_spec),
            shared_weights=False,
            internal_weights=False,
            dtype=dtype,
            math_dtype=_DTYPES[kernel_spec.accumulate_floor],
        )
        expected = _irreps(
            "+".join(f"{kernel_spec.num_features}x{path.irrep}" for path in paths)
        )
        if self.operation.irreps_out != expected:
            raise UnsupportedKernelSpecError(
                f"cuEquivariance orders the paths of {kernel_spec} as "
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
        # What `ChannelWiseTensorProduct.forward` does, with one difference:
        # the output's size is given by a tensor made on the device of the
        # inputs. The public forward makes it on the host and copies it over
        # on every call, which a CUDA graph cannot capture. This leans on the
        # module's `transpose_in1`, `transpose_in2`, `f` and `transpose_out`,
        # which cuEquivariance 0.10 has and does not document as public; the
        # conformance harness's CUDA graph check is what notices if they move.
        if sender.shape[0] == 0:
            # No edges, no messages. cuEquivariance's pure-torch path refuses
            # an empty edge batch outright, so it is not asked.
            return node_features.new_zeros(num_nodes, self.output_width)
        operation = self.operation
        output = operation.f(
            [
                # The width is spelled out: a structure with no edges has zero
                # rows, and `-1` cannot be inferred from an empty tensor.
                radial_weights.reshape(radial_weights.shape[0], self.weight_width),
                operation.transpose_in1(node_features),
                operation.transpose_in2(edge_attributes),
            ],
            input_indices={1: sender},
            output_shapes={0: node_features.new_empty(num_nodes, 1)},
            output_indices={0: receiver},
        )
        return operation.transpose_out(output[0])


@cache
def _contraction_map(
    irreps_in: str, irreps_out: str, correlation: int
) -> WeightReorder:
    """The map from canonical weights to cuEquivariance's, derived once.

    Both contractions are linear in their weights, so each weight direction's
    output on the same inputs is a basis vector of the function space the op
    spans: one set per implementation, in its own path order and
    normalization. :func:`~mace_core.kernels.derive_reorder` relates the two
    sets, decomposes the map into its independent blocks, and refuses when
    cuEquivariance does not reproduce the canonical space. Evaluating the ops
    rather than comparing the two basis arrays is what folds in whatever
    normalization cuEquivariance applies inside its kernel. One channel and one
    element suffice: the map is the same for every channel and element.
    """
    kernel_spec = SymmetricContractionSpec(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        correlation=correlation,
        num_elements=1,
        num_features=1,
    )
    reference = ReferenceBackend().make_symmetric_contraction(kernel_spec)
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
    paths = kernel_spec.path_count
    if operation.weight.shape[1] != paths:
        raise UnsupportedKernelSpecError(
            f"cuEquivariance's basis for {kernel_spec} has "
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
    canonical_outputs = torch.stack(columns_reference).numpy()
    cueq_outputs = torch.stack(columns_cueq).numpy()
    scale = max(float(np.abs(canonical_outputs).max()), 1.0)
    try:
        return derive_reorder(canonical_outputs, cueq_outputs, tolerance=1e-10 * scale)
    except ReorderError as error:
        raise UnsupportedKernelSpecError(
            f"cuEquivariance's contraction for {kernel_spec} cannot hold the "
            f"canonical weights: {error}"
        ) from error


class CuEqSymmetricContraction(nn.Module):
    """The many-body contraction over cuEquivariance's reduced basis.

    Its weights are cuEquivariance's own, ``[Z, A, mul]`` in its path order,
    and the canonical ones are carried across by a block-diagonal map derived
    once per basis shape. The map runs only at the checkpoint boundary.
    """

    def __init__(self, kernel_spec: SymmetricContractionSpec) -> None:
        super().__init__()
        dtype = _DTYPES[kernel_spec.precision]
        self.kernel_spec = kernel_spec
        features = kernel_spec.num_features
        self.operation = cuet.SymmetricContraction(
            _irreps(expanded_irreps(kernel_spec.irreps_in, features)),
            _irreps(expanded_irreps(kernel_spec.irreps_out, features)),
            kernel_spec.correlation,
            kernel_spec.num_elements,
            layout=_layout(kernel_spec),
            dtype=dtype,
            math_dtype=_DTYPES[kernel_spec.accumulate_floor],
            original_mace=False,
        )
        #: Canonical weights onto cuEquivariance's paths, and back.
        self.weight_reorder = _contraction_map(
            kernel_spec.irreps_in, kernel_spec.irreps_out, kernel_spec.correlation
        )
        self._restore = self.weight_reorder.inverse()
        self._reference = [ReferenceBackend().make_symmetric_contraction(kernel_spec)]
        self._path_counts = self._reference[0].to_canonical()["path_counts"]

    def forward(self, features: Tensor, element: Tensor) -> Tensor:
        return self.operation(features, element)

    def canonical_metadata(self) -> dict[str, object]:
        """The canonical layout's record, since what this op writes is the
        canonical layout whatever it holds internally."""
        return self._reference[0].canonical_metadata()

    def to_canonical(self) -> dict[str, Tensor]:
        held = self.operation.weight.detach()
        weight = self._restore.apply(held.cpu().double().numpy(), axis=1)
        return {
            "weight": torch.as_tensor(weight, dtype=held.dtype, device=held.device),
            "path_counts": self._path_counts.clone(),
        }

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        weight = self.weight_reorder.apply(
            state["weight"].detach().cpu().double().numpy(), axis=1
        )
        target = self.operation.weight
        with torch.no_grad():
            target.copy_(
                torch.as_tensor(weight, dtype=target.dtype, device=target.device)
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
            weight_layouts=frozenset({"mul_ir"}),
            activation_layouts=frozenset({"mul_ir", "ir_mul"}),
            native_activation_layout="ir_mul",
            bases=frozenset({"reduced"}),
            supports_double_backward=True,
            wide_accumulation=True,
        )

    def _check(self, kernel_spec: KernelSpec) -> None:
        self.capabilities().require(kernel_spec, self.name)

    def make_linear(self, kernel_spec: LinearSpec) -> CuEqLinear:
        self._check(kernel_spec)
        return CuEqLinear(kernel_spec)

    def make_channelwise_tp_conv(
        self, kernel_spec: ChannelwiseTPConvSpec
    ) -> CuEqChannelwiseTPConv:
        self._check(kernel_spec)
        return CuEqChannelwiseTPConv(kernel_spec)

    def make_symmetric_contraction(
        self, kernel_spec: SymmetricContractionSpec
    ) -> CuEqSymmetricContraction:
        self._check(kernel_spec)
        return CuEqSymmetricContraction(kernel_spec)

    def make_fully_connected_tp(
        self, kernel_spec: FullyConnectedTPSpec
    ) -> CuEqFullyConnectedTP:
        self._check(kernel_spec)
        return CuEqFullyConnectedTP(kernel_spec)

    def make_interaction_layer(self, kernel_specs: tuple[Any, ...]) -> None:
        return None
