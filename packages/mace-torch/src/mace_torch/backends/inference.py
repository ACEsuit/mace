"""Explicit preparation of optional inference regions from canonical weights.

Preparation produces an independent snapshot, not a cache attached to a model.
Reprepare after changing weights, device, dtype or member specifications. The
source model remains the only checkpoint and training authority. No registry,
weight conversion or device synchronization is consulted during a call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from mace_core.kernels.inference import (
    InferenceSpec,
    InteractionFusionSpec,
    ProductReadoutSpec,
)
from mace_core.kernels.protocol import KernelBackend
from torch import Tensor, nn

from mace_torch.backends.spans import SpanCapabilities, validate_span
from mace_torch.serialization import CheckpointError

__all__ = [
    "CanonicalRegionWeights",
    "InferenceBackend",
    "InferencePreparation",
    "PreparedInference",
    "prepare_inference",
]

CanonicalRegionWeights = Mapping[str, Mapping[str, Mapping[str, Tensor]]]


@dataclass(frozen=True)
class InferencePreparation:
    """A stateless operation and its derived tensor arguments.

    ``operation(inputs, tensors)`` returns a tuple in spec.output_names order.
    It declares SpanCapabilities: input derivatives are mandatory, weight
    gradients are False, and double_backward is explicit. All weights, tables
    and folded constants belong in ``tensors``, never hidden in the operation.
    Preparation may fold maps or reorder weights, preserving the declared
    precision, reduced basis, normalizations and function.

    Tensor contents and input tensors are read-only during execution. GPU ops
    use the calling framework's current stream and keep intermediates on device.
    Scratch allocation must be capture-safe; persistent scratch, if needed,
    belongs to a separate execution context, not mutable shared weight storage.
    Native registration and export support are separate deployment obligations.
    """

    operation: nn.Module
    tensors: tuple[Tensor, ...]


class InferenceBackend(Protocol):
    """Optional extension on an existing kernel backend, negotiated once.

    Absence of this method, or None for an unsupported shape, leaves the
    original operations in place. Existing KernelBackend implementers need
    no new methods. The backend must inspect every member's precision and
    layout; it may not silently promote or lower a policy to accept a region.
    """

    name: str

    def prepare_inference(
        self, spec: InferenceSpec, weights: CanonicalRegionWeights
    ) -> InferencePreparation | None: ...


class PreparedInference(nn.Module):
    """An immutable inference snapshot with differentiable runtime inputs.

    Call with a tensor tuple in spec.input_names order. No no_grad context is
    entered: forces and strain derivatives must reach the live inputs. Callers
    needing Hessians check ``capabilities.double_backward`` before selection.
    This object is separate from the source model; it is never its replacement
    for training or checkpoint I/O. torch.export can capture its derived
    buffers in a deployment artifact without changing a canonical checkpoint.
    """

    @torch.inference_mode(False)
    def __init__(self, spec: InferenceSpec, prepared: InferencePreparation) -> None:
        super().__init__()
        validate_span(prepared.operation, "prepared inference")
        self.capabilities = cast(SpanCapabilities, prepared.operation.capabilities)
        if self.capabilities.weight_gradients:
            raise ValueError("prepared inference must declare weight_gradients=False")
        if not isinstance(prepared.tensors, tuple) or not all(
            isinstance(tensor, Tensor) for tensor in prepared.tensors
        ):
            raise TypeError("prepared tensors must be a tuple of tensors")
        self.spec = spec
        self.operation = prepared.operation
        self._tensor_names = tuple(
            f"prepared_{index}" for index in range(len(prepared.tensors))
        )
        for name, tensor in zip(self._tensor_names, prepared.tensors, strict=True):
            # Own the snapshot even when a backend returned views into its inputs.
            self.register_buffer(name, tensor.detach().clone(), persistent=False)
        self.train(False)

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if len(inputs) != len(self.spec.input_names):
            raise ValueError(f"inference inputs must be {self.spec.input_names}")
        output = self.operation(
            inputs, tuple(getattr(self, name) for name in self._tensor_names)
        )
        if (
            not isinstance(output, tuple)
            or len(output) != len(self.spec.output_names)
            or not all(isinstance(value, Tensor) for value in output)
        ):
            raise ValueError(f"inference outputs must be {self.spec.output_names}")
        return output

    def train(self, mode: bool = True) -> PreparedInference:
        if mode:
            raise ValueError(
                "a prepared inference snapshot cannot train; use the source model"
            )
        super().train(False)
        return self

    def _apply(self, fn, recurse=True):
        raise ValueError("reprepare from canonical weights for a new device or dtype")

    def to_canonical(self) -> dict[str, Tensor]:
        raise CheckpointError(
            "save the source model's canonical checkpoint, not prepared inference"
        )


@torch.inference_mode(False)
def prepare_inference(
    backend: KernelBackend | InferenceBackend,
    spec: InferenceSpec,
    weights: CanonicalRegionWeights,
) -> PreparedInference | None:
    """Prepare once from raw canonical member tensors, without mutating them.

    Keys are exactly spec.weight_roles. Each value is that member's existing
    canonical_state() mapping, including buffers affecting its function. Product
    roles map directly to product components; interaction roles bind their
    named components. Normalization uses the interaction body's canonical
    state restricted to the root (the ``neighbours`` scalar). The radial basis
    uses its existing canonical components; its cutoff is also in the spec.

    Call after loading the checkpoint on the intended device and dtype.
    Missing support returns None. A malformed result or backend error is an
    error, never a silent fallback. This explicit API does not automatically
    replace modules in a model or enable an unvalidated native backend.
    """
    if not isinstance(spec, (ProductReadoutSpec, InteractionFusionSpec)):
        raise TypeError("unknown inference region specification")
    if set(weights) != set(spec.weight_roles):
        raise ValueError(f"canonical weight roles must be exactly {spec.weight_roles}")
    factory = getattr(backend, "prepare_inference", None)
    if factory is None:
        return None
    snapshot: dict[str, dict[str, dict[str, Tensor]]] = {}
    for role in spec.weight_roles:
        snapshot[role] = {}
        if not isinstance(weights[role], Mapping):
            raise TypeError(f"{role!r} must contain canonical_state()")
        for path, state in weights[role].items():
            if (
                not isinstance(path, str)
                or not isinstance(state, Mapping)
                or not all(
                    isinstance(name, str) and isinstance(tensor, Tensor)
                    for name, tensor in state.items()
                )
            ):
                raise TypeError(f"{role!r} must contain canonical_state()")
            snapshot[role][path] = {
                name: tensor.detach().clone() for name, tensor in state.items()
            }
    with torch.no_grad():
        prepared = factory(spec, snapshot)
    if prepared is None:
        return None
    if not isinstance(prepared, InferencePreparation):
        raise TypeError("prepare_inference must return InferencePreparation or None")
    return PreparedInference(spec, prepared)
