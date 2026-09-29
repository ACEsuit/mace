"""A backend whose ops are built at the precision the configuration gives each.

The model's builders ask the backend for ops and never choose a dtype. This
wrapper stands between them: every descriptor is stamped with its
:class:`~mace_core.config.precision.PrecisionPolicy` before the wrapped backend
sees it, and an op that computes in something other than the interface dtype is
given casts at its boundary. Everything is decided here, once, when the model is
built, so ``forward`` carries no dtype logic and a compiled graph sees the same
casts on every call.

A uniform configuration, every op in the interface dtype, wraps nothing: the
ops are the wrapped backend's own, unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from mace_core.config.precision import PrecisionConfig, PrecisionPolicy
from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

__all__ = ["PrecisionBackend", "boundary_precision", "cast_at_boundary"]

_TORCH_DTYPE = {
    "float64": torch.float64,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

#: The factories whose ops this wrapper stamps. Anything else the wrapped
#: backend offers is passed through as it is.
_FACTORIES = (
    "make_linear",
    "make_channelwise_tp_conv",
    "make_symmetric_contraction",
    "make_fully_connected_tp",
    "make_segment_reduce",
    "make_spherical_harmonics",
    "make_radial_basis",
)


class PrecisionBackend:
    """``backend``, with every op built at the precision ``config`` gives it.

    Args:
        backend: The backend that builds the ops.
        config: The precision of every op kind, and of the interface.
        supports_float64: Whether the device has float64. A float64 floor it
            cannot meet degrades to the interface dtype, and
            :attr:`degraded` names what was lowered.
    """

    def __init__(
        self, backend: Any, config: PrecisionConfig, supports_float64: bool = True
    ) -> None:
        self.backend = backend
        self.config, self.degraded = config.for_device(supports_float64)
        self.name: str = backend.name
        self._interface = config.interface_dtype
        for factory in _FACTORIES:
            if hasattr(backend, factory):
                setattr(self, factory, self._stamping(factory))

    def __getattr__(self, name: str) -> Any:
        # Only reached for what __init__ did not set: the wrapped backend's
        # other factories and attributes, such as a solver span.
        return getattr(self.backend, name)

    def capabilities(self) -> BackendCapabilities:
        return self.backend.capabilities()

    def build_report(self) -> str | None:
        """What was lowered at build time, if anything."""
        if not self.degraded:
            return None
        return (
            f"this device has no float64, so these run in {self._interface} "
            f"rather than float64: {', '.join(self.degraded)}."
        )

    def _stamping(self, factory: str) -> Any:
        make = getattr(self.backend, factory)

        def build(descriptor: Any) -> Any:
            stamped = self._stamp(descriptor)
            op = make(stamped)
            if op is not None and stamped.precision != self._interface:
                cast_at_boundary(op, stamped.precision, self._interface)
            return op

        return build

    def make_interaction_layer(self, descriptors: tuple[Any, ...]) -> Any | None:
        """A fused span, if the wrapped backend has one, at one policy.

        A fused kernel has one math dtype, so every member is stamped with the
        span's policy: the widest of its members', which meets each floor.
        """
        make = getattr(self.backend, "make_interaction_layer", None)
        if make is None:
            return None
        policy = self.config.policy_for(descriptors)
        stamped = tuple(self._stamp(member, policy) for member in descriptors)
        span = make(stamped)
        if span is not None and stamped[0].precision != self._interface:
            cast_at_boundary(span, stamped[0].precision, self._interface)
        return span

    def _stamp(self, descriptor: Any, policy: PrecisionPolicy | None = None) -> Any:
        if policy is None:
            policy = self.config.policy_for(descriptor)
        if policy.param_dtype != policy.compute_dtype:
            raise ValueError(
                f"the {type(descriptor).__name__} policy holds its weights in "
                f"{policy.param_dtype} and computes in {policy.compute_dtype}. "
                f"The {self.name!r} backend holds an op's weights in the dtype "
                f"it computes in; set param_dtype equal to compute_dtype."
            )
        if policy.accumulate_dtype == policy.compute_dtype:
            return dataclasses.replace(
                descriptor, precision=policy.compute_dtype, accumulate=None
            )
        if self.capabilities().wide_accumulation:
            return dataclasses.replace(
                descriptor,
                precision=policy.compute_dtype,
                accumulate=policy.accumulate_dtype,
            )
        # A plain-torch op has no accumulator of its own, so it meets a floor
        # wider than its compute dtype by computing in the floor.
        return dataclasses.replace(
            descriptor, precision=policy.accumulate_dtype, accumulate=None
        )


def cast_at_boundary(op: nn.Module, compute: Precision, interface: Precision) -> None:
    """Make ``op`` take its floating inputs in ``compute`` and hand its outputs
    back in ``interface``.

    Done with hooks on the op itself rather than a module around it, so the op
    keeps its place in the model: its parameters keep their names in a
    checkpoint, and it draws the same initial weights whatever the precision.
    Integer tensors, the edge and element indices, cross unchanged.
    """
    op.register_forward_pre_hook(_CastInputs(compute), with_kwargs=True)
    op.register_forward_hook(_CastOutput(interface))


def boundary_precision(op: nn.Module) -> tuple[Precision, Precision] | None:
    """``(compute, interface)`` if ``op`` casts at its boundary, else ``None``."""
    inputs = [h for h in op._forward_pre_hooks.values() if isinstance(h, _CastInputs)]
    output = [h for h in op._forward_hooks.values() if isinstance(h, _CastOutput)]
    if not inputs:
        return None
    return inputs[0].precision, output[0].precision


class _CastInputs:
    def __init__(self, precision: Precision) -> None:
        self.precision = precision
        self.dtype = _TORCH_DTYPE[precision]

    def __call__(self, module: nn.Module, args: tuple, kwargs: dict) -> Any:
        return _cast(args, self.dtype), {
            key: _cast(value, self.dtype) for key, value in kwargs.items()
        }


class _CastOutput:
    def __init__(self, precision: Precision) -> None:
        self.precision = precision
        self.dtype = _TORCH_DTYPE[precision]

    def __call__(self, module: nn.Module, args: tuple, output: Any) -> Any:
        return _cast(output, self.dtype)


def _cast(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, Tensor):
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, tuple | list):
        return type(value)(_cast(item, dtype) for item in value)
    return value
