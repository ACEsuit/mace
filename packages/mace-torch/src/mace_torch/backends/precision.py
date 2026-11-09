"""A backend whose ops are built at the precision the configuration gives each.

The model's builders ask the backend for ops and never choose a dtype. This
wrapper stands between them: every kernel_spec is stamped with its
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
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from mace_core.config.precision import OP_KINDS, PrecisionConfig, PrecisionPolicy
from mace_core.kernels.canonical import CANONICAL_LAYOUT
from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

from mace_torch.nn.edges import EdgeInputs

__all__ = [
    "PrecisionBackend",
    "boundary_precision",
    "cast_at_boundary",
    "constructed_in",
    "degradation_report",
]

_TORCH_DTYPE = {
    "float64": torch.float64,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
logger = logging.getLogger(__name__)

#: The factories whose ops this wrapper stamps. Anything else the wrapped
#: backend offers is passed through as it is.
_FACTORIES = tuple(f"make_{kind}" for kind in OP_KINDS)


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
        self.config_requested = config
        self.supports_float64 = supports_float64
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
        reports = [degradation_report(self.config_requested, self.supports_float64)]
        products = [item for item in self.degraded if item.endswith(" products")]
        if products:
            reports.append(
                f"backend {self.name!r} uses IEEE for {', '.join(products)}."
            )
        return " ".join(report for report in reports if report) or None

    def _stamping(self, factory: str) -> Any:
        make = getattr(self.backend, factory)

        def build(kernel_spec: Any) -> Any:
            stamped = self.op_spec(kernel_spec)
            op = make(stamped)
            if op is not None and stamped.precision != self._interface:
                cast_at_boundary(op, stamped.precision, self._interface)
            return op

        return build

    def make_interaction_layer(self, kernel_specs: tuple[Any, ...]) -> Any | None:
        """A fused span, if the wrapped backend has one, at one policy.

        A fused kernel has one math dtype, so every member is stamped with the
        span's policy: the widest of its members', which meets each floor.
        """
        if not kernel_specs:
            return None
        stamped = self.span_specs(kernel_specs)
        span = self.backend.make_interaction_layer(stamped)
        if span is not None and stamped[0].precision != self._interface:
            cast_at_boundary(span, stamped[0].precision, self._interface)
        return span

    def span_specs(self, kernel_specs: tuple[Any, ...]) -> tuple[Any, ...]:
        """Stamp one policy before compatibility is checked or the span is built."""
        if not kernel_specs:
            return ()
        policy = self.config.policy_for(kernel_specs)
        return tuple(
            self.op_spec(
                dataclasses.replace(
                    member,
                    activation_layout=member.activation_layout or CANONICAL_LAYOUT,
                ),
                policy,
            )
            for member in kernel_specs
        )

    def op_spec(self, kernel_spec: Any, policy: PrecisionPolicy | None = None) -> Any:
        """The effective specification, including recorded precision degradations."""
        if policy is None:
            policy = self.config.policy_for(kernel_spec)
        capabilities = self.capabilities()
        compute = policy.compute_dtype
        accumulate = None
        if policy.accumulate_dtype != compute:
            if capabilities.wide_accumulation:
                accumulate = policy.accumulate_dtype
            else:
                compute = policy.accumulate_dtype
        product = policy.product
        if product == "tf32" and (
            product not in capabilities.products or compute != "float32"
        ):
            what = f"{type(kernel_spec).__name__} tf32 products"
            if what not in self.degraded:
                self.degraded.append(what)
                logger.warning(
                    "backend %r uses IEEE products for %s at %s precision",
                    self.name,
                    type(kernel_spec).__name__,
                    compute,
                )
            product = "ieee"
        return dataclasses.replace(
            kernel_spec, precision=compute, accumulate=accumulate, product=product
        )


@contextmanager
def constructed_in(precision: Precision) -> Iterator[None]:
    """Initialize constants at their intended precision, restoring the default.

    Constructing in float32 and widening afterwards loses digits permanently.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(_TORCH_DTYPE[precision])
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def degradation_report(config: PrecisionConfig, supports_float64: bool) -> str | None:
    """What a device without float64 lowers in ``config``, if anything."""
    _, degraded = config.for_device(supports_float64)
    if not degraded:
        return None
    return (
        f"this device has no float64, so these run in {config.interface_dtype} "
        f"rather than float64: {', '.join(degraded)}."
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
    if isinstance(value, EdgeInputs):
        return value.to(dtype)
    if isinstance(value, Tensor):
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, tuple | list):
        return type(value)(_cast(item, dtype) for item in value)
    return value
