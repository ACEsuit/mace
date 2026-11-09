"""An accelerated backend with the reference behind it, op by op.

A model is built against one backend name. When that backend is an
accelerated one, it rarely builds every op the model asks for: oeq builds only
the convolution, and cueq declines the shapes its kernels get wrong. Each op
the chosen backend does not build is built by the reference instead, decided
here, once, at build time, from the backend's own answer to
``supports(kernel_spec)``. Nothing is tried and caught: the frozen tree's
convert, fail and reconvert (``mace/calculators/mace_torchsim.py:123-137``) has
no counterpart.

Every op reads and writes the negotiated activation layout. The reference
follows the primary backend's native layout throughout the chain.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

from mace_core.config.precision import PrecisionConfig
from mace_core.kernels.canonical import CANONICAL_LAYOUT, KERNEL_SPEC_VERSION
from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.protocol import REFERENCE_ONLY_OPS
from mace_core.kernels.registry import get_backend

from mace_torch.backends.layout import CANONICAL, Layout
from mace_torch.backends.precision import PrecisionBackend
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.backends.spans import validate_span

__all__ = [
    "BuildDecision",
    "CompositeBackend",
    "require_double_backward",
    "resolve_backend",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildDecision:
    """Which backend built one op, and why when it was not the chosen one.

    Attributes:
        op: The factory name, such as ``"channelwise_tp_conv"``.
        kernel_spec: What was asked for.
        backend: The name of the backend that built it.
        reason: Empty when the chosen backend built it; otherwise why not.
    """

    op: str
    kernel_spec: Any
    backend: str
    reason: str = ""
    layer: int | None = None


class CompositeBackend:
    """The chosen backend where it builds an op, the reference where it does not.

    Args:
        primary: The backend the configuration names.
        fallback: The reference backend.
    """

    def __init__(self, primary: Any, fallback: Any) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self._primary_capabilities: BackendCapabilities = primary.capabilities()
        self.decisions: list[BuildDecision] = []
        self._span_requests = 0
        native = self._primary_capabilities.native_layout
        self.layout = (
            Layout(native)
            if native in fallback.capabilities().activation_layouts
            else CANONICAL
        )

    def capabilities(self) -> BackendCapabilities:
        """What the pair can build: everything the reference can.

        Double backward is the chosen backend's answer, since any op it builds
        is on the path a force is differentiated through.
        """
        fallback = self.fallback.capabilities()
        primary = self._primary_capabilities
        return BackendCapabilities(
            ops=primary.ops | fallback.ops,
            devices=fallback.devices,
            dtypes=fallback.dtypes,
            max_lmax=fallback.max_lmax,
            layouts=fallback.layouts,
            activation_layouts=frozenset({self.layout.name}),
            native_layout=self.layout.name,
            products=primary.products | fallback.products,
            bases=fallback.bases,
            supports_double_backward=primary.supports_double_backward
            and fallback.supports_double_backward,
            edge_features=primary.edge_features | fallback.edge_features,
            spec_version=KERNEL_SPEC_VERSION,
        )

    def _build(self, op: str, kernel_spec: Any) -> Any:
        kernel_spec = replace(kernel_spec, layout=self.layout.name)
        capabilities = self._primary_capabilities
        effective = (
            self.primary.op_spec(kernel_spec)
            if op in capabilities.ops and isinstance(self.primary, PrecisionBackend)
            else kernel_spec
        )
        if op not in capabilities.ops:
            reason = f"{self.primary.name} does not implement {op}"
        elif not capabilities.supports(effective):
            reason = f"{self.primary.name} declines this kernel specification"
        else:
            built = getattr(self.primary, f"make_{op}")(kernel_spec)
            if built is not None:
                self.decisions.append(BuildDecision(op, effective, self.primary.name))
                return built
            reason = f"{self.primary.name} leaves {op} to the reference"
        fallback_spec = (
            self.fallback.op_spec(kernel_spec)
            if isinstance(self.fallback, PrecisionBackend)
            else kernel_spec
        )
        self.decisions.append(
            BuildDecision(op, fallback_spec, self.fallback.name, reason)
        )
        if op not in REFERENCE_ONLY_OPS and op in capabilities.ops:
            logger.warning("%s is built by the reference: %s", kernel_spec, reason)
        return getattr(self.fallback, f"make_{op}")(kernel_spec)

    def make_radial_mlp(self, kernel_spec: Any) -> Any:
        return self._build("radial_mlp", kernel_spec)

    def make_distance_transform(self, kernel_spec: Any) -> Any:
        return self._build("distance_transform", kernel_spec)

    def make_linear(self, kernel_spec: Any) -> Any:
        return self._build("linear", kernel_spec)

    def make_channelwise_tp_conv(self, kernel_spec: Any) -> Any:
        return self._build("channelwise_tp_conv", kernel_spec)

    def make_symmetric_contraction(self, kernel_spec: Any) -> Any:
        return self._build("symmetric_contraction", kernel_spec)

    def make_fully_connected_tp(self, kernel_spec: Any) -> Any:
        return self._build("fully_connected_tp", kernel_spec)

    def make_segment_reduce(self, kernel_spec: Any) -> Any:
        return self._build("segment_reduce", kernel_spec)

    def make_spherical_harmonics(self, kernel_spec: Any) -> Any:
        return self._build("spherical_harmonics", kernel_spec)

    def make_radial_basis(self, kernel_spec: Any) -> Any:
        return self._build("radial_basis", kernel_spec)

    def make_interaction_layer(self, kernel_specs: tuple[Any, ...]) -> Any | None:
        """One layout for the span; precision is stamped by PrecisionBackend."""
        members = tuple(
            replace(member, layout=self.layout.name) for member in kernel_specs
        )
        if isinstance(self.primary, PrecisionBackend):
            members = self.primary.span_specs(members)
        layer = self._span_requests
        self._span_requests += 1
        capabilities = self._primary_capabilities
        span = None
        if not all(capabilities.supports(member) for member in members):
            reason = "a span member is not supported"
        else:
            span = self.primary.make_interaction_layer(members)
            reason = "backend declined this span shape" if span is None else ""
        if span is not None:
            validate_span(span, self.primary.name)
        self.decisions.append(
            BuildDecision(
                "interaction_layer",
                members,
                self.primary.name if span is not None else self.fallback.name,
                reason,
                layer,
            )
        )
        return span

    def decline_interaction_layer(self, reason: str) -> None:
        """Record a layer whose members cannot form a described span."""
        self.decisions.append(
            BuildDecision(
                "interaction_layer", (), self.fallback.name, reason, self._span_requests
            )
        )
        self._span_requests += 1

    def served_by_primary(self) -> bool:
        """Whether the chosen backend built any op at all."""
        return any(d.backend == self.primary.name for d in self.decisions)

    def report(self) -> str:
        """One line per op kind: how many each backend built, and why not."""
        ops = [d for d in self.decisions if d.layer is None]
        counts = Counter((d.op, d.backend) for d in ops)
        reasons: dict[str, set[str]] = {}
        for decision in self.decisions:
            if decision.reason:
                reasons.setdefault(decision.op, set()).add(decision.reason)
        lines = [
            f"backend {self.primary.name}, with {self.fallback.name} for the ops "
            f"it does not build. Every op uses {self.layout.name} activations"
        ]
        for op in sorted({d.op for d in ops}):
            built = ", ".join(
                f"{counts[op, name]} by {name}"
                for name in (self.primary.name, self.fallback.name)
                if counts[op, name]
            )
            why = "; ".join(sorted(reasons.get(op, ())))
            lines.append(f"  {op}: {built}" + (f" ({why})" if why else ""))
        for decision in self.decisions:
            if decision.layer is not None:
                lines.append(
                    f"  layer {decision.layer} span: {decision.backend}"
                    + (f"; op by op ({decision.reason})" if decision.reason else "")
                )
        for member in (self.primary, self.fallback):
            if isinstance(member, PrecisionBackend) and (
                report := member.build_report()
            ):
                lines.append(report)
        return "\n".join(lines)


def resolve_backend(
    name: str,
    precision: PrecisionConfig | None = None,
    supports_float64: bool = True,
) -> Any:
    """The backend a model named ``name`` is built with.

    The reference on its own, or the named backend with the reference behind
    it for every op it does not build. Resolved once, at build time.

    With a precision, each of the two builds its ops at the precision it is
    given, and meets a wider accumulation floor its own way, so an op the
    reference builds for the chosen backend meets the floor as well.

    Raises:
        BackendNotAvailableError: The name is not registered, or it is and did
            not import on this machine. Nothing is substituted for it.
    """
    backend = get_backend(name)

    def at_precision(member: Any) -> Any:
        if precision is None:
            return member
        return PrecisionBackend(member, precision, supports_float64)

    if (
        isinstance(backend, ReferenceBackend)
        and backend.capabilities().native_layout == CANONICAL_LAYOUT
    ):
        return at_precision(backend)
    return CompositeBackend(at_precision(backend), at_precision(ReferenceBackend()))


def require_double_backward(name: str, why: str) -> None:
    """Refuse a backend that cannot be differentiated twice, for ``why``.

    Raises:
        UnsupportedKernelSpecError: Loudly, naming the backend and the reason
            it is needed. The backend stays usable for inference.
    """
    get_backend(name).capabilities().require_double_backward(name, why)
