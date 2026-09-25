"""An accelerated backend with the reference behind it, op by op.

A model is built against one backend name. When that backend is an
accelerated one, it rarely builds every op the model asks for: oeq builds only
the convolution, and cueq declines the shapes its kernels get wrong. Each op
the chosen backend does not build is built by the reference instead, decided
here, once, at build time, from the backend's own answer to
``supports(descriptor)``. Nothing is tried and caught: the frozen tree's
convert, fail and reconvert (``mace/calculators/mace_torchsim.py:123-137``) has
no counterpart.

Every op takes and returns the same layout, grouped by irrep with the channels
inside, whichever backend built it. A backend arranges its values however its
kernel wants inside the op. So a fallback puts no permute between two ops, and
there is no layout seam to count.

What was built where is recorded in :attr:`CompositeBackend.decisions` and
summarised by :meth:`CompositeBackend.report`, which the model stage logs. An
op the chosen backend declined is logged as a warning when it is decided, so
that a run on an accelerated backend that computes most of its layers on the
reference says so.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.protocol import REFERENCE_ONLY_OPS
from mace_core.kernels.registry import get_backend

from mace_torch.backends.reference import ReferenceBackend

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
        descriptor: What was asked for.
        backend: The name of the backend that built it.
        reason: Empty when the chosen backend built it; otherwise why not.
    """

    op: str
    descriptor: Any
    backend: str
    reason: str = ""


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
            bases=fallback.bases,
            supports_double_backward=primary.supports_double_backward
            and fallback.supports_double_backward,
        )

    def _build(self, op: str, descriptor: Any) -> Any:
        capabilities = self._primary_capabilities
        if op not in capabilities.ops:
            reason = f"{self.primary.name} does not implement {op}"
        elif not capabilities.supports(descriptor):
            reason = f"{self.primary.name} declines this descriptor"
        else:
            built = getattr(self.primary, f"make_{op}")(descriptor)
            if built is not None:
                self.decisions.append(BuildDecision(op, descriptor, self.primary.name))
                return built
            reason = f"{self.primary.name} leaves {op} to the reference"
        self.decisions.append(BuildDecision(op, descriptor, self.fallback.name, reason))
        if op not in REFERENCE_ONLY_OPS and op in capabilities.ops:
            logger.warning("%s is built by the reference: %s", descriptor, reason)
        return getattr(self.fallback, f"make_{op}")(descriptor)

    def make_linear(self, descriptor: Any) -> Any:
        return self._build("linear", descriptor)

    def make_channelwise_tp_conv(self, descriptor: Any) -> Any:
        return self._build("channelwise_tp_conv", descriptor)

    def make_symmetric_contraction(self, descriptor: Any) -> Any:
        return self._build("symmetric_contraction", descriptor)

    def make_fully_connected_tp(self, descriptor: Any) -> Any:
        return self._build("fully_connected_tp", descriptor)

    def make_segment_reduce(self, descriptor: Any) -> Any:
        return self._build("segment_reduce", descriptor)

    def make_spherical_harmonics(self, descriptor: Any) -> Any:
        return self._build("spherical_harmonics", descriptor)

    def make_radial_basis(self, descriptor: Any) -> Any:
        return self._build("radial_basis", descriptor)

    def make_interaction_layer(self, descriptors: tuple[Any, ...]) -> Any | None:
        return self.primary.make_interaction_layer(descriptors)

    def served_by_primary(self) -> bool:
        """Whether the chosen backend built any op at all."""
        return any(d.backend == self.primary.name for d in self.decisions)

    def report(self) -> str:
        """One line per op kind: how many each backend built, and why not."""
        counts = Counter((d.op, d.backend) for d in self.decisions)
        reasons: dict[str, set[str]] = {}
        for decision in self.decisions:
            if decision.reason:
                reasons.setdefault(decision.op, set()).add(decision.reason)
        lines = [
            f"backend {self.primary.name}, with {self.fallback.name} for the ops "
            f"it does not build; one layout throughout, so no seams"
        ]
        for op in sorted({d.op for d in self.decisions}):
            built = ", ".join(
                f"{counts[op, name]} by {name}"
                for name in (self.primary.name, self.fallback.name)
                if counts[op, name]
            )
            why = "; ".join(sorted(reasons.get(op, ())))
            lines.append(f"  {op}: {built}" + (f" ({why})" if why else ""))
        return "\n".join(lines)


def resolve_backend(name: str) -> Any:
    """The backend a model named ``name`` is built with.

    The reference on its own, or the named backend with the reference behind
    it for every op it does not build. Resolved once, at build time.

    Raises:
        BackendNotAvailableError: The name is not registered, or it is and did
            not import on this machine. Nothing is substituted for it.
    """
    backend = get_backend(name)
    if isinstance(backend, ReferenceBackend):
        return backend
    return CompositeBackend(backend, ReferenceBackend())


def require_double_backward(name: str, why: str) -> None:
    """Refuse a backend that cannot be differentiated twice, for ``why``.

    Raises:
        UnsupportedDescriptorError: Loudly, naming the backend and the reason
            it is needed. The backend stays usable for inference.
    """
    get_backend(name).capabilities().require_double_backward(name, why)
