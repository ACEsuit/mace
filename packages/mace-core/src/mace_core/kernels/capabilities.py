"""What a backend can do, coarsely and then exactly.

Two levels on purpose. The coarse fields are a cheap first filter that answers
"could this backend plausibly serve this model" without building anything, and
:meth:`BackendCapabilities.supports` is the authoritative answer over one
complete descriptor. A backend that can do something only for certain shapes
overrides :meth:`BackendCapabilities.unsupported_reason`, which the second is
built on, and leaves the coarse fields generous.

A backend **declares** what it supports. It never chooses. The frozen tree gets
this backwards: a host without ``cuequivariance`` silently changes the
Clebsch-Gordan basis and trains a different network, which is a backend picking
model state. Here a mismatch between what a model asks for and what a backend
declares is an error with both sides named.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mace_core.kernels.descriptors import (
    Descriptor,
    RadialBasisDescriptor,
    SymmetricContractionDescriptor,
)

__all__ = ["BackendCapabilities", "UnsupportedDescriptorError"]


class UnsupportedDescriptorError(RuntimeError):
    """A backend was asked for an op it does not support."""


@dataclass(frozen=True)
class BackendCapabilities:
    """The coarse filter, plus the hook for the exact answer.

    Attributes:
        ops: Which factory names the backend implements, for example
            ``"linear"`` or ``"symmetric_contraction"``, matched against each
            descriptor's ``op``. The default is empty: a backend declares
            what it builds.
        devices: Device kinds, as names: ``"cpu"``, ``"cuda"``.
        dtypes: Precision names it computes in.
        max_lmax: The highest rotation order it can build, checked against
            every descriptor's declared irreps. Zero means it declares no
            limit.
        layouts: Which weight layouts it can consume. Every backend must accept
            ``"mul_ir"``, since that is the canonical one and the only one a
            checkpoint is written in.
        bases: Which Clebsch-Gordan bases it can consume.
        supports_double_backward: Whether its ops are differentiable twice.
            Training on forces or stress needs the second derivative, so a
            backend without it is usable for inference and rejected at build
            time for that training, loudly.
    """

    ops: frozenset[str] = field(default_factory=frozenset)
    devices: frozenset[str] = field(default_factory=lambda: frozenset({"cpu"}))
    dtypes: frozenset[str] = field(default_factory=lambda: frozenset({"float64"}))
    max_lmax: int = 0
    layouts: frozenset[str] = field(default_factory=lambda: frozenset({"mul_ir"}))
    bases: frozenset[str] = field(default_factory=lambda: frozenset({"reduced"}))
    supports_double_backward: bool = False

    def supports(self, descriptor: Descriptor) -> bool:
        """Whether this backend can build exactly this op.

        The authoritative answer, and the one a factory must agree with: a
        descriptor this returns ``True`` for is one the backend builds. It is
        :meth:`unsupported_reason` with the reason dropped.
        """
        return self.unsupported_reason(descriptor) is None

    def unsupported_reason(self, descriptor: Descriptor) -> str | None:
        """Why this backend cannot build ``descriptor``, or ``None`` if it can.

        The default checks every coarse field that applies to the descriptor:
        the op is declared, the precision is declared, the layout the op works
        in is declared, the declared irreps stay within ``max_lmax``, and a
        symmetric contraction's basis is declared. A backend whose limits
        depend on the shape overrides this, calls the parent first, and adds
        its own refusals, so that :meth:`supports` and the factory never
        disagree. Nothing else in the stack is allowed to guess on its behalf.
        """
        if descriptor.op not in self.ops:
            return (
                f"it does not declare the op {descriptor.op!r}; its ops are "
                f"{sorted(self.ops)}"
            )
        if descriptor.precision not in self.dtypes:
            return (
                f"it does not compute in {descriptor.precision!r}; its "
                f"precisions are {sorted(self.dtypes)}"
            )
        if descriptor.layout is not None and descriptor.layout not in self.layouts:
            return (
                f"it does not consume the {descriptor.layout!r} layout this op "
                f"works in; its layouts are {sorted(self.layouts)}"
            )
        if self.max_lmax and descriptor.highest_degree > self.max_lmax:
            return (
                f"the op reaches l = {descriptor.highest_degree}, beyond its "
                f"maximum lmax of {self.max_lmax}"
            )
        if (
            isinstance(descriptor, SymmetricContractionDescriptor)
            and descriptor.basis not in self.bases
        ):
            return (
                f"it does not consume the {descriptor.basis!r} Clebsch-Gordan "
                f"basis; its bases are {sorted(self.bases)}"
            )
        if isinstance(descriptor, RadialBasisDescriptor) and descriptor.num_basis < 1:
            return f"a radial basis of {descriptor.num_basis} functions is empty"
        return None

    def require(self, descriptor: Descriptor, backend_name: str) -> None:
        """Raise unless this backend can build ``descriptor``.

        Raises:
            UnsupportedDescriptorError: With the descriptor, the backend and
                the reason all named, because the usual cause is a model
                config asking for something the chosen backend declared it
                cannot do, and either side might be the one to change.
        """
        if self.supports(descriptor):
            return
        reason = (
            self.unsupported_reason(descriptor)
            or "its supports() declines this descriptor"
        )
        raise UnsupportedDescriptorError(
            f"backend {backend_name!r} does not support {descriptor!r}: "
            f"{reason}. Change the model configuration, or choose a "
            f"backend that declares it."
        )

    def require_double_backward(self, backend_name: str, why: str) -> None:
        """Raise unless this backend differentiates twice.

        Args:
            backend_name: Named in the message.
            why: What needs it, for example "training on forces". Named too,
                because the fix is usually to change that rather than the
                backend.

        Raises:
            UnsupportedDescriptorError: Loudly, at build time. This must never
                degrade into a warning: a first derivative computed through a
                backward that is not itself differentiable gives wrong forces
                rather than no forces.
        """
        if not self.supports_double_backward:
            raise UnsupportedDescriptorError(
                f"{why} needs a second derivative, and backend "
                f"{backend_name!r} declares it does not support double "
                f"backward. It can still be used for inference. Choose a "
                f"backend that does, or stop training on that quantity."
            )
