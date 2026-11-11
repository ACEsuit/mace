"""What a backend can do, coarsely and then exactly.

Two levels on purpose. The coarse fields are a cheap first filter that answers
"could this backend plausibly serve this model" without building anything, and
:meth:`BackendCapabilities.supports` is the authoritative answer over one
complete kernel specification. A backend that can do something only for certain shapes
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

from mace_core.kernels.specs import (
    KernelSpec,
    RadialBasisSpec,
    SymmetricContractionSpec,
)

__all__ = ["BackendCapabilities", "UnsupportedKernelSpecError"]


class UnsupportedKernelSpecError(RuntimeError):
    """A backend was asked for an op it does not support."""


@dataclass(frozen=True)
class BackendCapabilities:
    """The coarse filter, plus the hook for the exact answer.

    Attributes:
        ops: Which factory names the backend implements, for example
            ``"linear"`` or ``"symmetric_contraction"``, matched against each
            kernel specification's ``op``. The default is empty: a backend declares
            what it builds.
        devices: Device kinds, as names: ``"cpu"``, ``"cuda"``.
        dtypes: Precision names it computes in.
        max_lmax: The highest rotation order it can build, checked against
            every kernel specification's declared irreps. Zero means it declares no
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

    def supports(self, kernel_spec: KernelSpec) -> bool:
        """Whether this backend can build exactly this op.

        The authoritative answer, and the one a factory must agree with: a
        kernel specification this returns ``True`` for is one the backend builds. It is
        :meth:`unsupported_reason` with the reason dropped.
        """
        return self.unsupported_reason(kernel_spec) is None

    def unsupported_reason(self, kernel_spec: KernelSpec) -> str | None:
        """Why this backend cannot build ``kernel_spec``, or ``None`` if it can.

        The default checks every coarse field that applies to the kernel specification:
        the op is declared, the precision is declared, the layout the op works
        in is declared, the declared irreps stay within ``max_lmax``, and a
        symmetric contraction's basis is declared. A backend whose limits
        depend on the shape overrides this, calls the parent first, and adds
        its own refusals, so that :meth:`supports` and the factory never
        disagree. Nothing else in the stack is allowed to guess on its behalf.
        """
        if kernel_spec.op not in self.ops:
            return (
                f"it does not declare the op {kernel_spec.op!r}; its ops are "
                f"{sorted(self.ops)}"
            )
        if kernel_spec.precision not in self.dtypes:
            return (
                f"it does not compute in {kernel_spec.precision!r}; its "
                f"precisions are {sorted(self.dtypes)}"
            )
        if kernel_spec.layout is not None and kernel_spec.layout not in self.layouts:
            return (
                f"it does not consume the {kernel_spec.layout!r} layout this op "
                f"works in; its layouts are {sorted(self.layouts)}"
            )
        if self.max_lmax and kernel_spec.highest_degree > self.max_lmax:
            return (
                f"the op reaches l = {kernel_spec.highest_degree}, beyond its "
                f"maximum lmax of {self.max_lmax}"
            )
        if (
            isinstance(kernel_spec, SymmetricContractionSpec)
            and kernel_spec.basis not in self.bases
        ):
            return (
                f"it does not consume the {kernel_spec.basis!r} Clebsch-Gordan "
                f"basis; its bases are {sorted(self.bases)}"
            )
        if isinstance(kernel_spec, RadialBasisSpec):
            if kernel_spec.num_basis < 1:
                return f"a radial basis of {kernel_spec.num_basis} functions is empty"
            if kernel_spec.kind == "gaussian" and kernel_spec.num_basis < 2:
                return (
                    "a gaussian basis needs at least two functions: its width "
                    "is the spacing between centres, cutoff / (num_basis - 1), "
                    "and one function has no spacing"
                )
        return None

    def require(self, kernel_spec: KernelSpec, backend_name: str) -> None:
        """Raise unless this backend can build ``kernel_spec``.

        Raises:
            UnsupportedKernelSpecError: With the kernel specification, the backend and
                the reason all named, because the usual cause is a model
                config asking for something the chosen backend declared it
                cannot do, and either side might be the one to change.
        """
        if self.supports(kernel_spec):
            return
        reason = (
            self.unsupported_reason(kernel_spec)
            or "its supports() declines this kernel specification"
        )
        raise UnsupportedKernelSpecError(
            f"backend {backend_name!r} does not support {kernel_spec!r}: "
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
            UnsupportedKernelSpecError: Loudly, at build time. This must never
                degrade into a warning: a first derivative computed through a
                backward that is not itself differentiable gives wrong forces
                rather than no forces.
        """
        if not self.supports_double_backward:
            raise UnsupportedKernelSpecError(
                f"{why} needs a second derivative, and backend "
                f"{backend_name!r} declares it does not support double "
                f"backward. It can still be used for inference. Choose a "
                f"backend that does, or stop training on that quantity."
            )
