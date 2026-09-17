"""What a backend can do, coarsely and then exactly.

Two levels on purpose. The coarse fields are a cheap first filter that answers
"could this backend plausibly serve this model" without building anything, and
:meth:`BackendCapabilities.supports` is the authoritative answer over one
complete descriptor. A backend that can do something only for certain shapes
overrides the second and leaves the first generous.

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
    SphericalHarmonicsDescriptor,
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
            ``"linear"`` or ``"symmetric_contraction"``.
        devices: Device kinds, as names: ``"cpu"``, ``"cuda"``.
        dtypes: Precision names it computes in.
        max_lmax: The highest rotation order it can build. Zero means it
            declares no limit.
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

        The default answer checks the coarse fields against the descriptor. A
        backend whose limits depend on the shape overrides this; nothing else
        in the stack is allowed to guess on its behalf.
        """
        if descriptor.precision not in self.dtypes:
            return False
        if (
            isinstance(descriptor, SymmetricContractionDescriptor)
            and descriptor.basis not in self.bases
        ):
            return False
        if (
            self.max_lmax
            and isinstance(descriptor, SphericalHarmonicsDescriptor)
            and descriptor.lmax > self.max_lmax
        ):
            return False
        return not (
            isinstance(descriptor, RadialBasisDescriptor) and descriptor.num_basis < 1
        )

    def require(self, descriptor: Descriptor, backend_name: str) -> None:
        """Raise unless this backend can build ``descriptor``.

        Raises:
            UnsupportedDescriptorError: With the descriptor and the backend both
                named, because the usual cause is a model config asking for
                something the chosen backend declared it cannot do, and either
                side might be the one to change.
        """
        if not self.supports(descriptor):
            raise UnsupportedDescriptorError(
                f"backend {backend_name!r} does not support {descriptor!r}. Its "
                f"declared precisions are {sorted(self.dtypes)}, bases "
                f"{sorted(self.bases)}, layouts {sorted(self.layouts)} and "
                f"maximum lmax {self.max_lmax or 'unbounded'}."
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
