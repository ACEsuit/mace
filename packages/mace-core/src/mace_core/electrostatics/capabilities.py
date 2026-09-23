"""What an electrostatics solver can do, coarsely and then exactly.

The same two levels as a kernel backend's: coarse fields as a cheap first
filter, and :meth:`SolverCapabilities.supports` as the authoritative answer
over one complete descriptor. A solver declares; it never chooses, and a
system it declines is an error with both sides named, never a quiet switch to
another solver.

**Bit parity decides whether a solver is a choice or model state.** A solver
that reproduces the reference bit for bit can be swapped at load time without
changing a number. One that does not changes the model's numbers, so its
identity travels with the checkpoint and it is never substituted, in either
direction, without the model asking for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mace_core.electrostatics.descriptor import (
    PERIODICITY_PROFILES,
    ElectrostaticsSolverDescriptor,
)

__all__ = ["SolverCapabilities", "UnsupportedSolveError"]


class UnsupportedSolveError(RuntimeError):
    """A solver was asked for a solve it declared it cannot do."""


@dataclass(frozen=True)
class SolverCapabilities:
    """The coarse filter, plus the hook for the exact answer.

    Attributes:
        ops: Which ops the solver implements. ``long_range_energy`` is the
            base one; ``scf_solve`` is the optional fused span.
        devices: Device kinds, as names.
        dtypes: Precision names it computes in.
        periodicity_profiles: The profiles it can solve.
        max_multipole_l: The highest multipole order it takes. ``None`` is no
            limit.
        slab_normals: The axes a slab correction can act along.
        field_flags: The applied-field and self-interaction options it
            understands. A flag it does not is a solve it cannot do, not one
            it can ignore.
        supports_double_backward: Whether its solve is differentiable twice,
            which training on forces or stress needs.
        bit_parity: Whether it reproduces the reference solver bit for bit.
    """

    ops: frozenset[str] = field(
        default_factory=lambda: frozenset({"long_range_energy"})
    )
    devices: frozenset[str] = field(default_factory=lambda: frozenset({"cpu"}))
    dtypes: frozenset[str] = field(default_factory=lambda: frozenset({"float64"}))
    periodicity_profiles: frozenset[str] = field(
        default_factory=lambda: frozenset(PERIODICITY_PROFILES)
    )
    max_multipole_l: int | None = None
    slab_normals: frozenset[int] = field(default_factory=lambda: frozenset({0, 1, 2}))
    field_flags: frozenset[str] = field(default_factory=frozenset)
    supports_double_backward: bool = False
    bit_parity: bool = False

    def supports(self, descriptor: ElectrostaticsSolverDescriptor) -> bool:
        """Whether this solver can build exactly this solve."""
        if "long_range_energy" not in self.ops:
            return False
        if descriptor.precision not in self.dtypes:
            return False
        if descriptor.periodicity_profile not in self.periodicity_profiles:
            return False
        if (
            descriptor.slab_normal is not None
            and descriptor.slab_normal not in self.slab_normals
        ):
            return False
        if not descriptor.external_field_flags <= self.field_flags:
            return False
        return self.max_multipole_l is None or (
            descriptor.multipole_max_l <= self.max_multipole_l
        )

    def require(self, descriptor: ElectrostaticsSolverDescriptor, solver: str) -> None:
        """Raise unless this solver can build ``descriptor``.

        Raises:
            UnsupportedSolveError: Naming the solver, the solve, and what the
                solver declared. Never a fallback: another solver is another
                set of numbers unless it is bit for bit the same.
        """
        if not self.supports(descriptor):
            limit = "no limit" if self.max_multipole_l is None else self.max_multipole_l
            raise UnsupportedSolveError(
                f"the electrostatics solver {solver!r} does not support "
                f"{descriptor!r}. It declares the profiles "
                f"{sorted(self.periodicity_profiles)}, the slab normals "
                f"{sorted(self.slab_normals)}, the field options "
                f"{sorted(self.field_flags)}, the precisions "
                f"{sorted(self.dtypes)} and a highest multipole order of "
                f"{limit}. "
                f"It is not replaced by another solver, because that would "
                f"change the model's numbers."
            )

    def require_double_backward(self, solver: str, why: str) -> None:
        """Raise unless this solver differentiates twice.

        Raises:
            UnsupportedSolveError: Loudly, at build time. A force taken through
                a solve whose backward is not itself differentiable trains on
                wrong gradients rather than failing.
        """
        if not self.supports_double_backward:
            raise UnsupportedSolveError(
                f"{why} needs the electrostatics solve differentiated twice, "
                f"and the solver {solver!r} declares it cannot be. It can run "
                f"inference; train with a solver that supports it."
            )
