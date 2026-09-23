"""What a long-range electrostatics solve is, before any solver agrees to it.

A descriptor is the complete statement of one solve: enough for a solver to
answer whether it can do it, and enough to build it. It carries numbers and
names only, so it is hashable, comparable and can be written into a
checkpoint. Dtypes are names, as everywhere in this package.

The solver is resolved from it once, when the model is built. Nothing resolves
in ``forward``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

from mace_core.kernels.precision import Precision

__all__ = [
    "PERIODICITY_PROFILES",
    "ElectrostaticsSolverDescriptor",
    "PeriodicityProfile",
    "ScfSpec",
]

#: Which systems a solve is set up for.
#:
#: ``full_periodic`` is a bulk crystal, periodic along all three axes.
#: ``z_slab`` is periodic along two axes with vacuum along the slab normal, and
#: takes the slab dipole correction. ``molecular`` is periodic along none: the
#: system sits in the box the graph builder made around it. ``partial`` is a
#: batch that mixes those, each structure handled by its own periodicity.
PeriodicityProfile = Literal["full_periodic", "z_slab", "molecular", "partial"]

PERIODICITY_PROFILES: tuple[str, ...] = get_args(PeriodicityProfile)


@dataclass(frozen=True)
class ScfSpec:
    """The self-consistent loop a charge-aware model closes around the solve.

    Attributes:
        max_iters: The iteration limit.
        tolerance: The change below which the loop has converged.
        mixing: The fraction of the new iterate taken at each step.
    """

    max_iters: int = 50
    tolerance: float = 1e-6
    mixing: float = 1.0

    def __post_init__(self) -> None:
        if self.max_iters < 1:
            raise ValueError(f"max_iters is {self.max_iters}; at least one is needed.")
        if self.tolerance <= 0:
            raise ValueError(f"tolerance is {self.tolerance}; it has to be positive.")
        if not 0.0 < self.mixing <= 1.0:
            raise ValueError(f"mixing is {self.mixing}; it lies in (0, 1].")


@dataclass(frozen=True)
class ElectrostaticsSolverDescriptor:
    """One long-range solve, as a solver is asked to build it.

    Attributes:
        periodicity_profile: Which systems it is set up for; see
            :data:`PeriodicityProfile`.
        slab_normal: The axis the slab correction acts along, for ``z_slab``.
            ``None`` otherwise.
        multipole_max_l: The highest multipole order of the source density.
        kspace_cutoff: The reciprocal-space cutoff, in inverse Angstrom.
        smearing_width: The Gaussian width of each source, in Angstrom.
        scf_spec: The self-consistent loop around the solve, or ``None`` for a
            model that evaluates it once.
        external_field_flags: Applied-field and self-interaction options, as
            names.
        precision: What it computes in.
    """

    periodicity_profile: PeriodicityProfile
    multipole_max_l: int
    kspace_cutoff: float
    smearing_width: float
    slab_normal: int | None = None
    scf_spec: ScfSpec | None = None
    external_field_flags: frozenset[str] = field(default_factory=frozenset)
    precision: Precision = "float64"

    def __post_init__(self) -> None:
        if self.periodicity_profile not in PERIODICITY_PROFILES:
            raise ValueError(
                f"{self.periodicity_profile!r} is not a periodicity profile. "
                f"They are {list(PERIODICITY_PROFILES)}."
            )
        if (self.periodicity_profile == "z_slab") != (self.slab_normal is not None):
            raise ValueError(
                f"slab_normal is {self.slab_normal!r} under the profile "
                f"{self.periodicity_profile!r}. A slab needs the axis its "
                f"correction acts along, and nothing else takes one."
            )
        if self.slab_normal is not None and self.slab_normal not in (0, 1, 2):
            raise ValueError(
                f"slab_normal is {self.slab_normal}; it is an axis, 0, 1 or 2."
            )
        if self.multipole_max_l < 0:
            raise ValueError(
                f"multipole_max_l is {self.multipole_max_l}; it is at least 0."
            )
        if self.kspace_cutoff <= 0 or self.smearing_width <= 0:
            raise ValueError(
                f"kspace_cutoff is {self.kspace_cutoff} and smearing_width is "
                f"{self.smearing_width}; both have to be positive."
            )
