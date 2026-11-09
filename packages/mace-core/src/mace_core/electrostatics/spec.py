"""What a long-range electrostatics solve is, before any solver agrees to it.

A solver specification is the complete statement of one solve: enough for a solver to
answer whether it can do it, and enough to build it. It carries numbers and
names only, so it is hashable, comparable and can be written into a
checkpoint. Dtypes are names, as everywhere in this package.

The solver is resolved from it once, when the model is built. Nothing resolves
in ``forward``.
"""

from __future__ import annotations

from dataclasses import asdict, field
from typing import Annotated, Any, Literal, get_args

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from mace_core.kernels.precision import Precision

__all__ = [
    "PERIODICITY_PROFILES",
    "REALSPACE_METHODS",
    "ElectrostaticsSpec",
    "FeatureProjection",
    "PeriodicityProfile",
    "RealspaceMethod",
    "ScfSpec",
    "solver_spec_record",
]

#: Which systems a solve is set up for.
#:
#: ``full_periodic`` is a bulk crystal, periodic along all three axes.
#: ``z_slab`` is periodic along two axes with vacuum along the slab normal, and
#: takes the slab dipole correction. ``molecular`` is periodic along none: the
#: system sits in the box the graph builder made around it. ``partial`` is a
#: batch that mixes those, each structure handled by its own periodicity, a
#: molecule among them in the box around it with the monopole and dipole
#: corrections that divide by that box's volume. ``per_structure`` is the same
#: batch with each molecule summed exactly in real space instead: its value is
#: then the isolated molecule's, whatever else the batch holds and whatever the
#: box. Measured on MACE-Polar S, a water cluster in its padding box is 2.1e-3 eV
#: from its real-space energy under ``partial``, and the gap changes with the
#: box and with the molecule's orientation.
PeriodicityProfile = Literal[
    "full_periodic", "z_slab", "molecular", "partial", "per_structure"
]

PERIODICITY_PROFILES: tuple[str, ...] = get_args(PeriodicityProfile)

#: How an open system's multipoles interact in real space.
#:
#: ``finite_difference`` represents each dipole as charges displaced along the
#: laboratory axes, which is what every published polar model was trained
#: against. Its error depends on the orientation of the system, so a rotated
#: molecule does not have quite the same energy. The method is part of what a
#: trained model computes, which is why it is a field here and not a setting
#: of the solver.
RealspaceMethod = Literal["finite_difference"]

REALSPACE_METHODS: tuple[str, ...] = get_args(RealspaceMethod)

# These constraints are shared by configuration files and direct solver calls.
PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeInteger = Annotated[int, Field(ge=0, strict=True)]
SlabNormal = Annotated[int, Field(ge=0, le=2, strict=True)]


def validate_slab_normal(profile: PeriodicityProfile, normal: int | None) -> None:
    """A slab declares an axis; other periodicity profiles do not."""
    if (profile == "z_slab") != (normal is not None):
        raise ValueError(
            f"slab_normal is {normal!r} under periodicity_profile {profile!r}; "
            "a slab needs its normal axis, and other profiles take none"
        )


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class FeatureProjection:
    """The potential of the source density, projected onto Gaussians per atom.

    The features a charge-aware model reads the field through. Each atom
    receives the potential of every other atom's density, projected onto
    Gaussians of its own.

    Attributes:
        max_l: The highest angular order projected onto.
        widths: The Gaussian widths projected onto, in Angstrom. One radial
            channel each.
        normalization: How each receiving Gaussian is normalized, by name.
        include_self_interaction: Whether an atom receives its own density's
            potential too.
        quadrupole_corrections: Whether an open system's projection carries
            the quadrupole correction as well as the monopole and dipole ones.
    """

    max_l: NonNegativeInteger
    widths: Annotated[tuple[PositiveFinite, ...], Field(min_length=1)]
    normalization: Literal["receiver", "multipoles"] = "receiver"
    include_self_interaction: bool = False
    quadrupole_corrections: bool = False

    @property
    def dimension(self) -> int:
        """Components per atom: every order, for every width."""
        return (self.max_l + 1) ** 2 * len(self.widths)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ScfSpec:
    """The self-consistent loop a charge-aware model closes around the solve.

    Attributes:
        max_iters: The iteration limit.
        tolerance: The change below which the loop has converged.
        mixing: The fraction of the new iterate taken at each step.
        method: The equation a fused span must solve. QEq and affine SCF
            use exact linear solves; the iteration controls do not apply.
            Affine and nonlinear SCF fix either total charge or Fermi level.
            Nonlinear methods use max_iters and a maximum residual tolerance;
            mixing is unused by the reference Newton solve.
    """

    max_iters: Annotated[int, Field(ge=1, strict=True)] = 50
    tolerance: PositiveFinite = 1e-6
    mixing: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 1.0
    method: Literal[
        "fixed_point",
        "qeq",
        "qeq_nonperiodic",
        "affine_fixed_charge",
        "affine_fixed_fermi",
        "nonlinear_fixed_charge",
        "nonlinear_fixed_fermi",
    ] = "fixed_point"


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ElectrostaticsSpec:
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
        features: The potential projection the model reads, or ``None`` for a
            model that reads only the energy.
        realspace_method: How an open system is summed in real space; see
            :data:`RealspaceMethod`.
        external_field_flags: Applied-field and self-interaction options, as
            names.
        precision: What it computes in.
    """

    periodicity_profile: PeriodicityProfile
    multipole_max_l: NonNegativeInteger
    kspace_cutoff: PositiveFinite
    smearing_width: PositiveFinite
    slab_normal: SlabNormal | None = None
    scf_spec: ScfSpec | None = None
    features: FeatureProjection | None = None
    realspace_method: RealspaceMethod = "finite_difference"
    external_field_flags: frozenset[str] = field(default_factory=frozenset)
    precision: Precision = "float64"

    def __post_init__(self) -> None:
        validate_slab_normal(self.periodicity_profile, self.slab_normal)


def solver_spec_record(solver_spec: ElectrostaticsSpec) -> dict[str, Any]:
    """The solve as plain JSON data: sets sorted, tuples as lists.

    What a checkpoint records, and what a rebuilt model's solve is compared
    against, so that a default that moved between versions is caught rather
    than silently changing a trained model's numbers.
    """

    def plain(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, frozenset | set):
            return sorted(plain(item) for item in value)
        if isinstance(value, tuple | list):
            return [plain(item) for item in value]
        return value

    return plain(asdict(solver_spec))
