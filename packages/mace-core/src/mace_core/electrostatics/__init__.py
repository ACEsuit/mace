"""Long-range electrostatics: the solve, what a solver can do, and finding one.

Framework-free, like the rest of this package: descriptors carry names and
numbers, and no solver is named here. The solvers themselves live with their
framework and register through entry points.
"""

from mace_core.electrostatics.capabilities import (
    SolverCapabilities,
    UnsupportedSolveError,
)
from mace_core.electrostatics.descriptor import (
    PERIODICITY_PROFILES,
    REALSPACE_METHODS,
    ElectrostaticsSolverDescriptor,
    FeatureProjection,
    PeriodicityProfile,
    RealspaceMethod,
    ScfSpec,
)
from mace_core.electrostatics.registry import (
    ENTRY_POINT_GROUPS,
    DiscoveredSolver,
    SolverNotAvailableError,
    available_solvers,
    get_solver,
)

__all__ = [
    "ENTRY_POINT_GROUPS",
    "PERIODICITY_PROFILES",
    "REALSPACE_METHODS",
    "DiscoveredSolver",
    "ElectrostaticsSolverDescriptor",
    "FeatureProjection",
    "PeriodicityProfile",
    "RealspaceMethod",
    "ScfSpec",
    "SolverCapabilities",
    "SolverNotAvailableError",
    "UnsupportedSolveError",
    "available_solvers",
    "get_solver",
]
