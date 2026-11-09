"""Long-range electrostatics: the solve, what a solver can do, and finding one.

Framework-free, like the rest of this package: specifications carry names and
numbers, and no solver is named here. The solvers themselves live with their
framework and register through entry points.
"""

from mace_core.electrostatics.capabilities import (
    SolverCapabilities,
    UnsupportedSolveError,
)
from mace_core.electrostatics.registry import (
    ENTRY_POINT_GROUPS,
    DiscoveredSolver,
    DuplicateSolverError,
    SolverNotAvailableError,
    SolverSubstitutionError,
    available_solvers,
    get_solver,
    solver_to_load,
)
from mace_core.electrostatics.spec import (
    PERIODICITY_PROFILES,
    REALSPACE_METHODS,
    ElectrostaticsSpec,
    FeatureProjection,
    PeriodicityProfile,
    RealspaceMethod,
    ScfSpec,
    solver_spec_record,
)

__all__ = [
    "ENTRY_POINT_GROUPS",
    "PERIODICITY_PROFILES",
    "REALSPACE_METHODS",
    "DiscoveredSolver",
    "DuplicateSolverError",
    "ElectrostaticsSpec",
    "FeatureProjection",
    "PeriodicityProfile",
    "RealspaceMethod",
    "ScfSpec",
    "SolverCapabilities",
    "SolverNotAvailableError",
    "SolverSubstitutionError",
    "UnsupportedSolveError",
    "available_solvers",
    "get_solver",
    "solver_spec_record",
    "solver_to_load",
]
