"""Long-range electrostatics for the torch stack: the solver and its dispatch.

The reference solver lives in :mod:`mace_torch.electrostatics.reference` and is
registered through the ``mace.electrostatics_backends.torch`` entry point
group, the way any other solver is. A model asks for its long-range op with
:func:`build_long_range` when it is built, and holds the op from then on.
"""

from mace_torch.electrostatics.solver import (
    PBC_HANDLING,
    ElectrostaticsSolver,
    LongRangeEnergy,
    LongRangeFeatures,
    LongRangeGeometry,
    ReferenceSolver,
    build_long_range,
    build_long_range_features,
    build_scf_solve,
    long_range_geometry,
    reciprocal_cell_and_volume,
)

__all__ = [
    "PBC_HANDLING",
    "ElectrostaticsSolver",
    "LongRangeEnergy",
    "LongRangeFeatures",
    "LongRangeGeometry",
    "ReferenceSolver",
    "build_long_range",
    "build_long_range_features",
    "build_scf_solve",
    "long_range_geometry",
    "reciprocal_cell_and_volume",
]
