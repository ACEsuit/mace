"""Long-range electrostatics, for the models that carry it.

An expert section: written in a configuration file and read by ``mace train``
like any other, and not one of the flags a basic run is described by. It
names the solver the model's long-range op is built with and the systems that
op is set up for. The solver is part of what the model computes, so it is
recorded with the rest of the configuration and read back with the model.
"""

from __future__ import annotations

from pydantic import model_validator

from mace_core.config.base import ConfigSection
from mace_core.electrostatics.spec import (
    PeriodicityProfile,
    PositiveFinite,
    SlabNormal,
    validate_slab_normal,
)

__all__ = ["ElectrostaticsConfig"]


class ElectrostaticsConfig(ConfigSection):
    """Which solver computes the long-range energy, and for which systems.

    Args:
        enabled: Whether the model carries a long-range term.
        solver: The name it is registered under in
            ``mace.electrostatics_backends.<framework>``.
        periodicity_profile: The systems the solve is set up for.
        slab_normal: The axis a slab correction acts along, for ``z_slab``.
        kspace_cutoff_factor: The reciprocal-space cutoff, as a multiple of
            the one the Gaussian widths call for.
    """

    enabled: bool = False
    solver: str = "reference"
    periodicity_profile: PeriodicityProfile = "partial"
    slab_normal: SlabNormal | None = None
    kspace_cutoff_factor: PositiveFinite = 1.5

    @model_validator(mode="after")
    def _a_slab_says_its_normal(self) -> ElectrostaticsConfig:
        validate_slab_normal(self.periodicity_profile, self.slab_normal)
        return self
