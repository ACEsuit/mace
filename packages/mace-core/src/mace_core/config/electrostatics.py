"""Long-range electrostatics, for the models that carry it.

An expert section: written in a configuration file and read by ``mace train``
like any other, and not one of the flags a basic run is described by. It
names the solver the model's long-range op is built with and the systems that
op is set up for. The solver is part of what the model computes, so it is
recorded with the rest of the configuration and read back with the model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from mace_core.config.section import FrozenSection
from mace_core.electrostatics.descriptor import PeriodicityProfile

__all__ = ["ElectrostaticsConfig"]


class ElectrostaticsConfig(FrozenSection):
    """Which solver computes the long-range energy, and for which systems.

    Args:
        enabled: Whether the model carries a long-range term.
        solver: The name it is registered under in
            ``mace.electrostatics_backends.<framework>``.
        periodicity_profile: The systems the solve is set up for.
        slab_normal: The axis a slab correction acts along, for ``z_slab``.
        kspace_cutoff_factor: The reciprocal-space cutoff, as a multiple of
            the one the Gaussian widths call for.
        training_step: How an epoch is trained when the model carries the
            term. ``standard`` is the ordinary step.
    """

    enabled: bool = False
    solver: str = "reference"
    periodicity_profile: PeriodicityProfile = "partial"
    slab_normal: int | None = None
    kspace_cutoff_factor: float = Field(default=1.5, gt=0)
    training_step: Literal["standard"] = "standard"

    @model_validator(mode="after")
    def _a_slab_says_its_normal(self) -> ElectrostaticsConfig:
        if (self.periodicity_profile == "z_slab") != (self.slab_normal is not None):
            raise ValueError(
                f"electrostatics.slab_normal is {self.slab_normal!r} under the "
                f"profile {self.periodicity_profile!r}. A slab needs the axis "
                f"its correction acts along, and nothing else takes one."
            )
        return self
