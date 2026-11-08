"""FixedPointSCF's affine field response and its physical boundary conditions."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from mace_core.config.base import ConfigSection

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class FixedPointSCFConfig(ConfigSection):
    """The linear-response FixedPointSCF architecture, in eV, e and Å.

    The reference's OneBodyVariableUpdate with a linear potential embedding
    and identity invariant activation is affine in the density. Its fixed
    point is solved exactly, with implicit derivatives of the linear system.
    Nonlinear update variants require a separate solver and are not accepted.
    """

    kind: Literal["fixed_point_scf"] = "fixed_point_scf"
    multipole_max_l: Literal[0, 1] = 0
    multipole_width: Positive = 1.0
    feature_max_l: Literal[0, 1] = 0
    feature_widths: tuple[Positive, ...] = (1.0,)
    feature_norms: tuple[Positive, ...] | None = None
    feature_self_interaction: bool = False
    energy_self_interaction: bool = False
    quadrupole_corrections: bool = False
    constant_charge: bool = True
    fermi_level_offset: float = Field(default=0.0, allow_inf_nan=False)
    local_source: Literal["linear", "gated"] = "gated"
    density_scales: tuple[float, ...] | None = None
    potential_embedding: Literal["linear", "biased_linear"] = "biased_linear"
    update_nonlinearity: Literal["none"] = "none"
    electron_energy: Literal["none", "quadratic_field", "linear_charges"] = "none"

    @model_validator(mode="after")
    def _validate_shapes(self):
        import math

        if not self.feature_widths:
            raise ValueError("fixed_point_scf.feature_widths must not be empty")
        if self.feature_norms is not None and len(self.feature_norms) != len(
            self.feature_widths
        ) * (self.feature_max_l + 1):
            raise ValueError(
                "fixed_point_scf.feature_norms needs one value per width and degree"
            )
        if self.density_scales is not None:
            if self.local_source != "linear":
                raise ValueError(
                    "fixed_point_scf.density_scales requires local_source='linear'"
                )
            if not all(math.isfinite(value) for value in self.density_scales):
                raise ValueError("fixed_point_scf.density_scales must be finite")
        return self
