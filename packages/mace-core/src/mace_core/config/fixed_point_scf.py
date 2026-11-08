"""FixedPointSCF field responses and their physical boundary conditions."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from mace_core.config.base import ConfigSection

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class FixedPointSCFConfig(ConfigSection):
    """FixedPointSCF architecture, in eV, e and Å.

    Affine responses use an exact linear solve. Nonlinear responses converge
    the residual equation, then differentiate it implicitly through second order.
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
    potential_embedding: Literal["linear", "biased_linear", "tanh", "norm_tanh"] = (
        "biased_linear"
    )
    update: Literal["one_body", "scaled", "scaled_dot", "many_body"] = "one_body"
    update_nonlinearity: Literal["none", "shallow", "mlp", "skip_mlp"] = "none"
    update_scales: tuple[float, ...] | None = None
    update_correlation: int = Field(default=1, ge=1, le=4)
    update_use_skip: bool = True
    update_interaction: Literal["density", "augmented"] = "density"
    update_radial_hidden: tuple[Annotated[int, Field(gt=0)], ...] = (64, 64, 64)
    central_mixer: Literal["none", "additive", "product"] = "additive"
    node_mixer: Literal["none", "additive", "product"] = "none"
    max_iters: int = Field(default=50, ge=1)
    tolerance: Positive = 1e-10
    electron_energy: Literal[
        "none",
        "quadratic_field",
        "linear_charges",
        "mlp_field",
        "quadratic_charges",
        "many_body_charges",
        "many_body_charges_field",
    ] = "none"

    @property
    def affine_response(self) -> bool:
        return (
            self.update == "one_body"
            and self.update_nonlinearity == "none"
            and self.potential_embedding in ("linear", "biased_linear")
        )

    @property
    def needs_edges(self) -> bool:
        return self.update == "many_body" or self.electron_energy.startswith(
            "many_body"
        )

    @model_validator(mode="after")
    def _validate_shapes(self):
        import math

        if self.potential_embedding == "tanh" and self.feature_max_l != 0:
            raise ValueError(
                "fixed_point_scf.potential_embedding='tanh' requires feature_max_l=0; "
                "use 'norm_tanh' for an equivariant vector potential"
            )
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
        if self.update_scales is not None:
            if self.update not in ("scaled", "scaled_dot"):
                raise ValueError(
                    "fixed_point_scf.update_scales requires a scaled update"
                )
            if not all(math.isfinite(value) for value in self.update_scales):
                raise ValueError("fixed_point_scf.update_scales must be finite")
        if self.update != "many_body" and (
            self.update_correlation != 1
            or not self.update_use_skip
            or self.update_interaction != "density"
            or self.update_radial_hidden != (64, 64, 64)
            or self.central_mixer != "additive"
            or self.node_mixer != "none"
        ):
            raise ValueError(
                "fixed_point_scf many-body options require update='many_body'"
            )
        return self
