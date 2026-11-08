"""Experimental, non-self-consistent local multipole models (ELEC-3)."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from mace_core.config.base import ConfigSection


class LocalChargeConfig(ConfigSection):
    """Direct local multipoles; widths in Å, density components in e Å**l."""

    kind: Literal["lc_mace"] = "lc_mace"
    multipole_max_l: int = Field(default=0, ge=0, le=1)
    multipole_width: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    energy_self_interaction: bool = False


class LocalSplitChargeConfig(ConfigSection):
    """Conservative bond transfers around supplied formal charges, in e.

    Choose either a declared per-atom input or a table keyed by atomic number.
    Formal charges are inputs, never fitted targets or an inferred total.
    """

    kind: Literal["lsc_mace"] = "lsc_mace"
    multipole_max_l: int = Field(default=0, ge=0, le=1)
    multipole_width: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    energy_self_interaction: bool = False
    formal_charge_input: str | None = None
    formal_charges: dict[int, Annotated[float, Field(allow_inf_nan=False)]] | None = (
        None
    )

    @model_validator(mode="after")
    def _one_charge_source(self):
        if (self.formal_charge_input is None) == (self.formal_charges is None):
            raise ValueError(
                "local_sources requires exactly one of formal_charge_input "
                "(a declared per-atom input) or formal_charges (atomic number → e)"
            )
        if self.formal_charge_input is not None:
            from mace_core.data.configuration import RESERVED_PROPERTY_NAMES
            from mace_core.graph.collate import feature_levels

            if (
                not self.formal_charge_input
                or "." in self.formal_charge_input
                or self.formal_charge_input in RESERVED_PROPERTY_NAMES
            ):
                raise ValueError(
                    "formal_charge_input must name a custom input, e.g. "
                    "oxidation_state; graph fields and dotted names are reserved"
                )
            feature_levels({self.formal_charge_input: "atom"})
        if self.formal_charges is not None and (
            not self.formal_charges
            or any(not 1 <= number <= 118 for number in self.formal_charges)
        ):
            raise ValueError("formal_charges requires atomic numbers in [1, 118]")
        return self


LocalSourcesConfig = Annotated[
    LocalChargeConfig | LocalSplitChargeConfig, Field(discriminator="kind")
]
