"""Quadratic charge equilibration, with explicit optional atomic inputs."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from mace_core.config.base import ConfigSection


class QEqConfig(ConfigSection):
    """The quadratic model: charges in e, lengths in Å and energies in eV.

    Learned electronegativities add to an optional atomic input. Hardness is
    either supplied or constant, plus an optional learned correction. No
    positivity transform or diagonal regularizer changes the reference model.
    """

    kind: Literal["qeq"] = "qeq"
    multipole_width: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    cutoff_widths: tuple[Annotated[float, Field(gt=0, allow_inf_nan=False)], ...] = (
        1.0,
    )
    energy_self_interaction: bool = True
    default_hardness: float = Field(default=2.0, allow_inf_nan=False)
    train_hardness: bool = False
    electronegativity_input: str | None = None
    hardness_input: str | None = None
    charge_method: Literal["energy", "nonperiodic"] = "energy"

    @property
    def atomic_inputs(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (self.electronegativity_input, self.hardness_input)
            if name is not None
        )

    @model_validator(mode="after")
    def _valid_inputs(self):
        from mace_core.data.configuration import RESERVED_PROPERTY_NAMES
        from mace_core.graph.collate import feature_levels

        if len(set(self.atomic_inputs)) != len(self.atomic_inputs):
            raise ValueError("QEq electronegativity and hardness need different inputs")
        for name in self.atomic_inputs:
            if not name or "." in name or name in RESERVED_PROPERTY_NAMES:
                raise ValueError("QEq atomic inputs need nonempty custom field names")
            feature_levels({name: "atom"})
        return self
