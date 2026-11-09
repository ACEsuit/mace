"""Physical input declarations shared by model families and data consumers."""

from mace_core.config.model import ModelConfig
from mace_core.observables import InputSpec

CHARGES = InputSpec(name="charges", irreps="0e", per_atom=True, units="e")
TOTAL_CHARGE = InputSpec(name="total_charge", irreps="0e", per_atom=False, units="e")
TOTAL_SPIN = InputSpec(name="total_spin", irreps="0e", per_atom=False, units="1")
EXTERNAL_FIELD = InputSpec(
    name="external_field", irreps="1o", per_atom=False, units="eV/(e*Å)"
)


def with_embedded_inputs(
    config: ModelConfig, physical: tuple[InputSpec, ...]
) -> tuple[InputSpec, ...]:
    """Include embedded features, retaining physical units for shared inputs."""
    specs = {
        feature.name: InputSpec(
            name=feature.name,
            irreps=f"{feature.input_dim}x0e",
            per_atom=feature.per == "atom",
            units="1",
        )
        for feature in config.graph_features
    }
    specs.update({spec.name: spec for spec in physical})
    return tuple(specs.values())
