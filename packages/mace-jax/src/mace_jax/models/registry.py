"""The closed JAX v1.0 support list; publication references belong to FM-3."""

from dataclasses import dataclass

from mace_core.model_registry import get_model


@dataclass(frozen=True)
class JaxModelSpec:
    model_id: str
    golden_ref: str
    required_blocks: frozenset[str]

    @property
    def artifact_ref(self) -> str | None:
        reference = get_model(self.model_id).artifact_ref
        return None if reference is None else reference.url


_COMMON = frozenset(
    {"linear_readout", "nonlinear_readout", "bessel", "polynomial_cutoff"}
)
SUPPORTED_MODELS = {
    name: JaxModelSpec(
        name, get_model(name).golden_refs[0], _COMMON | frozenset(blocks)
    )
    for name, blocks in (
        ("mace-mp-0-small", ("residual",)),
        (
            "mace-mpa-0-medium",
            ("density", "density_residual", "agnesi", "zbl"),
        ),
        ("mace-off23-medium", ("plain", "residual")),
    )
}
