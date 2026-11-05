"""The closed JAX v1.0 support list; publication references belong to FM-3."""

from dataclasses import dataclass


@dataclass(frozen=True)
class JaxModelSpec:
    model_id: str
    artifact_ref: str | None
    golden_ref: str
    required_blocks: frozenset[str]


_COMMON = frozenset(
    {"linear_readout", "nonlinear_readout", "bessel", "polynomial_cutoff"}
)
SUPPORTED_MODELS = {
    name: JaxModelSpec(name, None, golden, _COMMON | frozenset(blocks))
    for name, golden, blocks in (
        ("mace-mp-0-small", "mp_small_e3nn_cpu_fp64.json", ("residual",)),
        (
            "mace-mpa-0-medium",
            "mpa0_medium_e3nn_cpu_fp64.json",
            ("density", "density_residual", "agnesi", "zbl"),
        ),
        ("mace-off23-medium", "off_medium_e3nn_cpu_fp64.json", ("plain", "residual")),
    )
}
