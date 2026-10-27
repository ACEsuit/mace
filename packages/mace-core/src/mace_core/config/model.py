"""Config sections of the model schema.

The radial embedding, one section each for the radial basis, the distance
transform and the cutoff envelope, and the model section that holds them with
the backbone's architecture. A field of several kinds is a union tagged
by `kind` (`radial_basis: {kind: gaussian, num_basis: 16}`); "no transform" is
the kind `none`, not an absent key. The envelope has one kind, so its section
is a plain section without a tag.

The cutoff radius `r_max` is in none of them. The basis, the envelope and the
neighbour list must all use the same radius, so it is one model-level field,
handed to each section's module when the model is built.

Defaults are the legacy command-line defaults (`--num_radial_basis 8`,
`--num_cutoff_basis 5`), which differ from some of the legacy class defaults.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from mace_core.config.base import ConfigSection

__all__ = [
    "AgnesiTransformConfig",
    "BesselBasisConfig",
    "ChebyshevBasisConfig",
    "DistanceTransformConfig",
    "GaussianBasisConfig",
    "InteractionKind",
    "ModelConfig",
    "NoDistanceTransformConfig",
    "PolynomialCutoffConfig",
    "RadialBasisConfig",
    "SoftTransformConfig",
]

# ---------------------------------------------------------------------------
# Radial basis
# ---------------------------------------------------------------------------


class BesselBasisConfig(ConfigSection):
    """Spherical Bessel functions of order zero, the default basis."""

    kind: Literal["bessel"] = "bessel"
    num_basis: int = Field(default=8, ge=1)
    #: Learn the frequencies instead of fixing them at `n * pi / r_max`.
    trainable: bool = False


class GaussianBasisConfig(ConfigSection):
    """Gaussians on evenly spaced centres in `[0, r_max]`."""

    kind: Literal["gaussian"] = "gaussian"
    #: At least two: the width is the spacing `r_max / (num_basis - 1)`.
    num_basis: int = Field(default=8, ge=2)
    #: Learn the centres instead of fixing them.
    trainable: bool = False


class ChebyshevBasisConfig(ConfigSection):
    """Chebyshev polynomials of the raw length; the input is not rescaled by
    `r_max`, so this basis ignores it."""

    kind: Literal["chebyshev"] = "chebyshev"
    num_basis: int = Field(default=8, ge=1)
    include_constant: bool = False  #: Start at `T_0 = 1` instead of `T_1`.


RadialBasisConfig = Annotated[
    BesselBasisConfig | GaussianBasisConfig | ChebyshevBasisConfig,
    Field(discriminator="kind"),
]

# ---------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------


class NoDistanceTransformConfig(ConfigSection):
    """The basis sees the raw edge lengths."""

    kind: Literal["none"] = "none"


class AgnesiTransformConfig(ConfigSection):
    """The Agnesi transform of ACEpotentials.jl, `1 / (1 + a y^q / (1 + y^(q-p)))`
    with `y` the length over half the pair's covalent radii sum. Defaults are
    the paper's fitted values."""

    kind: Literal["agnesi"] = "agnesi"
    exponent_q: float = 0.9183
    exponent_p: float = 4.5791
    amplitude: float = 1.0805
    trainable: bool = False


class SoftTransformConfig(ConfigSection):
    """A tanh clamp that flattens lengths below three quarters of the pair's
    covalent radii sum."""

    kind: Literal["soft"] = "soft"
    #: Dimensionless; divided by the width of the switching window.
    steepness: float = 4.0
    trainable: bool = False


DistanceTransformConfig = Annotated[
    NoDistanceTransformConfig | AgnesiTransformConfig | SoftTransformConfig,
    Field(discriminator="kind"),
]

# ---------------------------------------------------------------------------
# Cutoff envelope
# ---------------------------------------------------------------------------


class PolynomialCutoffConfig(ConfigSection):
    """The polynomial envelope: 1 at `r = 0`, zero with two vanishing
    derivatives at `r_max`, exactly zero beyond. The only envelope there is."""

    polynomial_order: int = Field(default=5, ge=1)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

#: The interaction blocks. Each is a different function and a trained model
#: depends on which one each layer has, so the first layer and the rest are
#: named separately.
#:
#: * `plain` applies the skip connection to the message, which replaces it.
#: * `residual` takes the skip from the layer's input and adds it after the
#:   product basis.
#: * `density` and `density_residual` are those two, dividing the messages by
#:   a density learned per atom instead of the average neighbour count.
#: * `nonlinear` conditions its radial network on both elements, normalizes
#:   by a learned density with two trained scalars, and ends in a gate. It
#:   carries its skip to the product basis, in the first layer too.
InteractionKind = Literal[
    "plain", "residual", "density", "density_residual", "nonlinear"
]


class ModelConfig(ConfigSection):
    """The architecture of the backbone.

    Defaults are the legacy command-line defaults. The node features of every
    layer but the last carry `num_channels` copies of each irrep up to `max_L`
    with natural parity, `0e+1o+2e+...`; the last carries only the scalars.
    """

    #: The cutoff radius in Angstrom, shared by the radial basis, the envelope
    #: and the neighbour list.
    r_max: float = Field(default=5.0, gt=0.0)
    num_interactions: int = Field(default=2, ge=1)
    num_channels: int = Field(default=128, ge=1)
    #: The highest degree of the node features.
    max_L: int = Field(default=1, ge=0)
    #: The highest degree of the edge spherical harmonics.
    max_ell: int = Field(default=3, ge=0)
    #: The body order of the product basis.
    correlation: int = Field(default=3, ge=1)
    radial_basis: RadialBasisConfig = BesselBasisConfig()
    distance_transform: DistanceTransformConfig = NoDistanceTransformConfig()
    cutoff: PolynomialCutoffConfig = PolynomialCutoffConfig()
    #: Multiply the envelope into the radial basis. Off, every interaction
    #: multiplies it into what its radial networks produce instead.
    apply_cutoff: bool = True
    #: The hidden widths of every interaction's radial network.
    radial_mlp: tuple[int, ...] = (64, 64, 64)
    interaction_first: InteractionKind = "residual"
    interaction: InteractionKind = "residual"
    #: What a `nonlinear` layer convolves, a whole declaration with one
    #: multiplicity for every term, such as `128x0e+128x1o`. Unset, the node
    #: features at their own width.
    edge_irreps: str | None = None
    #: Convolve only the scalars of `edge_irreps` in a `nonlinear` first layer.
    use_edge_irreps_first: bool = False
    #: Share the product basis's weights across elements.
    use_agnostic_product: bool = False
    #: The Clebsch-Gordan basis the symmetric contraction's weights are written
    #: against. Recorded with the model and never inferred from what is
    #: installed; `full` builds only on a backend that declares it.
    clebsch_gordan_basis: Literal["reduced", "full"] = "reduced"

    @model_validator(mode="after")
    def _edge_irreps_need_a_nonlinear_layer(self) -> "ModelConfig":
        """`edge_irreps` and `use_edge_irreps_first` are read by `nonlinear`
        layers only, and a setting nothing reads is refused rather than kept."""
        later_nonlinear = self.num_interactions > 1 and self.interaction == "nonlinear"
        first_nonlinear = self.interaction_first == "nonlinear"
        first_reads = first_nonlinear and self.use_edge_irreps_first
        if self.edge_irreps is not None and not (later_nonlinear or first_reads):
            raise ValueError(
                "edge_irreps is set but no layer reads it: a 'nonlinear' layer "
                "after the first does, and a 'nonlinear' first layer does with "
                "use_edge_irreps_first."
            )
        if self.use_edge_irreps_first and (
            self.edge_irreps is None or not first_nonlinear
        ):
            raise ValueError(
                "use_edge_irreps_first needs edge_irreps and a 'nonlinear' first layer."
            )
        return self
