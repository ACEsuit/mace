"""Config sections of the model schema.

So far the radial embedding: the radial basis, the distance transform and the
cutoff envelope, one section each. A field of several kinds is a union tagged
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

from pydantic import Field

from mace_core.config.base import ConfigSection

__all__ = [
    "AgnesiTransformConfig",
    "BesselBasisConfig",
    "ChebyshevBasisConfig",
    "DistanceTransformConfig",
    "GaussianBasisConfig",
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
