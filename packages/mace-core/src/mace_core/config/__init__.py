"""Configuration schemas for MACE v1.

The base machinery lives in `base` and the command-line override grammar in
`cli`; the training schema sections (model, data, training, ...) arrive with
their own tickets and are re-exported from here.
"""

from mace_core.config.base import (
    BaseConfig,
    ConfigError,
    ConfigSection,
    read_config_file,
)
from mace_core.config.cli import apply_overrides, parse_overrides
from mace_core.config.model import (
    AgnesiTransformConfig,
    BesselBasisConfig,
    ChebyshevBasisConfig,
    DistanceTransformConfig,
    GaussianBasisConfig,
    NoDistanceTransformConfig,
    PolynomialCutoffConfig,
    RadialBasisConfig,
    SoftTransformConfig,
)

__all__ = [
    "AgnesiTransformConfig",
    "BaseConfig",
    "BesselBasisConfig",
    "ChebyshevBasisConfig",
    "ConfigError",
    "ConfigSection",
    "DistanceTransformConfig",
    "GaussianBasisConfig",
    "NoDistanceTransformConfig",
    "PolynomialCutoffConfig",
    "RadialBasisConfig",
    "SoftTransformConfig",
    "apply_overrides",
    "parse_overrides",
    "read_config_file",
]
