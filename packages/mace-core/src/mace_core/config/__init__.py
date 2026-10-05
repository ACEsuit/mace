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

__all__ = [
    "BaseConfig",
    "ConfigError",
    "ConfigSection",
    "apply_overrides",
    "parse_overrides",
    "read_config_file",
]
