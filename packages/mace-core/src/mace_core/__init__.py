"""Framework-agnostic contract and pure math for MACE v1.

Re-exports the public surface of the lightweight submodules. Heavy ones (the
Clebsch-Gordan basis, neighbours) are imported from their own modules.
"""

from importlib.metadata import PackageNotFoundError, version

from mace_core.config import ConfigError, ConfigSection, ReforgeBaseConfig
from mace_core.metadata import ModelMetadata, format_citations

__all__ = [
    "ConfigError",
    "ConfigSection",
    "ModelMetadata",
    "ReforgeBaseConfig",
    "__version__",
    "format_citations",
]

#: Version of the installed `mace-core` distribution. Read from installed metadata
#: rather than hardcoded, so it cannot drift from what pip resolved.
try:
    __version__ = version("mace-core")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0.0.0"
