"""Framework-agnostic contract and pure math for MACE v1.

This package imports no framework: everything here is plain Python, numpy and
ase, so the same types serve the torch and the jax implementations alike.

The configuration system and the model metadata are re-exported here. The
other submodules are imported from their own modules, because some of them are
not cheap to import and a caller that only wants a unit constant should not
pay for a file parser:

``mace_core.data``
    :class:`~mace_core.data.configuration.Configuration`, the boundary object
    of the data layer, its key specification, and the parsing and splitting
    functions over it.

``mace_core.elements``
    the default property keys and the element index table.

``mace_core.units``
    unit constants, and the single statement of each physics sign convention.

Heavier ones still (the Clebsch-Gordan basis, neighbours) follow the same rule.
"""

from importlib.metadata import PackageNotFoundError, version

from mace_core.config import BaseConfig, ConfigError, ConfigSection
from mace_core.metadata import ModelMetadata, format_citations

__all__ = [
    "BaseConfig",
    "ConfigError",
    "ConfigSection",
    "ModelMetadata",
    "__version__",
    "format_citations",
]

#: Version of the installed `mace-core` distribution. Read from installed metadata
#: rather than hardcoded, so it cannot drift from what pip resolved.
try:
    __version__ = version("mace-core")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0.0.0"
