"""Framework-agnostic contract and pure math for MACE v1.

This package imports no framework. Everything here is expressed over plain
Python, pydantic, numpy and ase, so the same types carry ``torch.Tensor`` in
``mace_torch`` and ``jax.Array`` in ``mace_jax``.

The configuration system, the model metadata, the observable specification and
the typed model output are re-exported here. The other submodules are imported
from their own modules, because some of them are not cheap to import and a
caller that only wants a unit constant should not pay for a file parser:

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
from mace_core.observables import (
    DEFAULT_CATALOGUE,
    DerivativeSpec,
    InputSpec,
    ObservableCatalogue,
    ObservableSpec,
)
from mace_core.outputs import MACEOutput

__all__ = [
    "DEFAULT_CATALOGUE",
    "BaseConfig",
    "ConfigError",
    "ConfigSection",
    "DerivativeSpec",
    "InputSpec",
    "MACEOutput",
    "ModelMetadata",
    "ObservableCatalogue",
    "ObservableSpec",
    "__version__",
    "format_citations",
]

#: Version of the installed `mace-core` distribution. Read from installed metadata
#: rather than hardcoded, so it cannot drift from what pip resolved.
try:
    __version__ = version("mace-core")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0.0.0"
