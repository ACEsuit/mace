"""Framework-agnostic contract and pure math for MACE v1.

This package imports no framework. Everything here is expressed over plain
Python, numpy and pydantic, so the same types carry ``torch.Tensor`` in
``mace_torch`` and ``jax.Array`` in ``mace_jax``.
"""

from importlib.metadata import PackageNotFoundError, version

from mace_core.observables import (
    DerivativeSpec,
    InputSpec,
    ObservableCatalogue,
    ObservableSpec,
    load_default_catalogue,
)
from mace_core.outputs import MACEOutput

__all__ = [
    "DerivativeSpec",
    "InputSpec",
    "MACEOutput",
    "ObservableCatalogue",
    "ObservableSpec",
    "__version__",
    "load_default_catalogue",
]

#: Version of the installed `mace-core` distribution. Read from installed metadata
#: rather than hardcoded, so it cannot drift from what pip resolved.
try:
    __version__ = version("mace-core")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0.0.0"
