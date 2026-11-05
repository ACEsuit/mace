"""The JAX stack for MACE v1, inference only.

Load native checkpoints through ``mace_jax.deploy.neutral_io.load_model``.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

#: Version of the installed `mace-jax` distribution. Read from installed metadata
#: rather than hardcoded, so it cannot drift from what pip resolved.
try:
    __version__ = version("mace-jax")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0.0.0"
