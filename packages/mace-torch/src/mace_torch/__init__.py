"""The PyTorch stack for MACE v1.

Scaffold only. The public surface arrives with the tickets that build it.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

#: Version of the installed `mace-torch-v1` distribution. Read from installed metadata
#: rather than hardcoded, so it cannot drift from what pip resolved. The import
#: name and the distribution name differ for this package, which is why the
#: lookup spells the distribution name out.
try:
    __version__ = version("mace-torch-v1")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0.0.0"
