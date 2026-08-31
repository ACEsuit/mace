"""Framework-agnostic contract and pure math for MACE v1.

Scaffold only. The public surface arrives with the tickets that build it.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

#: Version of the installed `mace-core` distribution. Read from installed metadata
#: rather than hardcoded, so it cannot drift from what pip resolved.
try:
    __version__ = version("mace-core")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0.0.0"
