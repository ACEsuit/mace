"""The PyTorch stack for MACE v1.

The public surface arrives with the tickets that build it. Present so far:

* ``mace_torch.nn.radial`` and ``mace_torch.nn.embedding``: the radial bases,
  cutoff, pair repulsion, distance transforms and the two embedding blocks;
* ``mace_torch.backends.reference.spherical_harmonics``: the native spherical harmonics.

Nothing is re-exported here: this module stays light, so importing the package
never pulls in a submodule a caller did not ask for.
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
