"""Proves the install, import and test wiring of the mace-core package."""

import mace_core


def test_version_is_exported():
    """A string here means pip resolved the distribution and the import name maps to it."""
    assert isinstance(mace_core.__version__, str)
    assert mace_core.__version__
