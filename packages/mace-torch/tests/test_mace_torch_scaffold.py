"""Proves the install, import and test wiring of the mace-torch package."""

import mace_torch


def test_version_is_exported():
    """A string here means pip resolved the distribution and the import
    name maps to it."""
    assert isinstance(mace_torch.__version__, str)
    assert mace_torch.__version__
