"""Proves the install, import and test wiring of the mace-jax package."""

import mace_jax


def test_version_is_exported():
    """A string here means pip resolved the distribution and the import
    name maps to it."""
    assert isinstance(mace_jax.__version__, str)
    assert mace_jax.__version__
