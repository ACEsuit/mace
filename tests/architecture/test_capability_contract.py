"""The repository-wide plugin must skip locally and fail promised capabilities."""

from pathlib import Path
import pytest
from tests import conftest as capabilities


class Item:
    path = Path(__file__)

    def get_closest_marker(self, name):
        return pytest.mark.gpu if name == "gpu" else None


def test_unavailable_capability_skips_locally(monkeypatch):
    monkeypatch.setattr(capabilities, "capability_available", lambda name: False)
    monkeypatch.delenv("MACE_REQUIRE_CAPS", raising=False)
    with pytest.raises(pytest.skip.Exception, match="gpu"):
        capabilities.capability_runtest_setup(Item())


def test_promised_capability_fails_instead_of_skipping(monkeypatch):
    monkeypatch.setattr(capabilities, "capability_available", lambda name: False)
    monkeypatch.setenv("MACE_REQUIRE_CAPS", "gpu")
    with pytest.raises(pytest.fail.Exception, match="guarantees"):
        capabilities.capability_runtest_setup(Item())


def test_promised_capability_with_no_collected_test_fails(monkeypatch):
    monkeypatch.setenv("MACE_REQUIRE_CAPS", "gpu")
    with pytest.raises(pytest.UsageError, match="no collected test"):
        capabilities.capability_collection_modifyitems(None, [])
