"""Apply the same capability contract to package and repository tests."""

from tests.conftest import (
    capability_collection_modifyitems as pytest_collection_modifyitems,
    capability_runtest_setup as pytest_runtest_setup,
    configure_capabilities as pytest_configure,
)
