"""The environment a packaged conversion runs its extraction in.

Built with no network here: the pin names a wheel that does not exist, so the
environment is created for real and the installation is what fails. That is the
path whose diagnostics a user sees when their platform has no wheel.
"""

from __future__ import annotations

import pytest
from mace_torch.deploy.legacy import ExtractionFailed, pinned_environment


def test_an_uninstallable_pin_fails_with_the_installer_s_words(tmp_path):
    missing = tmp_path / "mace_torch-0.3.17-py3-none-any.whl"
    with pytest.raises(ExtractionFailed, match="could not install") as caught:
        pinned_environment(tmp_path / "environment", pin=str(missing))
    assert str(missing) in str(caught.value) or "does not exist" in str(caught.value)
    assert (tmp_path / "environment" / "bin" / "python").exists()


def test_a_failed_installation_is_not_mistaken_for_a_ready_one(tmp_path):
    """The marker is written only after a successful install, so a second
    call tries again instead of returning an empty environment."""
    missing = str(tmp_path / "absent.whl")
    for _ in range(2):
        with pytest.raises(ExtractionFailed):
            pinned_environment(tmp_path / "environment", pin=missing)
    assert not (tmp_path / "environment" / "mace-legacy-pin.txt").exists()
