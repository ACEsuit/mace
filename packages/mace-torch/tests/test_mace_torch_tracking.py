"""Where a run reports itself, and what it costs a run that reports nowhere.

What these assert is that a run with tracking off never reaches the client, and
that a run with tracking on and the package missing says which extra to install
rather than raising from inside a constructor. Neither depends on whether the
client happens to be installed where the suite runs, since it is installed
beside the frozen tree and not beside this package.
"""

from __future__ import annotations

import sys

import pytest
from mace_core.config.resolved import ResolvedConfig
from mace_torch.train import NullTracker, Tracker, epoch_values, open_tracker


def configuration(**wandb) -> ResolvedConfig:
    return ResolvedConfig.model_validate(
        {
            "runtime": {"name": "run", "wandb": wandb},
            "data": {"heads": {"default": {"train_file": "train.xyz"}}},
            "model": {"observables": ["energy", "forces"]},
        }
    )


def test_tracking_is_off_by_default():
    assert isinstance(open_tracker(configuration()), NullTracker)


def test_a_run_without_tracking_never_imports_the_client():
    """Asked as "did this call import it", not "is it imported": the suite
    runs beside the frozen tree, which has its own wandb integration, so
    something else may well have imported it already."""
    before = "wandb" in sys.modules
    open_tracker(configuration())
    assert ("wandb" in sys.modules) == before


def test_the_null_tracker_answers_the_whole_interface():
    """The loop holds one of these unconditionally, so a method it does not
    have is a crash at the end of a training run and not at the start."""
    tracker = NullTracker()
    assert isinstance(tracker, Tracker)
    tracker.log({"loss": 1.0}, step=0)
    tracker.summary({"final": 1.0})
    tracker.finish()


def test_asking_for_a_client_that_is_not_installed_names_the_extra(monkeypatch):
    """The import is made to fail rather than the package uninstalled, so the
    message is asserted wherever the suite runs."""
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match=r"mace-torch\[wandb\]"):
        open_tracker(configuration(enabled=True))


def test_the_epoch_keys_are_the_table_row_with_the_metric_appended():
    """So a chart and a table row are about the same thing."""
    values = epoch_values({"water": {"loss": 1.0, "rmse_forces": 0.5}})
    assert values == {"valid_water_loss": 1.0, "valid_water_rmse_forces": 0.5}
