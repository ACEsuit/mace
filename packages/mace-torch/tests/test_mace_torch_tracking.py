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
from conftest import fp64_only
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
    with pytest.raises(ImportError, match=r"mace-torch-v1\[wandb\]"):
        open_tracker(configuration(enabled=True))


def test_the_epoch_keys_are_the_table_row_with_the_metric_appended():
    """So a chart and a table row are about the same thing."""
    values = epoch_values({"water": {"loss": 1.0, "rmse_forces": 0.5}})
    assert values == {"valid_water_loss": 1.0, "valid_water_rmse_forces": 0.5}


def test_flags_and_full_config_reach_the_owned_run(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mace_torch.cli.run_train import parse

    calls, logged, finished = [], [], []
    run = SimpleNamespace(
        summary={},
        log=lambda values, **kw: logged.append((values, kw)),
        finish=lambda **kw: finished.append(kw),
    )

    def init(**kwargs):
        calls.append(kwargs)
        return run

    def global_log(*args, **kwargs):
        pytest.fail("metrics must be sent to the owned run, not the global active run")

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=init, log=global_log)
    )
    config = parse(
        [
            "--wandb",
            "--wandb_project",
            "audit-project",
            "--wandb_entity",
            "audit-team",
            "--wandb_name",
            "audit-run",
            "--wandb_dir",
            str(tmp_path),
        ]
    )
    tracker = open_tracker(config)
    assert calls == [
        dict(
            project="audit-project",
            entity="audit-team",
            name="audit-run",
            dir=str(tmp_path),
            config=config.to_resolved_dict(),
        )
    ]
    values = epoch_values({"water": {"loss": 1.0}, "salt": {"loss": 2.0}})
    tracker.log(values, step=3)
    tracker.summary({"final_valid_water_loss": 1.0})
    tracker.finish()
    assert logged == [(values, {"step": 3})]
    assert run.summary == {"final_valid_water_loss": 1.0}
    assert finished == [{"exit_code": 0}]


def test_non_main_rank_never_opens_a_client(monkeypatch):
    from mace_torch.train import tracking

    monkeypatch.setattr(tracking.dist, "is_available", lambda: True)
    monkeypatch.setattr(tracking.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tracking.dist, "get_rank", lambda: 1)
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert isinstance(open_tracker(configuration(enabled=True)), NullTracker)


def test_allowlist_changes_are_refused():
    from mace_torch.cli.run_train import parse

    for names in (["lr"], ["lr", "not_an_argument"]):
        with pytest.raises(ValueError, match="resolved config"):
            parse(["--wandb_log_hypers", *names])


@fp64_only
@pytest.mark.parametrize(
    "outcome", ["complete", "dry", "step_failure", "report_failure"]
)
def test_tracker_closes_on_every_training_exit(tmp_path, monkeypatch, outcome):
    from types import SimpleNamespace

    from mace_torch.train import loop, run_data_stage, run_model_stage
    from test_mace_torch_training_pipeline import CATALOGUE
    from test_mace_torch_training_pipeline import configuration as task

    config = task(tmp_path, max_num_epochs=1, dry_run=outcome == "dry")
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    calls, exits = [], []
    run = SimpleNamespace(
        summary={},
        log=lambda *args, **kwargs: None,
        finish=lambda **kwargs: exits.append(kwargs),
    )

    def init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, log=run.log))
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "wandb": config.runtime.wandb.model_copy(update={"enabled": True})
                }
            )
        }
    )

    def fail(*args, **kwargs):
        raise RuntimeError("injected training failure")

    if outcome.endswith("failure"):
        target = "train_one_epoch" if outcome == "step_failure" else "report_errors"
        monkeypatch.setattr(loop, target, fail)
        with pytest.raises(RuntimeError, match="injected training failure"):
            loop.run_train_stage(config, built)
    else:
        loop.run_train_stage(config, built)
    assert len(calls) == (0 if outcome == "dry" else 1)
    assert exits == (
        [] if outcome == "dry" else [{"exit_code": int(outcome.endswith("failure"))}]
    )
