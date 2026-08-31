"""Engine selection, argv handling and the not-yet-migrated error."""

import sys

import mace_launcher
import pytest
from mace_launcher import DEFAULT_ENGINE, ENGINE_ENV_VAR, TARGETS, _take_engine


def test_every_console_script_has_a_dispatcher():
    """pyproject declares twelve scripts; each must resolve to a callable here."""
    for script in TARGETS:
        attribute = script.removeprefix("mace_")
        # Four scripts are not named after their module, but every one of them
        # is named after its dispatcher, which is what pyproject points at.
        assert hasattr(mace_launcher, attribute), f"{script} has no dispatcher"
        assert callable(getattr(mace_launcher, attribute))


def test_there_are_exactly_twelve_targets():
    assert len(TARGETS) == 12


def test_default_engine_is_legacy_so_an_install_behaves_as_before(monkeypatch):
    monkeypatch.delenv(ENGINE_ENV_VAR, raising=False)
    engine, argv = _take_engine(["--foo", "bar"])
    assert engine == DEFAULT_ENGINE == "legacy"
    assert argv == ["--foo", "bar"]


@pytest.mark.parametrize("spelling", [["--engine", "v1"], ["--engine=v1"]])
def test_engine_flag_is_consumed_before_the_target_sees_argv(spelling, monkeypatch):
    monkeypatch.delenv(ENGINE_ENV_VAR, raising=False)
    engine, argv = _take_engine([*spelling, "--config", "tiny.yaml"])
    assert engine == "v1"
    assert argv == ["--config", "tiny.yaml"]


def test_the_environment_variable_is_the_equivalent_of_the_flag(monkeypatch):
    monkeypatch.setenv(ENGINE_ENV_VAR, "v1")
    engine, argv = _take_engine(["--config", "tiny.yaml"])
    assert engine == "v1"
    assert argv == ["--config", "tiny.yaml"]


def test_the_flag_wins_over_the_environment_variable(monkeypatch):
    monkeypatch.setenv(ENGINE_ENV_VAR, "v1")
    engine, _ = _take_engine(["--engine", "legacy"])
    assert engine == "legacy"


def test_nothing_but_the_engine_flag_is_touched(monkeypatch):
    """The config-file path is the one where an extra argv pass changes precedence."""
    monkeypatch.delenv(ENGINE_ENV_VAR, raising=False)
    original = ["--config", "a.yaml", "--seed", "7", "--name", "--engine-ish"]
    engine, argv = _take_engine(list(original))
    assert engine == "legacy"
    assert argv == original


def test_arguments_after_a_bare_separator_are_left_alone(monkeypatch):
    monkeypatch.delenv(ENGINE_ENV_VAR, raising=False)
    engine, argv = _take_engine(["--", "--engine", "v1"])
    assert engine == "legacy"
    assert argv == ["--", "--engine", "v1"]


def test_an_unknown_engine_is_refused_by_name(monkeypatch):
    monkeypatch.delenv(ENGINE_ENV_VAR, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        _take_engine(["--engine", "jax"])
    assert "jax" in str(excinfo.value)
    assert "legacy" in str(excinfo.value)


def test_engine_flag_without_a_value_is_refused(monkeypatch):
    monkeypatch.delenv(ENGINE_ENV_VAR, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        _take_engine(["--engine"])
    assert "needs a value" in str(excinfo.value)


def test_an_unmigrated_capability_names_itself_rather_than_a_module(monkeypatch):
    """A missing v1 CLI is a routine migration state, not an ImportError."""
    monkeypatch.setattr(sys, "argv", ["mace_run_train", "--engine", "v1"])
    with pytest.raises(SystemExit) as excinfo:
        mace_launcher.run_train()
    message = str(excinfo.value)
    assert "mace_run_train" in message
    assert "not yet available on v1 engine" in message
    assert "--engine legacy" in message
