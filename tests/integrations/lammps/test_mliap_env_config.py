"""The six `MACE_*` variables the ML-IAP coupling reads.

`MACELammpsConfig.__init__` is the only place they are read, and nothing
exercised any of them. They are the knobs a user reaches for when a LAMMPS run is
slow or on the wrong device, which is the worst time to discover that one of them
is spelled differently than the docs say or no longer reaches anything.

Two are more than knobs. `MACE_ALLOW_CPU` decides whether a KOKKOS run with
host-side tensors raises or proceeds, and `MACE_FORCE_CPU` overrides the device
choice entirely, so both change what a production run does. `MACE_PROFILE_END`
calls `sys.exit()`, which is worth knowing before it happens mid-simulation.

The parsing tests spell out the accepted spellings because `_get_env_bool`
accepts exactly `true`, `1`, `t` and `yes`, case-insensitively, and silently
treats everything else as false: `on`, `y` and `TRUE ` with a trailing space are
all off, and none of that is visible at the call site.
"""

import logging
import sys

import numpy as np
import pytest
import torch

from mace.calculators.lammps_mliap_mace import (
    LAMMPS_MLIAP_MACE,
    MACELammpsConfig,
    timer,
)
from tests.integrations.lammps._harness import StubMACE

BOOL_VARS = ["MACE_TIME", "MACE_PROFILE", "MACE_ALLOW_CPU", "MACE_FORCE_CPU"]
ATTRIBUTES = {
    "MACE_TIME": "debug_time",
    "MACE_PROFILE": "debug_profile",
    "MACE_ALLOW_CPU": "allow_cpu",
    "MACE_FORCE_CPU": "force_cpu",
}


@pytest.fixture(name="clean_env", autouse=True)
def fixture_clean_env(monkeypatch):
    """No MACE_* left over from the ambient environment or another test."""
    for var in BOOL_VARS + ["MACE_PROFILE_START", "MACE_PROFILE_END"]:
        monkeypatch.delenv(var, raising=False)


class _PlainData:
    """The non-KOKKOS coupling: its module name has no "kokkos" in it."""

    def __init__(self):
        self.elems = np.zeros(3, dtype=np.int32)


class _KokkosData:
    """Stands in for the KOKKOS coupling, which is recognised by module name."""

    def __init__(self, device="cpu"):
        self.elems = torch.zeros(3, dtype=torch.int32, device=device)


# The device branch keys off `data.__class__.__module__`, so the stub has to
# claim a kokkos module rather than merely be named one.
_KokkosData.__module__ = "lammps.mliap.mliap_unified_couple_kokkos"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_every_switch_is_off_by_default():
    config = MACELammpsConfig()

    assert not config.debug_time
    assert not config.debug_profile
    assert not config.allow_cpu
    assert not config.force_cpu


def test_the_profile_window_defaults_to_five_and_ten():
    config = MACELammpsConfig()

    assert config.profile_start_step == 5
    assert config.profile_end_step == 10


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("var", BOOL_VARS)
@pytest.mark.parametrize("value", ["true", "TRUE", "True", "1", "t", "T", "yes", "YES"])
def test_the_accepted_spellings_turn_a_switch_on(monkeypatch, var, value):
    monkeypatch.setenv(var, value)

    assert getattr(MACELammpsConfig(), ATTRIBUTES[var]) is True


@pytest.mark.parametrize("var", BOOL_VARS)
@pytest.mark.parametrize("value", ["false", "0", "no", "off", "on", "y", "", "2"])
def test_everything_else_leaves_it_off(monkeypatch, var, value):
    """`on`, `y` and `no` are all off. Documented here because the call site
    reads like any truthy string would do, and a user who writes `MACE_TIME=on`
    gets silence rather than an error."""
    monkeypatch.setenv(var, value)

    assert getattr(MACELammpsConfig(), ATTRIBUTES[var]) is False


@pytest.mark.parametrize("var,attribute", [
    ("MACE_PROFILE_START", "profile_start_step"),
    ("MACE_PROFILE_END", "profile_end_step"),
])
def test_the_profile_window_is_read_from_the_environment(monkeypatch, var, attribute):
    monkeypatch.setenv(var, "42")

    assert getattr(MACELammpsConfig(), attribute) == 42


@pytest.mark.parametrize("var", ["MACE_PROFILE_START", "MACE_PROFILE_END"])
def test_a_non_integer_window_is_refused_at_construction(monkeypatch, var):
    """`int()` raises, and it raises here rather than at the step it would have
    fired, which is the difference between a typo and a run that dies at step 5.
    """
    monkeypatch.setenv(var, "soon")

    with pytest.raises(ValueError):
        MACELammpsConfig()


# ---------------------------------------------------------------------------
# MACE_TIME
# ---------------------------------------------------------------------------


def test_the_timer_logs_when_it_is_enabled(caplog):
    with caplog.at_level(logging.INFO):
        with timer("probe", enabled=True):
            pass

    assert any("Timer - probe" in record.message for record in caplog.records)


def test_the_timer_says_nothing_when_it_is_not(caplog):
    with caplog.at_level(logging.INFO):
        with timer("probe", enabled=False):
            pass

    assert not any("Timer - probe" in record.message for record in caplog.records)


def test_mace_time_reaches_the_timer(monkeypatch, caplog):
    """The flag and the timer, connected: `debug_time` is what the step passes as
    `enabled`."""
    monkeypatch.setenv("MACE_TIME", "1")
    config = MACELammpsConfig()

    with caplog.at_level(logging.INFO):
        with timer("step", enabled=config.debug_time):
            pass

    assert any("Timer - step" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# MACE_PROFILE, MACE_PROFILE_START, MACE_PROFILE_END
# ---------------------------------------------------------------------------


@pytest.fixture(name="unified")
def fixture_unified():
    return LAMMPS_MLIAP_MACE(StubMACE(1))


def test_profiling_off_does_not_touch_the_profiler(monkeypatch, unified):
    calls = []
    monkeypatch.setattr(torch.cuda.profiler, "start", lambda: calls.append("start"))
    monkeypatch.setattr(torch.cuda.profiler, "stop", lambda: calls.append("stop"))
    unified.config.debug_profile = False
    unified.step = unified.config.profile_start_step

    unified._manage_profiling()  # pylint: disable=protected-access

    assert calls == []


def test_the_profiler_starts_at_the_step_it_was_told(monkeypatch, unified):
    calls = []
    monkeypatch.setattr(torch.cuda.profiler, "start", lambda: calls.append("start"))
    unified.config.debug_profile = True
    unified.config.profile_start_step = 3
    unified.step = 3

    unified._manage_profiling()  # pylint: disable=protected-access

    assert calls == ["start"]


def test_no_start_on_any_other_step(monkeypatch, unified):
    calls = []
    monkeypatch.setattr(torch.cuda.profiler, "start", lambda: calls.append("start"))
    unified.config.debug_profile = True
    unified.config.profile_start_step = 3
    unified.step = 2

    unified._manage_profiling()  # pylint: disable=protected-access

    assert calls == []


def test_the_end_step_stops_the_profiler_and_exits(monkeypatch, unified):
    """`sys.exit()` in the middle of a LAMMPS run is the surprising part, so it
    is stated rather than left to be discovered."""
    calls = []
    monkeypatch.setattr(torch.cuda.profiler, "stop", lambda: calls.append("stop"))
    unified.config.debug_profile = True
    unified.config.profile_start_step = 3
    unified.config.profile_end_step = 7
    unified.step = 7

    with pytest.raises(SystemExit):
        unified._manage_profiling()  # pylint: disable=protected-access

    assert calls == ["stop"]


# ---------------------------------------------------------------------------
# MACE_ALLOW_CPU, MACE_FORCE_CPU
# ---------------------------------------------------------------------------


def test_a_non_kokkos_coupling_always_runs_on_cpu(unified):
    """The plain coupling has no device to disagree about."""
    unified._initialize_device(_PlainData())  # pylint: disable=protected-access

    assert unified.device == torch.device("cpu")


def test_kokkos_with_host_tensors_is_refused_by_default(unified):
    """The message names the variable that would allow it, which is the only
    reason this failure is actionable."""
    with pytest.raises(ValueError, match="MACE_ALLOW_CPU"):
        unified._initialize_device(_KokkosData())  # pylint: disable=protected-access


def test_allow_cpu_lets_a_kokkos_run_proceed_on_the_host(monkeypatch):
    monkeypatch.setenv("MACE_ALLOW_CPU", "true")
    unified = LAMMPS_MLIAP_MACE(StubMACE(1))

    unified._initialize_device(_KokkosData())  # pylint: disable=protected-access

    assert unified.device == torch.device("cpu")


def test_force_cpu_skips_the_device_check_entirely(monkeypatch):
    """`MACE_FORCE_CPU` takes the non-kokkos branch, so the CPU tensors never
    reach the check that `MACE_ALLOW_CPU` guards."""
    monkeypatch.setenv("MACE_FORCE_CPU", "true")
    unified = LAMMPS_MLIAP_MACE(StubMACE(1))

    unified._initialize_device(_KokkosData())  # pylint: disable=protected-access

    assert unified.device == torch.device("cpu")


def test_force_cpu_wins_over_allow_cpu_being_unset(monkeypatch):
    """The two are independent: forcing CPU does not require allowing it."""
    monkeypatch.setenv("MACE_FORCE_CPU", "yes")
    unified = LAMMPS_MLIAP_MACE(StubMACE(1))

    assert unified.config.force_cpu is True
    assert unified.config.allow_cpu is False
    unified._initialize_device(_KokkosData())  # pylint: disable=protected-access
    assert unified.device == torch.device("cpu")
