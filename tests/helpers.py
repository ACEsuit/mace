"""Shared helpers for the MACE test suite.

Consolidates the fixtures/utilities that used to be copy-pasted across test
files:

* ``run_mace_train`` — run the native train command (or another native CLI)
  in a subprocess with the repo prepended to ``PYTHONPATH``, replicating the
  historical pattern used throughout the suite.
* canonical availability flags (``CUET_AVAILABLE`` & co.) — one definition
  each, preserving the semantics of the original per-file definitions
  (a real ``import`` where files did a real import, ``find_spec`` otherwise).

Note: the shared ``fitting_configs`` / ``pretraining_configs`` fixtures live
in ``tests/conftest.py``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent.parent
TESTS_ROOT = Path(__file__).parent
CLI_DIRECTORY = REPO_ROOT / "packages/mace-torch/src/mace_torch/cli"
run_train = CLI_DIRECTORY / "run_train.py"
preprocess_data = CLI_DIRECTORY / "preprocess_data.py"
CLI_COMMANDS = {
    "run_train.py": ("train",),
    "preprocess_data.py": ("data", "prepare"),
    "eval_configs.py": ("eval",),
    "select_head.py": ("model", "select-head"),
    "create_lammps_model.py": ("export", "lammps"),
    "convert_device.py": ("model", "convert-device"),
    "fine_tuning_select.py": ("data", "select"),
    "plot_train.py": ("plot", "training"),
}


def v1_command(*command: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        "import sys; from mace_core.cli import main; raise SystemExit(main(sys.argv[1:]))",
        *command,
    ]


def cli_command(script) -> list[str]:
    path = Path(script)
    if path.parent == CLI_DIRECTORY:
        return v1_command(*CLI_COMMANDS[path.name])
    return [sys.executable, str(script)]


def skip_if_not_migrated(output) -> None:
    """An unavailable native command is a failure after retirement."""
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    assert "not yet available on v1 engine" not in (output or ""), output


# ---------------------------------------------------------------------------
# Canonical availability flags.
#
# Real try/import (not importlib.util.find_spec) wherever the original
# per-file definitions did a real import: find_spec would report an installed
# but broken package as available, changing skip behavior.
# ---------------------------------------------------------------------------

try:
    import cuequivariance as cue  # noqa: F401  # pylint: disable=unused-import

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

try:
    import cuequivariance_torch as cuet  # noqa: F401  # pylint: disable=unused-import

    CUET_OPS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUET_OPS_AVAILABLE = False

try:
    import openequivariance as oeq  # noqa: F401  # pylint: disable=unused-import

    OEQ_AVAILABLE = True
except ImportError:
    OEQ_AVAILABLE = False

try:
    import graph_longrange  # noqa: F401  # pylint: disable=unused-import

    GRAPH_LONGRANGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GRAPH_LONGRANGE_AVAILABLE = False

try:
    import torch_sim as ts  # noqa: F401  # pylint: disable=unused-import

    TORCHSIM_AVAILABLE = True
except ImportError:
    TORCHSIM_AVAILABLE = False

# These were only ever probed via find_spec (test_maceles.py / conftest.py).
LES_AVAILABLE = importlib.util.find_spec("les") is not None
SCHEDULEFREE_AVAILABLE = importlib.util.find_spec("schedulefree") is not None

CUDA_AVAILABLE = torch.cuda.is_available()


def make_fitting_configs():
    """Canonical water fitting data (same construction, seed and RNG order as
    the ``fitting_configs`` fixture in tests/conftest.py, which delegates
    here). Plain function so session-scoped fixtures can use it too."""
    import numpy as np  # pylint: disable=import-outside-toplevel
    from ase.atoms import Atoms  # pylint: disable=import-outside-toplevel

    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    fit_configs = [
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3),
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3),
    ]
    fit_configs[0].info["REF_energy"] = 0.0
    fit_configs[0].info["config_type"] = "IsolatedAtom"
    fit_configs[1].info["REF_energy"] = 0.0
    fit_configs[1].info["config_type"] = "IsolatedAtom"

    np.random.seed(5)
    for _ in range(20):
        c = water.copy()
        c.positions += np.random.normal(0.1, size=c.positions.shape)
        c.info["REF_energy"] = np.random.normal(0.1)
        c.new_array("REF_forces", np.random.normal(0.1, size=c.positions.shape))
        c.info["REF_stress"] = np.random.normal(0.1, size=6)
        fit_configs.append(c)

    return fit_configs


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def run_mace_train(
    mace_params: dict,
    extra_argv: list | None = None,
    *,
    script: Path = run_train,
    check: bool = True,
    capture_output: bool = False,
    text: bool = False,
    cwd=None,
    env_extra: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run a MACE CLI script in a subprocess, as the tests historically did.

    Replicates the repeated pattern of test_run_train.py: copy ``os.environ``,
    prepend the repo to ``sys.path`` and export it as ``PYTHONPATH`` (so the
    subprocess uses the mace currently under test), build ``--k=v`` args
    (``--k`` alone for ``None`` values), and run with ``check=True``.

    The command is built as an argv list (equivalent to the historical
    ``cmd.split()`` for whitespace-free values, and identical to the list
    variants used for values that contain spaces).

    Args:
        mace_params: mapping of CLI flag name -> value (``None`` => bare flag).
        extra_argv: extra raw arguments appended after the flags.
        script: Native command adapter path (defaults to training).
        check / capture_output / text / cwd: passed to ``subprocess.run``.
        env_extra: extra environment variables set on top of the copied env.

    Returns:
        The ``subprocess.CompletedProcess``.
    """
    # make sure the script is using the mace that is currently being tested
    run_env = os.environ.copy()
    sys.path.insert(0, str(REPO_ROOT))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    if env_extra:
        run_env.update(env_extra)

    cmd = cli_command(script)
    for k, v in mace_params.items():
        if v is None:
            cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}={v}")
    if extra_argv:
        cmd += [str(a) for a in extra_argv]

    return subprocess.run(
        cmd,
        env=run_env,
        check=check,
        capture_output=capture_output,
        text=text,
        cwd=cwd,
    )
