"""Installed entry points preserve the native trainer's inputs and results.

The original legacy assertions remain in test_launcher_engine_parity.
Here the direct call and both console commands all train V1, with no legacy
model imports, and compare numerical artifacts rather than pickle archives.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from tests.golden.harness import FP64_CPU_REFERENCE
from tests.workflows.test_run_train_v1 import (
    needs_the_v1_engine,
    tiny_task,
    training_records,
    variant,
)


def console_script(name):
    # Use the current interpreter's installation before an unrelated PATH.
    local = Path(sys.executable).parent / name
    found = str(local) if local.is_file() else shutil.which(name)
    if found is None:
        pytest.skip(f"mace-launcher console script {name} is not installed")
    return found


def run_command(command, arguments, directory, engine):
    finished = subprocess.run(
        [*command, *arguments, "--work_dir", str(directory)],
        env={**os.environ, "MACE_ENGINE": engine},
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert finished.returncode == 0, finished.stdout + finished.stderr


@pytest.fixture(scope="module", params=["flags", "config"])
def direct_run(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"launcher {request.param}")
    config = tiny_task(directory)
    if request.param == "config":
        config = variant(config, {
            "runtime.name": "overridden", "runtime.seed": 99,
            "training.max_num_epochs": 9,
        })
        arguments = ["--config", str(config)]
    else:
        arguments = [
            "--train_file", str(directory / "train.xyz"), "--E0s", "isolated",
            "--num_channels", "4", "--max_L", "1", "--max_ell", "1",
            "--correlation", "2", "--batch_size", "4", "--valid_batch_size", "4",
            "--plot", "false", "--default_dtype", "float64",
        ]
    arguments += ["--name", "tiny", "--seed", "7", "--max_num_epochs", "2"]
    direct = directory / "direct"
    run_command(
        [sys.executable, "-c",
         "from mace_torch.cli.run_train import main; raise SystemExit(main())"],
        arguments, direct, "legacy",
    )
    return directory, direct, arguments


@needs_the_v1_engine
@pytest.mark.parametrize("script", ["mace", "mace_run_train"])
@pytest.mark.parametrize("selection", ["flag", "environment"])
def test_installed_launcher_preserves_training(direct_run, script, selection):
    from mace_torch.calculators import MACECalculator
    from safetensors.torch import load_file

    directory, direct, arguments = direct_run
    launched = directory / f"{script}-{selection}"
    command = [console_script(script)]
    if script == "mace":
        command.append("train")
    if selection == "flag":
        # An explicit engine must win over a conflicting environment value.
        command += ["--engine=v1"]
    run_command(command, arguments, launched, "legacy" if selection == "flag" else "v1")

    resolved = []
    for root in (direct, launched):
        config = json.loads((root / "tiny.json").read_text())["config"]["config"]["resolved"]
        assert config["runtime"]["work_dir"] == str(root)
        assert config["runtime"]["name"] == "tiny"
        assert config["runtime"]["seed"] == 7
        assert config["training"]["max_num_epochs"] == 2
        config["runtime"].pop("work_dir")
        resolved.append(config)
    assert resolved[0] == resolved[1]
    expected_records, actual_records = training_records(direct), training_records(launched)
    assert len(actual_records) == len(expected_records)
    for actual_record, expected_record in zip(actual_records, expected_records, strict=True):
        assert actual_record == pytest.approx(
            expected_record, abs=FP64_CPU_REFERENCE.atol, rel=FP64_CPU_REFERENCE.rtol
        )

    expected = load_file(direct / "tiny.safetensors")
    actual = load_file(launched / "tiny.safetensors")
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(
            actual[key], expected[key],
            atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol,
        )
    atoms = read(directory / "train.xyz", index=-1)
    predictions = []
    for root in (direct, launched):
        atoms.calc = MACECalculator(root / "tiny.safetensors", device="cpu")
        predictions.append((atoms.get_potential_energy(), atoms.get_forces()))
    for actual_value, expected_value in zip(predictions[1], predictions[0], strict=True):
        np.testing.assert_allclose(
            actual_value, expected_value,
            atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol,
        )
