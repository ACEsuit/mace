import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.test import gradient_test
from ase.constraints import ExpCellFilter

from mace.calculators.mace import MACECalculator

pytest_mace_dir = Path(__file__).parent.parent
run_train = Path(__file__).parent.parent / "scripts" / "run_train.py"


@pytest.fixture(scope="module", name="fitting_configs")
def fitting_configs_fixture():
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


@pytest.fixture(scope="module", name="trained_model")
def trained_model_fixture(tmp_path_factory, fitting_configs):
    _mace_params = {
        "name": "MACE",
        "valid_fraction": 0.05,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "MACE",
        "hidden_irreps": "128x0e",
        "r_max": 3.5,
        "batch_size": 5,
        "max_num_epochs": 10,
        "swa": None,
        "start_swa": 5,
        "ema": None,
        "ema_decay": 0.99,
        "amsgrad": None,
        "restart_latest": None,
        "device": "cpu",
        "seed": 5,
        "loss": "stress",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
    }

    tmp_path = tmp_path_factory.mktemp("run_")

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    # make sure run_train.py is using the mace that is currently being tested
    run_env = os.environ
    run_env["PYTHONPATH"] = str(pytest_mace_dir) + ":" + os.environ["PYTHONPATH"]

    cmd = (
        sys.executable
        + " "
        + str(run_train)
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), env=run_env, check=True)

    assert p.returncode == 0

    return MACECalculator(tmp_path / "MACE.model", device="cpu")


def test_calculator_forces(fitting_configs, trained_model):
    at = fitting_configs[2].copy()
    at.calc = trained_model

    # test just forces
    grads = gradient_test(at)

    assert np.allclose(grads[0], grads[1])


def test_calculator_stress(fitting_configs, trained_model):
    at = fitting_configs[2].copy()
    at.calc = trained_model

    # test forces and stress
    at_wrapped = ExpCellFilter(at)
    grads = gradient_test(at_wrapped)

    assert np.allclose(grads[0], grads[1])
