import sys
import os
import subprocess

from pathlib import Path
pytest_mace_dir = Path(__file__).parent.parent
run_train = Path(__file__).parent.parent / "scripts" / "run_train.py"

import numpy as np

import ase.io
from ase.atoms import Atoms

from mace.calculators.mace import MACECalculator

def test_run_train(tmp_path):
    water = Atoms(numbers=[8, 1, 1], positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]], cell=[4]*3, pbc = [True]*3)
    fit_configs = [Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6]*3),
               Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6]*3)]
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
    ase.io.write(tmp_path / "fit.xyz", fit_configs)

    mace_params = {
        "name": "MACE",
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "train_file": tmp_path / "fit.xyz",
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
        "stress_key": "REF_stress"
    }

    # make sure run_train.py is using the mace that is currently being tested
    run_env = os.environ
    run_env["PYTHONPATH"] = str(pytest_mace_dir) + ":" + os.environ["PYTHONPATH"]

    cmd = sys.executable + " " + str(run_train) + " " + " ".join([(f"--{k}={v}" if v is not None else f"--{k}") for k, v in mace_params.items()])

    p = subprocess.run(cmd.split(), env=run_env)

    assert p.returncode == 0

    calc = MACECalculator(tmp_path / "MACE.model", device="cpu")
