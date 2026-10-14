"""Train end-to-end with ``batch_size == 1`` so every train/valid batch exercises the
``num_graphs == 1`` fast path in ``get_symmetric_displacement`` through the full
forward + backward + optimizer loop (loss="stress").

This complements ``test_stress_single_graph.py`` (which only checks the forward pass):
here we confirm the new branch also backpropagates and trains without error / NaNs.

NOTE: like the other subprocess-based training tests in this repo, this passes on
WSL/Linux but is known to fail on native Windows. Run it on WSL.
"""

import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from ase.atoms import Atoms

from mace.calculators import MACECalculator

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_TRAIN = REPO_ROOT / "mace" / "cli" / "run_train.py"


@pytest.fixture(name="fitting_configs")
def fixture_fitting_configs():
    """Periodic water configs (+ isolated atoms) carrying energy/forces/stress."""
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


_mace_params = {
    "name": "MACE",
    "valid_fraction": 0.1,
    "energy_weight": 1.0,
    "forces_weight": 10.0,
    "stress_weight": 1.0,
    "model": "MACE",
    "hidden_irreps": "16x0e",
    "r_max": 3.5,
    "batch_size": 1,  # <- forces num_graphs == 1 on every training step
    "valid_batch_size": 1,  # <- and on every validation step
    "max_num_epochs": 5,
    "seed": 5,
    "loss": "stress",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "eval_interval": 1,
}


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="no CUDA device"
            ),
        ),
    ],
)
def test_run_train_stress_batch_size_1(tmp_path, fitting_configs, device):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    params = _mace_params.copy()
    params["device"] = device
    params["checkpoints_dir"] = str(tmp_path)
    params["model_dir"] = str(tmp_path)
    params["train_file"] = str(tmp_path / "fit.xyz")

    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + run_env.get("PYTHONPATH", "")
    run_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    cmd = [sys.executable, str(RUN_TRAIN)] + [
        f"--{k}={v}" if v is not None else f"--{k}" for k, v in params.items()
    ]
    proc = subprocess.run(cmd, env=run_env, check=False)
    assert proc.returncode == 0, "batch_size=1 stress training crashed"

    # The trained model must load and produce finite energies (no NaNs from the branch).
    model_path = tmp_path / "MACE.model"
    assert model_path.exists(), "no model produced"
    calc = MACECalculator(model_paths=model_path, device="cpu")
    energies = []
    for at in fitting_configs:
        at.calc = calc
        energies.append(at.get_potential_energy())
    assert np.all(np.isfinite(energies)), f"non-finite energies: {energies}"
