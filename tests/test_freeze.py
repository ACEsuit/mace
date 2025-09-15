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

try:
    import cuequivariance as cue  # pylint: disable=unused-import

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"


@pytest.fixture(name="fitting_configs")
def fixture_fitting_configs():
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
        print(c.info["REF_energy"])
        c.new_array("REF_forces", np.random.normal(0.1, size=c.positions.shape))
        c.info["REF_stress"] = np.random.normal(0.1, size=6)
        fit_configs.append(c)

    return fit_configs


@pytest.fixture(name="pretraining_configs")
def fixture_pretraining_configs():
    configs = []
    for _ in range(10):
        atoms = Atoms(
            numbers=[8, 1, 1],
            positions=np.random.rand(3, 3) * 3,
            cell=[5, 5, 5],
            pbc=[True] * 3,
        )
        atoms.info["REF_energy"] = np.random.normal(0, 1)
        atoms.arrays["REF_forces"] = np.random.normal(0, 1, size=(3, 3))
        atoms.info["REF_stress"] = np.random.normal(0, 1, size=6)
        configs.append(atoms)
    configs.append(
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3),
    )
    configs.append(
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3)
    )
    configs[-2].info["REF_energy"] = -2.0
    configs[-2].info["config_type"] = "IsolatedAtom"
    configs[-1].info["REF_energy"] = -4.0
    configs[-1].info["config_type"] = "IsolatedAtom"
    return configs


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
    "max_num_epochs": 2,
    "swa": None,
    "start_swa": 5,
    "ema": None,
    "ema_decay": 0.99,
    "amsgrad": None,
    "restart_latest": None,
    "device": device,
    "seed": 5,
    "loss": "stress",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "eval_interval": 2,
}


def test_run_train_freeze(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    mace_params["loss"] = "weighted"
    mace_params["foundation_model"] = "small"
    mace_params["hidden_irreps"] = "128x0e"
    mace_params["r_max"] = 6.0
    mace_params["default_dtype"] = "float64"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    mace_params["multiheads_finetuning"] = False
    mace_params["freeze"] = 6

    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)

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
    
    print(f"Running command: {cmd}")
    p = subprocess.run(cmd.split(), env=run_env, check=True)
    assert p.returncode == 0

    calc = MACECalculator(
        model_paths=tmp_path / "MACE.model", device=device, default_dtype="float64"
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    ref_Es = [
        5.0515191462,
        2.3054579400,
        7.8241036773,
        7.0553514878,
        5.3206756801,
        8.2035107897,
        7.8436588785,
        7.3792063144,
        7.3911904609,
        7.8037639811,
        11.7320008784,
        6.3847356085,
        7.5927331920,
        5.0871388009,
        6.7783866071,
        5.6115314645,
        8.1487276026,
        6.8827813354,
        5.5037692727,
        6.3979570745,
        6.4291638326,
        6.3360278320,
    ]

    assert np.allclose(Es, ref_Es)


