import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms

from mace.calculators.mace import MACECalculator

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


def test_run_train(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    # make sure run_train.py is using the mace that is currently being tested
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

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

    calc = MACECalculator(tmp_path / "MACE.model", device="cpu")

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)
    # from a run on 28/03/2023 on main 88d49f9ed6925dec07d1777043a36e1fe4872ff3
    ref_Es = [
        0.0,
        0.0,
        -0.03911274694160493,
        -0.0913651377675312,
        -0.14973695873658766,
        -0.0664839502025434,
        -0.09968814898703926,
        0.1248460531971883,
        -0.0647495831154953,
        -0.14589298347245963,
        0.12918668431788108,
        -0.13996496272772996,
        -0.053211348522482806,
        0.07845141245421094,
        -0.08901520083723416,
        -0.15467129065263446,
        0.007727727865546765,
        -0.04502061132025605,
        -0.035848783030374,
        -0.24410687104937906,
        -0.0839034724949955,
        -0.14756571357354326,
    ]

    assert np.allclose(Es, ref_Es)


def test_run_train_missing_data(tmp_path, fitting_configs):
    del fitting_configs[5].info["REF_energy"]
    del fitting_configs[6].arrays["REF_forces"]
    del fitting_configs[7].info["REF_stress"]

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    # make sure run_train.py is using the mace that is currently being tested
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

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

    calc = MACECalculator(tmp_path / "MACE.model", device="cpu")

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)
    # from a run on 28/03/2023 on main 88d49f9ed6925dec07d1777043a36e1fe4872ff3
    ref_Es = [
        0.0,
        0.0,
        -0.05449966431966507,
        -0.11237663925685797,
        0.03914539466246801,
        -0.07500800414261456,
        -0.13471106701173396,
        0.02937255038020199,
        -0.0652196693921633,
        -0.14946129637190012,
        0.19412338220281133,
        -0.13546947741234333,
        -0.05235148626886153,
        -0.04957190959243316,
        -0.07081384032242896,
        -0.24575839901841345,
        -0.0020512332640394916,
        -0.038630330106902526,
        -0.13621347044601181,
        -0.2338465954158298,
        -0.11777474787291177,
        -0.14895508008918812,
    ]
    assert np.allclose(Es, ref_Es)


def test_run_train_no_stress(tmp_path, fitting_configs):
    del fitting_configs[5].info["REF_energy"]
    del fitting_configs[6].arrays["REF_forces"]
    del fitting_configs[7].info["REF_stress"]

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    mace_params["loss"] = "weighted"

    # make sure run_train.py is using the mace that is currently being tested
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

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

    calc = MACECalculator(tmp_path / "MACE.model", device="cpu")

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)
    # from a run on 28/03/2023 on main 88d49f9ed6925dec07d1777043a36e1fe4872ff3
    ref_Es = [
        0.0,
        0.0,
        -0.05450093218377135,
        -0.11235475232750518,
        0.03914558031854152,
        -0.07500839914816063,
        -0.13469160624431492,
        0.029384214243251838,
        -0.06521819204166135,
        -0.14944896282001804,
        0.19413948083049481,
        -0.13543541860473626,
        -0.05235495076237124,
        -0.049556206595684105,
        -0.07080758913030646,
        -0.24571898386301153,
        -0.002070636306950905,
        -0.03863113401320783,
        -0.13620291339913712,
        -0.23383074855679695,
        -0.11776449630199368,
        -0.1489441490225184,
    ]
    assert np.allclose(Es, ref_Es)


def test_run_train_foundation(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    mace_params["loss"] = "weighted"
    mace_params["foundation_model"] = "small"
    mace_params["hidden_irreps"] = "128x0e"
    mace_params["r_max"] = 6.0
    mace_params["default_dtype"] = "float32"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    # make sure run_train.py is using the mace that is currently being tested
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

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

    calc = MACECalculator(
        tmp_path / "MACE.model", device="cpu", default_dtype="float32"
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)
    # from a run on 28/03/2023 on repulsion a63434aaab70c84ee016e13e4aca8d57297a0f26
    ref_Es = [
        1.6324548721313477,
        0.8016132712364197,
        0.5674604177474976,
        0.4571378827095032,
        0.5002979040145874,
        0.5423622131347656,
        0.4228057265281677,
        0.6198095083236694,
        0.4928140640258789,
        0.34875422716140747,
        3.448421001434326,
        0.5291348695755005,
        0.463774710893631,
        0.7975829243659973,
        0.6095817685127258,
        0.5113707780838013,
        0.6703207492828369,
        0.6445103883743286,
        0.6734066605567932,
        0.25074106454849243,
        0.44250693917274475,
        0.5659750699996948,
    ]
    assert np.allclose(Es, ref_Es)
