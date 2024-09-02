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
    "eval_interval": 2,
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
    # from a run on 04/06/2024 on stress_bugfix 967f0bfb6490086599da247874b24595d149caa7
    ref_Es = [
        0.0,
        0.0,
        -0.039181344585828524,
        -0.0915223395136733,
        -0.14953484236456582,
        -0.06662480820063998,
        -0.09983737353050133,
        0.12477442296789745,
        -0.06486086271762856,
        -0.1460607988519944,
        0.12886334908465508,
        -0.14000990081920373,
        -0.05319886578958313,
        0.07780520158391,
        -0.08895480281886901,
        -0.15474719614734422,
        0.007756765146527644,
        -0.044879267197498685,
        -0.036065736712447574,
        -0.24413743841886623,
        -0.0838104612106429,
        -0.14751978636626545,
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
    # from a run on 04/06/2024 on stress_bugfix 967f0bfb6490086599da247874b24595d149caa7
    ref_Es = [
        0.0,
        0.0,
        -0.05464025113696155,
        -0.11272131295940478,
        0.039200919331076826,
        -0.07517990972827505,
        -0.13504202474582666,
        0.0292022872055344,
        -0.06541099574579018,
        -0.1497824717832886,
        0.19397709360828813,
        -0.13587609467143014,
        -0.05242956276828463,
        -0.0504862057364953,
        -0.07095795959430119,
        -0.2463753796753703,
        -0.002031543147676121,
        -0.03864918790300681,
        -0.13680153117705554,
        -0.23418951968636786,
        -0.11790833839379238,
        -0.14930562311066484,
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


def test_run_train_multihead(tmp_path, fitting_configs):
    fitting_configs_dft = []
    fitting_configs_mp2 = []
    fitting_configs_ccd = []
    for _, c in enumerate(fitting_configs):
        c_dft = c.copy()
        c_dft.info["head"] = "DFT"
        fitting_configs_dft.append(c_dft)

        c_mp2 = c.copy()
        c_mp2.info["head"] = "MP2"
        fitting_configs_mp2.append(c_mp2)

        c_ccd = c.copy()
        c_ccd.info["head"] = "CCD"
        fitting_configs_ccd.append(c_ccd)
    ase.io.write(tmp_path / "fit_multihead_dft.xyz", fitting_configs_dft)
    ase.io.write(tmp_path / "fit_multihead_mp2.xyz", fitting_configs_mp2)
    ase.io.write(tmp_path / "fit_multihead_ccd.xyz", fitting_configs_ccd)

    heads = {
        "DFT": {"train_file": f"{str(tmp_path)}/fit_multihead_dft.xyz"},
        "MP2": {"train_file": f"{str(tmp_path)}/fit_multihead_mp2.xyz"},
        "CCD": {"train_file": f"{str(tmp_path)}/fit_multihead_ccd.xyz"},
    }
    yaml_str = "heads:\n"
    for key, value in heads.items():
        yaml_str += f"  {key}:\n"
        for sub_key, sub_value in value.items():
            yaml_str += f"    {sub_key}: {sub_value}\n"
    filename = tmp_path / "config.yaml"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(yaml_str)

    mace_params = _mace_params.copy()
    mace_params["valid_fraction"] = 0.1
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["loss"] = "weighted"
    mace_params["hidden_irreps"] = "128x0e"
    mace_params["r_max"] = 6.0
    mace_params["default_dtype"] = "float64"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    mace_params["config"] = tmp_path / "config.yaml"
    mace_params["batch_size"] = 2
    mace_params["num_samples_pt"] = 50
    mace_params["subselect_pt"] = "random"
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
        tmp_path / "MACE.model", device="cpu", default_dtype="float64"
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)
    # from a run on 02/09/2024 on develop branch
    ref_Es = [
        0.0,
        0.0,
        0.10637113905361611,
        -0.012499594026624754,
        0.08983077108171753,
        0.21071322543112597,
        -0.028921849222784398,
        -0.02423359575741567,
        0.022923252188079057,
        -0.02048334610058991,
        0.4349711162741364,
        -0.04455577015569887,
        -0.09765806785570091,
        0.16013134616829822,
        0.0758442928017698,
        -0.05931856557011721,
        0.33964473532953265,
        0.134338442158641,
        0.18024119757783053,
        -0.18914740992058765,
        -0.06503477155294624,
        0.03436649147415213,
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
    mace_params["default_dtype"] = "float64"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    mace_params["multiheads_finetuning"] = False
    print("mace_params", mace_params)
    # mace_params["num_samples_pt"] = 50
    # mace_params["subselect_pt"] = "random"
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
        tmp_path / "MACE.model", device="cpu", default_dtype="float64"
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)
    # from a run on 28/03/2023 on repulsion a63434aaab70c84ee016e13e4aca8d57297a0f26
    ref_Es = [
        1.6780993938446045,
        0.8916864395141602,
        0.7290308475494385,
        0.6194742918014526,
        0.6697757840156555,
        0.7025266289710999,
        0.5818213224411011,
        0.7897703647613525,
        0.6558921337127686,
        0.5071806907653809,
        3.581131935119629,
        0.691562294960022,
        0.6257331967353821,
        0.9560437202453613,
        0.7716934680938721,
        0.6730310916900635,
        0.8297463655471802,
        0.8053972721099854,
        0.8337507247924805,
        0.4107491970062256,
        0.6019601821899414,
        0.7301387786865234,
    ]
    assert np.allclose(Es, ref_Es)


def test_run_train_foundation_multihead(tmp_path, fitting_configs):
    fitting_configs_dft = []
    fitting_configs_mp2 = []
    for i, c in enumerate(fitting_configs):
        if i in (0, 1):
            c_dft = c.copy()
            c_dft.info["head"] = "DFT"
            fitting_configs_dft.append(c_dft)
            fitting_configs_dft.append(c)
            c_mp2 = c.copy()
            c_mp2.info["head"] = "MP2"
            fitting_configs_mp2.append(c_mp2)
        elif i % 2 == 0:
            c.info["head"] = "DFT"
            fitting_configs_dft.append(c)
        else:
            c.info["head"] = "MP2"
            fitting_configs_mp2.append(c)
    ase.io.write(tmp_path / "fit_multihead_dft.xyz", fitting_configs_dft)
    ase.io.write(tmp_path / "fit_multihead_mp2.xyz", fitting_configs_mp2)

    heads = {
        "DFT": {"train_file": f"{str(tmp_path)}/fit_multihead_dft.xyz"},
        "MP2": {"train_file": f"{str(tmp_path)}/fit_multihead_mp2.xyz"},
    }
    yaml_str = "heads:\n"
    for key, value in heads.items():
        yaml_str += f"  {key}:\n"
        for sub_key, sub_value in value.items():
            yaml_str += f"    {sub_key}: {sub_value}\n"
    filename = tmp_path / "config.yaml"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(yaml_str)
    mace_params = _mace_params.copy()
    mace_params["valid_fraction"] = 0.1
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["config"] = tmp_path / "config.yaml"
    mace_params["loss"] = "weighted"
    mace_params["foundation_model"] = "small"
    mace_params["hidden_irreps"] = "128x0e"
    mace_params["r_max"] = 6.0
    mace_params["default_dtype"] = "float64"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    mace_params["batch_size"] = 2
    mace_params["valid_batch_size"] = 1
    mace_params["num_samples_pt"] = 50
    mace_params["subselect_pt"] = "random"
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
        tmp_path / "MACE.model", device="cpu", default_dtype="float64"
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)
    # from a run on 20/08/2024 on commit
    ref_Es = [
        1.654685616493225,
        0.44693732261657715,
        0.8741313815116882,
        0.569085955619812,
        0.7161882519721985,
        0.8654778599739075,
        0.8722733855247498,
        0.49582308530807495,
        0.814422607421875,
        0.7027317881584167,
        0.7196993827819824,
        0.517953097820282,
        0.8631765246391296,
        0.4679797887802124,
        0.8163984417915344,
        0.4252359867095947,
        1.0861445665359497,
        0.6829671263694763,
        0.7136879563331604,
        0.5160345435142517,
        0.7002358436584473,
        0.5574042201042175,
    ]
    assert np.allclose(Es, ref_Es, atol=1e-1)
