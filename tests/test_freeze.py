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
    "max_num_epochs": 10,
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


def test_run_train_freeze_par(tmp_path, fitting_configs):
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
    mace_params["freeze_par"] = 9

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
        3.4404297977,
        1.7308105589,
        3.8779711236,
        3.8445259524,
        3.3526783077,
        3.9690051137,
        3.9487003864,
        2.8296044219,
        3.8706612453,
        3.9819326697,
        9.8389733374,
        3.7397551274,
        3.8149238693,
        3.2222994681,
        3.7686655925,
        3.4685892606,
        4.0117956515,
        3.7946286923,
        3.3714840691,
        3.6015980979,
        3.6587433162,
        3.6195737863,
    ]

    assert np.allclose(Es, ref_Es)


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


@pytest.mark.skipif(
    not CUET_AVAILABLE or device != "cuda",
    reason="cuequivariance not installed or cuda not available",
)
def test_run_train_foundation_freeze_cueq(tmp_path, fitting_configs):
    torch.set_default_dtype(torch.float64)
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
    mace_params["freeze_par"] = 8
    mace_params["enable_cueq"] = True

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
        3.4404297977,
        1.7308105589,
        3.8779711236,
        3.8445259524,
        3.3526783077,
        3.9690051137,
        3.9487003864,
        2.8296044219,
        3.8706612453,
        3.9819326697,
        9.8389733374,
        3.7397551274,
        3.8149238693,
        3.2222994681,
        3.7686655925,
        3.4685892606,
        4.0117956515,
        3.7946286923,
        3.3714840691,
        3.6015980979,
        3.6587433162,
        3.6195737863,
    ]

    assert np.allclose(Es, ref_Es, atol=1e-1)


@pytest.mark.skipif(
    not CUET_AVAILABLE or device != "cuda",
    reason="cuequivariance not installed or cuda not available",
)
def test_run_train_soft_freeze_cueq(tmp_path, fitting_configs):
    torch.set_default_dtype(torch.float64)
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
    mace_params["enable_cueq"] = True
    mace_params["freeze_par"] = 8
    mace_params["soft_freeze"] = 1
    mace_params["soft_freeze_factor"] = 0.1
    mace_params["soft_freeze_swa"] = None

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
        4.7097111585,
        2.1657316953,
        4.5721787830,
        4.4974878306,
        3.7895048208,
        4.5641742601,
        4.5254280041,
        3.6814502951,
        4.4947306918,
        4.4938875957,
        11.0660592534,
        4.3791088182,
        4.4610165341,
        4.4047065188,
        4.4151594051,
        4.1548401290,
        4.6304058745,
        4.4546468493,
        4.4308637268,
        4.2262612843,
        4.4208466570,
        4.2547575544,
    ]

    assert np.allclose(Es, ref_Es, atol=1e-1)


def test_run_train_soft_freeze(tmp_path, fitting_configs):
    torch.set_default_dtype(torch.float64)
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
    mace_params["enable_cueq"] = False
    mace_params["freeze_par"] = 9
    mace_params["soft_freeze"] = 1
    mace_params["soft_freeze_factor"] = 0.1
    mace_params["soft_freeze_swa"] = None

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
        4.7097111585,
        2.1657316953,
        4.5721787830,
        4.4974878306,
        3.7895048208,
        4.5641742601,
        4.5254280041,
        3.6814502951,
        4.4947306918,
        4.4938875957,
        11.0660592534,
        4.3791088182,
        4.4610165341,
        4.4047065188,
        4.4151594051,
        4.1548401290,
        4.6304058745,
        4.4546468493,
        4.4308637268,
        4.2262612843,
        4.4208466570,
        4.2547575544,
    ]

    assert np.allclose(Es, ref_Es, atol=1e-1)


def test_run_train_multihead_soft_freeze(tmp_path, fitting_configs):
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
    mace_params["soft_freeze"] = 2
    mace_params["soft_freeze_factor"] = 0.1

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
        model_paths=tmp_path / "MACE.model", device=device, default_dtype="float64"
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    ref_Es = [
        0.0000000000,
        0.0000000000,
        0.0347944589,
        -0.0056869672,
        -0.1099380035,
        0.1345284345,
        -0.0786784911,
        -0.1064854434,
        0.0800015489,
        -0.0523257851,
        0.2412342945,
        -0.0333065080,
        -0.0648696150,
        0.2572237230,
        0.0701157782,
        0.0110373787,
        0.2518143028,
        0.1114972901,
        0.2267955514,
        -0.1615110818,
        -0.0524467197,
        0.0989867308,
    ]
    assert np.allclose(Es, ref_Es)


@pytest.mark.skipif(
    not CUET_AVAILABLE or device != "cuda",
    reason="cuequivariance not installed or cuda not available",
)
def test_run_train_soft_freeze_noswa_cueq(tmp_path, fitting_configs):
    torch.set_default_dtype(torch.float64)
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
    mace_params["enable_cueq"] = True
    mace_params["freeze_par"] = 8
    mace_params["soft_freeze"] = 1
    mace_params["soft_freeze_factor"] = 0.1

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
        4.7097111585,
        2.1657316953,
        4.5721787830,
        4.4974878306,
        3.7895048208,
        4.5641742601,
        4.5254280041,
        3.6814502951,
        4.4947306918,
        4.4938875957,
        11.0660592534,
        4.3791088182,
        4.4610165341,
        4.4047065188,
        4.4151594051,
        4.1548401290,
        4.6304058745,
        4.4546468493,
        4.4308637268,
        4.2262612843,
        4.4208466570,
        4.2547575544,
    ]

    assert np.allclose(Es, ref_Es, atol=1e-1)
