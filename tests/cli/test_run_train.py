from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
import argparse

import ase.io
import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import numpy.testing as npt

from mace.cli.run_train import run
from mace.tools import build_default_arg_parser, dict_to_arg_list

from mace.calculators import MACECalculator, mace_mp

try:
    import cuequivariance as cue  # pylint: disable=unused-import

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

run_train = "mace_run_train"


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
    "device": "cpu",
    "default_dtype": "float64",
    "seed": 5,
    "loss": "stress",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "eval_interval": 2,
    "use_reduced_cg": False,
}


def test_run_train(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
    mace_params["train_file"] = (tmp_path / "fit.xyz").as_posix()

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), check=True)
    assert p.returncode == 0

    calc = MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        default_dtype=mace_params["default_dtype"],
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    # UPDATED alongside https://github.com/ACEsuit/mace/pull/1009
    # fmt: off
    ref_Es = [0.0, 0.0, -0.05210175452593896, -0.13921025881568108, -0.4368689292044523, -0.03620908697361255, -0.0958774337443896, -0.03642912472019945, -0.09061914090441943, -0.12225003887490886, 0.11996333228856383, -0.23439758090640703, -0.0811642196082918, 0.08651948518824783, -0.16392247064962123, -0.27488954640817853, 0.04069277568329649, -0.11950528650871345, -0.04837193044162801, -0.35980607108994644, -0.26800821789049123, -0.21034632802301417]
    # fmt: on

    npt.assert_allclose(Es, ref_Es)


def test_run_train_direct(tmp_path, fitting_configs):
    """This test is here to make debugging easier as it doesn't rely on the
    script being run through a subprocess.
    """
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
    mace_params["train_file"] = (tmp_path / "fit.xyz").as_posix()

    args = build_default_arg_parser().parse_args(dict_to_arg_list(mace_params))
    run(args)

    calc = MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        default_dtype=mace_params["default_dtype"],
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    # UPDATED alongside https://github.com/ACEsuit/mace/pull/1009
    # fmt: off
    ref_Es = [0.0, 0.0, -0.04925730922316317, -0.1744891106147678, -0.3382783979033222, -0.0857094093217958, -0.19675216058999634, -0.26148257144509635, -0.0770317936486534, -0.2270164136195037, 0.36233587804844036, -0.21577922781964037, -0.040603275127259295, -0.164272699472646, -0.055454711648923266, -0.2800495778279926, 0.04013590514963973, 0.009409736484142048, -0.16900045017526188, -0.3867431975751795, -0.12000373096477862, -0.16618337674705164]
    # fmt: on

    npt.assert_allclose(Es, ref_Es)


def test_run_train_missing_data(tmp_path, fitting_configs):
    del fitting_configs[5].info["REF_energy"]
    del fitting_configs[6].arrays["REF_forces"]
    del fitting_configs[7].info["REF_stress"]

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
    mace_params["train_file"] = (tmp_path / "fit.xyz").as_posix()

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), check=True)
    assert p.returncode == 0

    calc = MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        default_dtype=mace_params["default_dtype"],
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    # UPDATED alongside https://github.com/ACEsuit/mace/pull/1009
    # fmt: off
    ref_Es = [0.0, 0.0, -0.05597474180398447, -0.1475678497763891, -0.41851703349950153, -0.03133883290145706, -0.10089924152222293, -0.04796868889210438, -0.09163031738012561, -0.11998700223711734, 0.14230460152331958, -0.24340107725799592, -0.08496106755112191, 0.08558057489791332, -0.16361880003616727, -0.290966170935295, 0.043955136278508715, -0.12156731206706181, -0.053117788272777536, -0.37142428603800326, -0.28579033985842817, -0.2139733015636862]
    # fmt: on

    npt.assert_allclose(Es, ref_Es)


def test_run_train_no_stress(tmp_path, fitting_configs):
    del fitting_configs[5].info["REF_energy"]
    del fitting_configs[6].arrays["REF_forces"]
    del fitting_configs[7].info["REF_stress"]

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
    mace_params["train_file"] = (tmp_path / "fit.xyz").as_posix()
    mace_params["loss"] = "weighted"

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), check=True)
    assert p.returncode == 0

    calc = MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        default_dtype=mace_params["default_dtype"],
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    # UPDATED alongside https://github.com/ACEsuit/mace/pull/1009
    # fmt: off
    ref_Es = [0.0, 0.0, -0.055621940428552084, -0.1470964738905481, -0.4184802258796809, -0.030918426561104988, -0.10062074574998127, -0.04703537753523078, -0.09121952814954548, -0.11963694775585104, 0.14274856155117183, -0.24331360917107508, -0.08488502355750432, 0.08616729690005585, -0.1636299853338604, -0.2911483089378377, 0.043774583910522524, -0.12183399473945355, -0.05317636152855498, -0.37165293676508715, -0.2860931083551229, -0.21411508158504922]
    # fmt: on

    npt.assert_allclose(Es, ref_Es)


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
        "DFT": {"train_file": (tmp_path / "fit_multihead_dft.xyz").as_posix()},
        "MP2": {"train_file": (tmp_path / "fit_multihead_mp2.xyz").as_posix()},
        "CCD": {"train_file": (tmp_path / "fit_multihead_ccd.xyz").as_posix()},
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
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
    mace_params["loss"] = "weighted"
    mace_params["hidden_irreps"] = "128x0e"
    mace_params["r_max"] = 6.0
    mace_params["default_dtype"] = "float64"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    mace_params["config"] = (tmp_path / "config.yaml").as_posix()
    mace_params["batch_size"] = 2
    mace_params["num_samples_pt"] = 50
    mace_params["subselect_pt"] = "random"

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), check=True)
    assert p.returncode == 0

    calc = MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        default_dtype="float64",
        head="CCD",
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    # UPDATED alongside https://github.com/ACEsuit/mace/pull/1009
    # fmt: off
    ref_Es = [0.0, 0.0, 0.14927263651219513, 0.12760446172324147, 0.18094769008792522, 0.2017525535131856, 0.0947377232202872, 0.20055465235975933, 0.16739670729809553, 0.1053607542696717, 0.2917874651133893, 0.06670614088927873, 0.09735984475039591, 0.23458709864804722, 0.09877470521444243, -0.02295768274710428, 0.27387224799962095, 0.13694316838311232, 0.12737608108478338, -0.07650972963159812, -0.012938743388744223, 0.06122839183498221]
    # fmt: on

    npt.assert_allclose(Es, ref_Es)


def test_run_train_foundation(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
    mace_params["train_file"] = tmp_path / "fit.xyz"
    mace_params["loss"] = "weighted"
    mace_params["foundation_model"] = "small"
    mace_params["hidden_irreps"] = "128x0e"
    mace_params["r_max"] = 6.0
    mace_params["default_dtype"] = "float64"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    mace_params["multiheads_finetuning"] = False

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), check=True)
    assert p.returncode == 0

    calc = MACECalculator(
        model_paths=tmp_path / "MACE.model", device="cpu", default_dtype="float64"
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    # from a run on 28/03/2023 on repulsion a63434aaab70c84ee016e13e4aca8d57297a0f26
    # fmt: off
    ref_Es = [1.6780993938446045, 0.8916864395141602, 0.7290308475494385, 0.6194742918014526, 0.6697757840156555, 0.7025266289710999, 0.5818213224411011, 0.7897703647613525, 0.6558921337127686, 0.5071806907653809, 3.581131935119629, 0.691562294960022, 0.6257331967353821, 0.9560437202453613, 0.7716934680938721, 0.6730310916900635, 0.8297463655471802, 0.8053972721099854, 0.8337507247924805, 0.4107491970062256, 0.6019601821899414, 0.7301387786865234]
    # fmt: on

    npt.assert_allclose(Es, ref_Es, atol=1e-1)


def test_run_train_foundation_multihead(tmp_path, fitting_configs):
    fitting_configs_dft = []
    fitting_configs_mp2 = []
    atomic_numbers = np.unique(
        np.concatenate([at.numbers for at in fitting_configs])
    ).tolist()
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
        "DFT": {"train_file": (tmp_path / "fit_multihead_dft.xyz").as_posix()},
        "MP2": {"train_file": (tmp_path / "fit_multihead_mp2.xyz").as_posix()},
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
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
    mace_params["config"] = (tmp_path / "config.yaml").as_posix()
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
    mace_params["atomic_numbers"] = "[" + ",".join(map(str, atomic_numbers)) + "]"
    mace_params["filter_type_pt"] = "combinations"
    mace_params["force_mh_ft_lr"] = True

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    try:
        completed_process = subprocess.run(
            cmd.split(), capture_output=True, text=True, check=True
        )
        # Process executed successfully
        print(completed_process.stdout)
    except subprocess.CalledProcessError as e:
        # Process failed with non-zero exit code
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise e
    assert completed_process.returncode == 0

    Es = []
    for at in fitting_configs:
        config_head = at.info.get("head", "MP2")
        calc = MACECalculator(
            model_paths=tmp_path / "MACE.model",
            device="cpu",
            default_dtype="float64",
            head=config_head,
        )
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    # from a run on 20/08/2024 on commit
    # fmt: off
    ref_Es = [1.654685616493225, 0.44693732261657715, 0.8741313815116882, 0.569085955619812, 0.7161882519721985, 0.8654778599739075, 0.8722733855247498, 0.49582308530807495, 0.814422607421875, 0.7027317881584167, 0.7196993827819824, 0.517953097820282, 0.8631765246391296, 0.4679797887802124, 0.8163984417915344, 0.4252359867095947, 1.0861445665359497, 0.6829671263694763, 0.7136879563331604, 0.5160345435142517, 0.7002358436584473, 0.5574042201042175]
    # fmt: on

    npt.assert_allclose(Es, ref_Es, atol=1e-1)


def test_run_train_foundation_multihead_json(tmp_path, fitting_configs):
    fitting_configs_dft = []
    fitting_configs_mp2 = []
    atomic_numbers = np.unique(
        np.concatenate([at.numbers for at in fitting_configs])
    ).tolist()
    for i, c in enumerate(fitting_configs):
        if i in (0, 1):
            continue  # skip isolated atoms, as energies specified by json files below
        if i % 2 == 0:
            c.info["head"] = "DFT"
            fitting_configs_dft.append(c)
        else:
            c.info["head"] = "MP2"
            fitting_configs_mp2.append(c)
    ase.io.write(tmp_path / "fit_multihead_dft.xyz", fitting_configs_dft)
    ase.io.write(tmp_path / "fit_multihead_mp2.xyz", fitting_configs_mp2)

    # write E0s to json files
    E0s = {1: 0.0, 8: 0.0}
    with open(tmp_path / "fit_multihead_dft.json", "w", encoding="utf-8") as f:
        json.dump(E0s, f)
    with open(tmp_path / "fit_multihead_mp2.json", "w", encoding="utf-8") as f:
        json.dump(E0s, f)

    heads = {
        "DFT": {
            "train_file": (tmp_path / "fit_multihead_dft.xyz").as_posix(),
            "E0s": (tmp_path / "fit_multihead_dft.json").as_posix(),
        },
        "MP2": {
            "train_file": (tmp_path / "fit_multihead_mp2.xyz").as_posix(),
            "E0s": (tmp_path / "fit_multihead_mp2.json").as_posix(),
        },
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
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
    mace_params["config"] = (tmp_path / "config.yaml").as_posix()
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
    mace_params["atomic_numbers"] = "[" + ",".join(map(str, atomic_numbers)) + "]"
    mace_params["filter_type_pt"] = "combinations"
    mace_params["force_mh_ft_lr"] = True

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    try:
        completed_process = subprocess.run(
            cmd.split(), capture_output=True, text=True, check=True
        )
        # Process executed successfully
        print(completed_process.stdout)
    except subprocess.CalledProcessError as e:
        # Process failed with non-zero exit code
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise e
    assert completed_process.returncode == 0

    Es = []
    for at in fitting_configs:
        config_head = at.info.get("head", "MP2")
        calc = MACECalculator(
            model_paths=tmp_path / "MACE.model",
            device="cpu",
            default_dtype="float64",
            head=config_head,
        )
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    # from a run on 20/08/2024 on commit
    # fmt: off
    ref_Es = [1.654685616493225, 0.44693732261657715, 0.8741313815116882, 0.569085955619812, 0.7161882519721985, 0.8654778599739075, 0.8722733855247498, 0.49582308530807495, 0.814422607421875, 0.7027317881584167, 0.7196993827819824, 0.517953097820282, 0.8631765246391296, 0.4679797887802124, 0.8163984417915344, 0.4252359867095947, 1.0861445665359497, 0.6829671263694763, 0.7136879563331604, 0.5160345435142517, 0.7002358436584473, 0.5574042201042175]
    # fmt: on

    npt.assert_allclose(Es, ref_Es, atol=1e-1)


def test_run_train_multihead_replay_custom_finetuning(
    tmp_path, fitting_configs, pretraining_configs
):
    ase.io.write(tmp_path / "pretrain.xyz", pretraining_configs)

    foundation_params = {
        "name": "foundation",
        "train_file": os.path.join(tmp_path, "pretrain.xyz"),
        "valid_fraction": 0.2,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "MACE",
        "hidden_irreps": "32x0e",
        "r_max": 5.0,
        "batch_size": 2,
        "max_num_epochs": 5,
        "swa": None,
        "start_swa": 3,
        "device": "cpu",
        "seed": 42,
        "loss": "weighted",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
        "default_dtype": "float64",
        "checkpoints_dir": (tmp_path).as_posix(),
        "model_dir": (tmp_path).as_posix(),
    }

    cmd = [run_train]
    for k, v in foundation_params.items():
        if v is None:
            cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}={v}")

    p = subprocess.run(cmd, check=True)
    assert p.returncode == 0

    # Step 3: Create finetuning set
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

    # Step 4: Finetune the pretrained model with multihead replay
    heads = {
        "DFT": {"train_file": (tmp_path / "fit_multihead_dft.xyz").as_posix()},
        "MP2": {"train_file": (tmp_path / "fit_multihead_mp2.xyz").as_posix()},
    }
    yaml_str = "heads:\n"
    for key, value in heads.items():
        yaml_str += f"  {key}:\n"
        for sub_key, sub_value in value.items():
            yaml_str += f"    {sub_key}: {sub_value}\n"
    filename = tmp_path / "config.yaml"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(yaml_str)

    finetuning_params = {
        "name": "finetuned",
        "valid_fraction": 0.1,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "MACE",
        "hidden_irreps": "32x0e",
        "r_max": 5.0,
        "batch_size": 2,
        "max_num_epochs": 5,
        "device": "cpu",
        "seed": 42,
        "loss": "weighted",
        "default_dtype": "float64",
        "checkpoints_dir": (tmp_path).as_posix(),
        "model_dir": (tmp_path).as_posix(),
        "foundation_model": (tmp_path / "foundation.model").as_posix(),
        "config": (tmp_path / "config.yaml").as_posix(),
        "pt_train_file": (tmp_path / "pretrain.xyz").as_posix(),
        "num_samples_pt": 3,
        "subselect_pt": "random",
        "force_mh_ft_lr": True,
    }

    cmd = [run_train]
    for k, v in finetuning_params.items():
        if v is None:
            cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}={v}")

    p = subprocess.run(cmd, check=True)
    assert p.returncode == 0

    # Load and test the finetuned model
    calc = MACECalculator(
        model_paths=tmp_path / "finetuned.model",
        device="cpu",
        default_dtype="float64",
        head="pt_head",
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Energies:", Es)

    # Add some basic checks
    assert len(Es) == len(fitting_configs)
    assert all(isinstance(E, float) for E in Es)
    assert len(set(Es)) > 1  # Ens


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_run_train_cueq(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
    mace_params["train_file"] = (tmp_path / "fit.xyz").as_posix()
    mace_params["enable_cueq"] = True
    mace_params["default_dtype"] = "float64"

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    try:
        completed_process = subprocess.run(
            cmd.split(), capture_output=True, text=True, check=True
        )
        # Process executed successfully
        print(completed_process.stdout)
    except subprocess.CalledProcessError as e:
        # Process failed with non-zero exit code
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise e
    assert completed_process.returncode == 0

    calc = MACECalculator(model_paths=tmp_path / "MACE.model", device="cpu")
    Es = []
    for at in fitting_configs[2:]:
        at.calc = calc
        Es.append(at.get_potential_energy())

    calc = MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        enable_cueq=True,
        default_dtype=mace_params["default_dtype"],
    )
    Es_cueq = []
    for at in fitting_configs[2:]:
        at.calc = calc
        Es_cueq.append(at.get_potential_energy())

    print("Es", Es)
    print("Es_cueq", Es_cueq)

    # UPDATED alongside https://github.com/ACEsuit/mace/pull/1009
    # fmt: off
    ref_Es = [-0.05210174182665501, -0.13921024435707463, -0.43686890358685637, -0.03620905905959385, -0.09587741132383193, -0.03642914232837401, -0.09061912426962816, -0.12225001431429058, 0.11996311411730964, -0.23439757936628605, -0.0811642168331494, 0.08651948767818116, -0.16392245873271388, -0.2748895764749181, 0.04069278648779307, -0.11950527876466685, -0.04837193818213108, -0.35980614872676686, -0.2680082677516162, -0.21034634333869015]
    ref_Es_cueq = [-0.052101733271744485, -0.13921019977985483, -0.4368688963317992, -0.03620905790708441, -0.0958774132466064, -0.036429133866270016, -0.09061913318765531, -0.12225002233703638, 0.11996323954201198, -0.23439757000676983, -0.08116421200755648, 0.08651949572098083, -0.16392242752235478, -0.2748895882496253, 0.04069278143605604, -0.11950528592893625, -0.048371899343382596, -0.35980613817785434, -0.26800826497759184, -0.21034635436542232]
    # fmt: on

    npt.assert_allclose(Es, ref_Es)
    npt.assert_allclose(Es_cueq, ref_Es_cueq)
    npt.assert_allclose(Es, Es_cueq)  # FIXME: this fails due to cueq math_dtype defaulting to fp32


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
def test_run_train_foundation_multihead_json_cueq(tmp_path, fitting_configs):
    fitting_configs_dft = []
    fitting_configs_mp2 = []
    atomic_numbers = np.unique(
        np.concatenate([at.numbers for at in fitting_configs])
    ).tolist()
    for i, c in enumerate(fitting_configs):
        if i in (0, 1):
            continue  # skip isolated atoms, as energies specified by json files below
        if i % 2 == 0:
            c.info["head"] = "DFT"
            fitting_configs_dft.append(c)
        else:
            c.info["head"] = "MP2"
            fitting_configs_mp2.append(c)
    ase.io.write(tmp_path / "fit_multihead_dft.xyz", fitting_configs_dft)
    ase.io.write(tmp_path / "fit_multihead_mp2.xyz", fitting_configs_mp2)

    # write E0s to json files
    E0s = {1: 0.0, 8: 0.0}
    with open(tmp_path / "fit_multihead_dft.json", "w", encoding="utf-8") as f:
        json.dump(E0s, f)
    with open(tmp_path / "fit_multihead_mp2.json", "w", encoding="utf-8") as f:
        json.dump(E0s, f)

    heads = {
        "DFT": {
            "train_file": (tmp_path / "fit_multihead_dft.xyz").as_posix(),
            "E0s": (tmp_path / "fit_multihead_dft.json").as_posix(),
        },
        "MP2": {
            "train_file": (tmp_path / "fit_multihead_mp2.xyz").as_posix(),
            "E0s": (tmp_path / "fit_multihead_mp2.json").as_posix(),
        },
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
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
    mace_params["config"] = (tmp_path / "config.yaml").as_posix()
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
    mace_params["enable_cueq"] = True
    mace_params["atomic_numbers"] = "[" + ",".join(map(str, atomic_numbers)) + "]"
    mace_params["filter_type_pt"] = "combinations"
    mace_params["device"] = "cpu"
    mace_params["force_mh_ft_lr"] = True
    mace_params["use_reduced_cg"] = False

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    try:
        completed_process = subprocess.run(
            cmd.split(), capture_output=True, text=True, check=True
        )
        # Process executed successfully
        print(completed_process.stdout)
    except subprocess.CalledProcessError as e:
        # Process failed with non-zero exit code
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise e
    assert completed_process.returncode == 0

    calc = MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        default_dtype="float64",
        head="DFT",
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    # UPDATED alongside https://github.com/ACEsuit/mace/pull/1009
    # fmt: off
    ref_Es = [1.6126484599874644, 0.4184880784252573, 0.8881349380248947, 0.7669711420055156, 0.756405979190964, 0.9937257617991545, 0.8766625716461272, 0.7137811055985595, 0.8364737355346334, 0.8480687152643747, 0.7739037307080712, 0.7526499109184205, 0.8747432115288898, 0.6772536043733475, 0.8508245616480256, 0.6710095165298396, 1.1030868958792266, 0.8995543411287088, 0.7393368817950196, 0.7272496442406727, 0.735766226546427, 0.7821928448276221]
    # fmt: on

    npt.assert_allclose(Es, ref_Es)


def test_run_train_lbfgs(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
    mace_params["train_file"] = tmp_path / "fit.xyz"
    mace_params["lbfgs"] = None
    mace_params["max_num_epochs"] = 2

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), check=True)
    assert p.returncode == 0

    calc = MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        default_dtype=mace_params["default_dtype"],
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    # UPDATED alongside https://github.com/ACEsuit/mace/pull/1009
    # fmt: off
    ref_Es = [0.0, 0.0, -0.14064538044263225, -0.22130826908996706, 0.2204217634514321, -0.06882634515488535, -0.205467759025469, 0.08297656057994587, -0.14149773748034508, -0.2424757107898613, -0.002851084837577504, -0.18803071226693993, -0.22137481975574141, -0.08483014429170144, -0.07052495165040817, -0.21562999397031926, 0.07283523655459216, -0.052257797271031685, -0.1031833252111041, -0.3935502039828358, -0.2393776404168181, -0.1379201726932965]
    # fmt: on

    npt.assert_allclose(Es, ref_Es)


def test_run_train_foundation_elements(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    base_params = {
        "name": "MACE",
        "checkpoints_dir": (tmp_path).as_posix(),
        "model_dir": (tmp_path).as_posix(),
        "train_file": tmp_path / "fit.xyz",
        "loss": "weighted",
        "foundation_model": "small",
        "hidden_irreps": "128x0e",
        "r_max": 6.0,
        "default_dtype": "float64",
        "max_num_epochs": 5,
        "num_radial_basis": 10,
        "interaction_first": "RealAgnosticResidualInteractionBlock",
        "multiheads_finetuning": False,
    }

    # First run: without foundation_model_elements (default behavior)
    mace_params = base_params.copy()
    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )
    p = subprocess.run(cmd.split(), check=True)
    assert p.returncode == 0

    # Load model and check elements
    model_filtered = torch.load(tmp_path / "MACE.model", map_location="cpu")
    filtered_elements = set(int(z) for z in model_filtered.atomic_numbers)
    assert filtered_elements == {1, 8}  # Only H and O should be present

    # Second run: with foundation_model_elements
    mace_params = base_params.copy()
    mace_params["name"] = "MACE_all_elements"
    mace_params["foundation_model_elements"] = True  # Flag-only argument
    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )
    p = subprocess.run(cmd.split(), check=True)
    assert p.returncode == 0

    # Load model and check elements
    model_all = torch.load(tmp_path / "MACE_all_elements.model", map_location="cpu")
    all_elements = set(int(z) for z in model_all.atomic_numbers)

    # Get elements from foundation model for comparison
    calc = mace_mp(model="small", device="cpu")
    foundation_elements = set(int(z) for z in calc.models[0].atomic_numbers)

    # Check that all foundation model elements are preserved
    assert all_elements == foundation_elements
    assert len(all_elements) > len(filtered_elements)

    # Check that both models can make predictions
    at = fitting_configs[2].copy()

    # Test filtered model
    calc_filtered = MACECalculator(
        model_paths=tmp_path / "MACE.model", device="cpu", default_dtype="float64"
    )
    at.calc = calc_filtered
    e1 = at.get_potential_energy()

    # Test all-elements model
    calc_all = MACECalculator(
        model_paths=tmp_path / "MACE_all_elements.model",
        device="cpu",
        default_dtype="float64",
    )
    at.calc = calc_all
    e2 = at.get_potential_energy()

    # Energies should be different since the models are trained differently,
    # but both should give reasonable results
    assert np.isfinite(e1)
    assert np.isfinite(e2)


def test_run_train_foundation_elements_multihead(tmp_path, fitting_configs):
    fitting_configs_dft = []
    fitting_configs_mp2 = []
    atomic_numbers = np.unique(
        np.concatenate([at.numbers for at in fitting_configs])
    ).tolist()
    for i, c in enumerate(fitting_configs):
        if i in (0, 1):
            c_dft = c.copy()
            c_dft.info["head"] = "DFT"
            fitting_configs_dft.append(c_dft)
            c_mp2 = c.copy()
            c_mp2.info["head"] = "MP2"
            fitting_configs_mp2.append(c_mp2)
        if i % 2 == 0:
            c_copy = c.copy()
            c_copy.info["head"] = "DFT"
            fitting_configs_dft.append(c_copy)
        else:
            c_copy = c.copy()
            c_copy.info["head"] = "MP2"
            fitting_configs_mp2.append(c_copy)

    ase.io.write(tmp_path / "fit_dft.xyz", fitting_configs_dft)
    ase.io.write(tmp_path / "fit_mp2.xyz", fitting_configs_mp2)

    # Create multihead configuration
    heads = {
        "DFT": {"train_file": (tmp_path / "fit_dft.xyz").as_posix()},
        "MP2": {"train_file": (tmp_path / "fit_mp2.xyz").as_posix()},
    }
    yaml_str = "heads:\n"
    for key, value in heads.items():
        yaml_str += f"  {key}:\n"
        for sub_key, sub_value in value.items():
            yaml_str += f"    {sub_key}: {sub_value}\n"
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as file:
        file.write(yaml_str)

    base_params = {
        "name": "MACE",
        "checkpoints_dir": (tmp_path).as_posix(),
        "model_dir": (tmp_path).as_posix(),
        "config": (config_file).as_posix(),
        "loss": "weighted",
        "foundation_model": "small",
        "hidden_irreps": "128x0e",
        "r_max": 6.0,
        "default_dtype": "float64",
        "max_num_epochs": 5,
        "num_radial_basis": 10,
        "interaction_first": "RealAgnosticResidualInteractionBlock",
        "force_mh_ft_lr": True,
        "batch_size": 1,
        "num_samples_pt": 50,
        "subselect_pt": "random",
        "atomic_numbers": "[" + ",".join(map(str, atomic_numbers)) + "]",
        "filter_type_pt": "combinations",
        "valid_fraction": 0.1,
        "valid_batch_size": 1,
    }

    # First run: without foundation_model_elements (default behavior)
    mace_params = base_params.copy()
    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )
    try:
        completed_process = subprocess.run(
            cmd.split(), capture_output=True, text=True, check=True
        )
        # Process executed successfully
        print(completed_process.stdout)
    except subprocess.CalledProcessError as e:
        # Process failed with non-zero exit code
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise e
    assert completed_process.returncode == 0

    # Load model and check elements
    model_filtered = torch.load(tmp_path / "MACE.model", map_location="cpu")
    filtered_elements = set(int(z) for z in model_filtered.atomic_numbers)
    assert filtered_elements == {1, 8}  # Only H and O should be present
    assert len(model_filtered.heads) == 3  # pt_head + DFT + MP2

    # Second run: with foundation_model_elements
    mace_params = base_params.copy()
    mace_params["name"] = "MACE_all_elements"
    mace_params["foundation_model_elements"] = True
    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )
    p = subprocess.run(cmd.split(), check=True)
    assert p.returncode == 0

    # Load model and check elements
    model_all = torch.load(tmp_path / "MACE_all_elements.model", map_location="cpu")
    all_elements = set(int(z) for z in model_all.atomic_numbers)

    # Get elements from foundation model for comparison
    calc = mace_mp(model="small", device="cpu")
    foundation_elements = set(int(z) for z in calc.models[0].atomic_numbers)

    # Check that all foundation model elements are preserved
    assert all_elements == foundation_elements
    assert len(all_elements) > len(filtered_elements)
    assert len(model_all.heads) == 3  # pt_head + DFT + MP2

    # Check that both models can make predictions
    at = fitting_configs_dft[2].copy()

    # Test filtered model
    calc_filtered = MACECalculator(
        model_paths=tmp_path / "MACE.model",
        device="cpu",
        default_dtype="float64",
        head="DFT",
    )
    at.calc = calc_filtered
    e1 = at.get_potential_energy()

    # Test all-elements model
    calc_all = MACECalculator(
        model_paths=tmp_path / "MACE_all_elements.model",
        device="cpu",
        default_dtype="float64",
        head="DFT",
    )
    at.calc = calc_all
    e2 = at.get_potential_energy()

    assert np.isfinite(e1)
    assert np.isfinite(e2)


def test_run_train_foundation_multihead_pseudolabeling(tmp_path, fitting_configs):
    """Test multihead foundation finetuning with pseudolabeling enabled."""
    fitting_configs_dft = []
    fitting_configs_mp2 = []
    atomic_numbers = np.unique(
        np.concatenate([at.numbers for at in fitting_configs])
    ).tolist()
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
    mace_params["atomic_numbers"] = "[" + ",".join(map(str, atomic_numbers)) + "]"
    mace_params["filter_type_pt"] = "combinations"
    mace_params["force_mh_ft_lr"] = True
    mace_params["pseudolabel_replay"] = True
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

    try:
        completed_process = subprocess.run(
            cmd.split(), env=run_env, capture_output=True, text=True, check=True
        )
        # Process executed successfully
        print(completed_process.stdout)
    except subprocess.CalledProcessError as e:
        # Process failed with non-zero exit code
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise e
    assert completed_process.returncode == 0

    Es = []
    for at in fitting_configs:
        config_head = at.info.get("head", "MP2")
        calc = MACECalculator(
            model_paths=tmp_path / "MACE.model",
            device="cpu",
            default_dtype="float64",
            head=config_head,
        )
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)
    # Placeholder reference values - to be updated after first successful run
    ref_Es = [
        1.733670868926674,
        0.3503205769480672,
        1.1063898890973298,
        0.8729299295124868,
        0.8659647927914408,
        1.2055872171939568,
        1.0692213958943206,
        0.8462353744801087,
        1.032807515403141,
        1.020364286693989,
        0.7909197025613658,
        0.8318386759173623,
        1.0407572443331499,
        0.8463884897043394,
        1.0809739472058826,
        0.6669607792294759,
        1.3825620247612367,
        1.0146534339101825,
        0.9917190804182996,
        0.6952642758709486,
        0.8193046105321836,
        0.8927566093599664,
    ]
    assert np.allclose(Es, ref_Es, atol=1e-1)


@pytest.fixture(name="pretraining_configs_3_elems", scope="module")
def fixture_pretraining_configs_3_elems():
    """data for pretraining a mini foundation model with 3 elements
    returns configurations as list(Atoms)
    """
    configs = []
    for _ in range(20):
        atoms = Atoms(
            numbers=[8, 1, 3],
            positions=np.random.rand(3, 3) * 3,
            cell=[5, 5, 5],
            pbc=[True] * 3,
        )
        atoms.info["REF_energy"] = np.random.normal(0, 1)
        atoms.arrays["REF_forces"] = np.random.normal(0, 1, size=(3, 3))
        atoms.info["REF_stress"] = np.random.normal(0, 1, size=6)
        configs.append(atoms)
    for Z_i, Z in enumerate([8, 1, 3]):
        configs.append(
            Atoms(numbers=[Z], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3),
        )
        configs[-1].info["REF_energy"] = -4.0 + Z_i
        configs[-1].info["config_type"] = "IsolatedAtom"
    return configs


@pytest.fixture(name="mini_foundation_model", scope="module")
def fixture_mini_foundation_model(tmp_path_factory, pretraining_configs_3_elems):
    """fits tiny model that can be used as foundation for multihead replay finetuning
    returns path of model file and value of XDG_CACHE_HOME where pretraining data is cached
    """
    # generate pretraining data
    tmp_path = tmp_path_factory.mktemp("mini_foundation_model")

    ase.io.write(tmp_path / "pretrain.xyz", pretraining_configs_3_elems)

    foundation_params = {
        "name": "foundation",
        "train_file": os.path.join(tmp_path, "pretrain.xyz"),
        "valid_fraction": 0.2,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "MACE",
        "hidden_irreps": "32x0e",
        "r_max": 5.0,
        "batch_size": 2,
        "max_num_epochs": 5,
        "swa": None,
        "start_swa": 3,
        "device": "cpu",
        "seed": 42,
        "loss": "weighted",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
        "default_dtype": "float64",
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
    }

    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)

    # create fake cached MP data
    run_env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    (tmp_path / "cache" / "mace").mkdir(parents=True)
    with open(
        tmp_path / "cache" / "mace" / "mp_traj_combinedxyz", "w", encoding="utf-8"
    ) as fout:
        for atoms in pretraining_configs_3_elems:
            if atoms.info.get("config_type") == "IsolatedAtom":
                continue
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=atoms.info["REF_energy"],
                forces=atoms.arrays.get("REF_forces"),
            )
            ase.io.write(fout, atoms, format="extxyz")
            atoms.calc = None

    cmd = [sys.executable, str(run_train)]
    for k, v in foundation_params.items():
        if v is None:
            cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}={v}")

    print("pretraining fit cmd", cmd)
    try:
        orig_dir = Path.cwd()
        os.chdir(tmp_path)
        p = subprocess.run(cmd, env=run_env, check=True)
        os.chdir(orig_dir)
    except Exception:
        os.chdir(orig_dir)
        raise

    assert p.returncode == 0

    return tmp_path / "foundation.model", run_env["XDG_CACHE_HOME"]


@pytest.fixture(name="multihead_finetuning_config_20")
def fixture_multihead_finetuning_config_20(tmp_path, fitting_configs):
    return do_fixture_multihead_finetuning_config(tmp_path, fitting_configs, 20)


@pytest.fixture(name="multihead_finetuning_config_5")
def fixture_multihead_finetuning_config_5(tmp_path, fitting_configs):
    return do_fixture_multihead_finetuning_config(tmp_path, fitting_configs, 5)


def do_fixture_multihead_finetuning_config(tmp_path, fitting_configs, n_fit):
    # Step 3: Create finetuning set
    fitting_configs_dft = []
    fitting_configs_mp2 = []
    atomic_numbers = set()

    rng = np.random.default_rng(10)
    n_isolated = sum(
        config.info.get("config_type") == "IsolatedAtom" for config in fitting_configs
    )
    # assume that _first_ n_isolated are IsolatedAtom
    n_dups = int(np.ceil(n_fit / (len(fitting_configs) - n_isolated)))
    avail_configs = fitting_configs[n_isolated:] * n_dups
    rng.shuffle(avail_configs)
    avail_configs = fitting_configs[0:n_isolated] + avail_configs[0:n_fit]

    for i, c in enumerate(avail_configs):
        atomic_numbers |= set(c.numbers)
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
    ase.io.write(tmp_path / f"fit_multihead_dft_{n_fit}.xyz", fitting_configs_dft)
    ase.io.write(tmp_path / f"fit_multihead_mp2_{n_fit}.xyz", fitting_configs_mp2)

    # Step 4: Finetune the pretrained model with multihead replay
    heads = {
        "DFT": {"train_file": str(tmp_path / f"fit_multihead_dft_{n_fit}.xyz")},
        "MP2": {"train_file": str(tmp_path / f"fit_multihead_mp2_{n_fit}.xyz")},
    }
    yaml_str = "heads:\n"
    for key, value in heads.items():
        yaml_str += f"  {key}:\n"
        for sub_key, sub_value in value.items():
            yaml_str += f"    {sub_key}: {sub_value}\n"
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w", encoding="utf-8") as file:
        file.write(yaml_str)

    return config_filename, atomic_numbers


# test multihead replay fine-tuning when fine-tuning data
# has only a subset of the species present in the filtered replay data
def test_run_train_multihead_replay_filtered_pt_data(
    tmp_path,
    monkeypatch,
    mini_foundation_model,
    multihead_finetuning_config_20,
):
    finetuning_params = {
        "name": "finetuned",
        "valid_fraction": 0.1,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "MACE",
        "hidden_irreps": "32x0e",
        "r_max": 5.0,
        "batch_size": 2,
        "max_num_epochs": 5,
        "device": "cpu",
        "seed": 42,
        "loss": "weighted",
        "default_dtype": "float64",
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "foundation_model": str(mini_foundation_model[0]),
        "config": str(multihead_finetuning_config_20[0]),
        "pt_train_file": "mp",
        "num_samples_pt": 10,
        "subselect_pt": "random",
        "filter_type_pt": "exclusive",
        "force_mh_ft_lr": True,
        "atomic_numbers": str(sorted(multihead_finetuning_config_20[1])),
        "dry_run": None,
    }

    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)

    # create fake cached MP data
    run_env["XDG_CACHE_HOME"] = mini_foundation_model[1]

    cmd = [sys.executable, str(run_train)]
    for k, v in finetuning_params.items():
        if v is None:
            cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}={v}")

    print("fine-tuning fit cmd", cmd)
    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        p = subprocess.run(cmd, env=run_env, check=True)
    assert p.returncode == 0


# test multihead replay fine-tuning ratio of real to ft data
# try to reduce time by using refactored eodule-scope mini_foundation_model
# fixture and --dry_run
def test_run_train_real_pt_data_ratio(
    tmp_path,
    monkeypatch,
    mini_foundation_model,
    multihead_finetuning_config_5,
):
    finetuning_params = {
        "name": "finetuned",
        "valid_fraction": 0.1,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "MACE",
        "hidden_irreps": "32x0e",
        "r_max": 5.0,
        "batch_size": 2,
        "max_num_epochs": 5,
        "device": "cpu",
        "seed": 42,
        "loss": "weighted",
        "default_dtype": "float64",
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "foundation_model": str(mini_foundation_model[0]),
        "config": str(multihead_finetuning_config_5[0]),
        "pt_train_file": "mp",
        "num_samples_pt": 10,
        "subselect_pt": "random",
        "filter_type_pt": "exclusive",
        "force_mh_ft_lr": True,
        "atomic_numbers": str(sorted(multihead_finetuning_config_5[1])),
        "dry_run": None,
    }

    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)

    # create fake cached MP data
    run_env["XDG_CACHE_HOME"] = mini_foundation_model[1]

    def _create_cmd(finetuning_params):
        cmd = [sys.executable, str(run_train)]
        for k, v in finetuning_params.items():
            if v is None:
                cmd.append(f"--{k}")
            else:
                cmd.append(f"--{k}={v}")

        return cmd

    cmd = _create_cmd(finetuning_params)
    print("fine-tuning fit cmd", cmd)
    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        p = subprocess.run(cmd, env=run_env, check=True, capture_output=True)
    print(p.stdout.decode("utf-8"))
    assert p.returncode == 0

    # real to pt data ratio should not be triggered by 5 / 20 > default of 0.1
    assert (
        len(
            [
                l
                for l in p.stdout.decode("utf-8").splitlines()
                if "Ratio of the number of configurations in the "
                "training set and the in the pt_train_file" in l
            ]
        )
        == 0
    )

    finetuning_params["name"] = "finetuned_repeated_data"
    finetuning_params["real_pt_data_ratio_threshold"] = 0.5
    cmd = _create_cmd(finetuning_params)
    print("fine-tuning fit cmd", cmd)
    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        p = subprocess.run(cmd, env=run_env, check=True, capture_output=True)
    assert p.returncode == 0
    print(p.stdout.decode("utf-8"))

    # real to pt data ratio should not be triggered by 5 / 20 > default of 0.1
    l_ratio = [
        l
        for l in p.stdout.decode("utf-8").splitlines()
        if "Ratio of the number of configurations in the "
        "training set and the in the pt_train_file" in l
    ]
    assert len(l_ratio) == 1
    assert l_ratio[0].strip().endswith(" 1")
