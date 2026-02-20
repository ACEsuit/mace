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

RUN_TRAIN = Path(__file__).resolve().parents[1] / "mace" / "cli" / "run_train.py"


def _write_polar_data(tmp_path):
    configs = []
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[6.0, 6.0, 6.0],
        pbc=[True, True, True],
    )
    rng = np.random.default_rng(123)
    for i in range(4):
        sample = atoms.copy()
        sample.positions += rng.normal(0, 0.1, size=sample.positions.shape)
        sample.info["REF_energy"] = rng.normal(0, 1e-2)
        sample.new_array(
            "REF_forces", rng.normal(0, 1e-2, size=sample.positions.shape)
        )
        sample.info["REF_stress"] = rng.normal(0, 1e-2, size=6)
        configs.append(sample)
    path = tmp_path / "polar_smoke.xyz"
    ase.io.write(path, configs)
    return path, configs


def _write_polar_multihead_data(tmp_path, configs):
    dft_configs = []
    mp2_configs = []
    for i, atoms in enumerate(configs):
        sample = atoms.copy()
        if i % 2 == 0:
            sample.info["head"] = "DFT"
            dft_configs.append(sample)
        else:
            sample.info["head"] = "MP2"
            mp2_configs.append(sample)

    dft_path = tmp_path / "polar_dft.xyz"
    mp2_path = tmp_path / "polar_mp2.xyz"
    ase.io.write(dft_path, dft_configs)
    ase.io.write(mp2_path, mp2_configs)
    return dft_path, mp2_path


def _write_heads_config(path, dft_path, mp2_path):
    yaml_str = "\n".join(
        [
            "heads:",
            "  DFT:",
            f"    train_file: {dft_path}",
            "    E0s: foundation",
            "  MP2:",
            f"    train_file: {mp2_path}",
            "    E0s: foundation",
        ]
    )
    path.write_text(yaml_str, encoding="utf-8")


def _run_train(params, extra_env=None):
    run_env = os.environ.copy()
    repo = Path(__file__).resolve().parents[1]
    pythonpath = run_env.get("PYTHONPATH")
    run_env["PYTHONPATH"] = ":".join(filter(None, [str(repo), pythonpath]))
    if extra_env:
        run_env.update(extra_env)
    cmd = [sys.executable, str(RUN_TRAIN)]
    for key, value in params.items():
        if value is None:
            cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}={value}")
    try:
        subprocess.run(cmd, env=run_env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout)
        print(exc.stderr)
        raise


def _assert_model_predicts(model_path, configs, heads=("Default",)):
    for head in heads:
        calc = MACECalculator(
            model_paths=model_path,
            device="cpu",
            default_dtype="float64",
            head=head,
        )
        for atoms in configs:
            test_atoms = atoms.copy()
            test_atoms.calc = calc
            assert np.isfinite(test_atoms.get_potential_energy())


def _base_train_params(tmp_path, train_file, name):
    return {
        "name": name,
        "train_file": train_file,
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "valid_fraction": 0.2,
        "model": "PolarMACE",
        "hidden_irreps": "16x0e",
        "r_max": 3.5,
        "batch_size": 2,
        "max_num_epochs": 1,
        "lr": 1e-4,
        "loss": "stress",
        "default_dtype": "float64",
        "device": "cpu",
        "E0s": "average",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
    }


def test_run_train_polar_finetuning_from_checkpoint(tmp_path):
    train_file, configs = _write_polar_data(tmp_path)
    base_params = _base_train_params(tmp_path, train_file, "polar_base")
    base_params["valid_fraction"] = 0.3
    _run_train(base_params)

    ft_params = {
        **base_params,
        "name": "polar_ft",
        "foundation_model": str(tmp_path / "polar_base.model"),
        "force_mh_ft_lr": True,
    }
    _run_train(ft_params)

    model_path = tmp_path / "polar_ft.model"
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    assert model.__class__.__name__ == "PolarMACE"
    assert model.heads == ["Default"]
    _assert_model_predicts(model_path, configs, heads=("Default",))


@pytest.mark.parametrize("foundation_model", ["mace-polar-2L", "mace-polar-3L"])
def test_run_train_polar_finetuning_foundation_model(tmp_path, foundation_model):
    train_file, configs = _write_polar_data(tmp_path)
    params = _base_train_params(tmp_path, train_file, f"polar_{foundation_model}_ft")
    params["foundation_model"] = foundation_model
    params["force_mh_ft_lr"] = True
    params["loss"] = "weighted"
    params["stress_weight"] = 0.0
    _run_train(params)

    model_path = tmp_path / f"polar_{foundation_model}_ft.model"
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    assert model.__class__.__name__ == "PolarMACE"
    assert model.heads == ["Default"]
    if foundation_model == "mace-polar-2L":
        _assert_model_predicts(model_path, configs, heads=("Default",))


@pytest.mark.parametrize("foundation_model", ["mace-polar-2L", "mace-polar-3L"])
def test_run_train_polar_multihead_finetuning_foundation_model(
    tmp_path, foundation_model
):
    pt_train_file, configs = _write_polar_data(tmp_path)
    dft_path, mp2_path = _write_polar_multihead_data(tmp_path, configs)
    config_path = tmp_path / "polar_heads.yaml"
    _write_heads_config(config_path, dft_path, mp2_path)

    params = {
        "name": "polar_2l_mh",
        "config": config_path,
        "pt_train_file": pt_train_file,
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "valid_fraction": 0.2,
        "model": "PolarMACE",
        "hidden_irreps": "16x0e",
        "r_max": 3.5,
        "batch_size": 2,
        "max_num_epochs": 1,
        "lr": 1e-4,
        "loss": "weighted",
        "force_mh_ft_lr": True,
        "num_samples_pt": 4,
        "default_dtype": "float64",
        "foundation_model": foundation_model,
        "device": "cpu",
    }
    _run_train(params)

    model_path = tmp_path / "polar_2l_mh.model"
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    assert model.__class__.__name__ == "PolarMACE"
    assert set(model.heads) == {"DFT", "MP2", "pt_head"}
    _assert_model_predicts(model_path, configs, heads=("DFT", "MP2", "pt_head"))
