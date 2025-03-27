import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
import yaml
from ase.atoms import Atoms

pytest_mace_dir = Path(__file__).parent.parent
preprocess_data = Path(__file__).parent.parent / "mace" / "cli" / "preprocess_data.py"


@pytest.fixture(name="sample_configs")
def fixture_sample_configs():
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    configs = [
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3),
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3),
    ]
    configs[0].info["REF_energy"] = 0.0
    configs[0].info["config_type"] = "IsolatedAtom"
    configs[1].info["REF_energy"] = 0.0
    configs[1].info["config_type"] = "IsolatedAtom"

    np.random.seed(5)
    for _ in range(10):
        c = water.copy()
        c.positions += np.random.normal(0.1, size=c.positions.shape)
        c.info["REF_energy"] = np.random.normal(0.1)
        c.new_array("REF_forces", np.random.normal(0.1, size=c.positions.shape))
        c.info["REF_stress"] = np.random.normal(0.1, size=6)
        configs.append(c)

    return configs


def test_preprocess_data(tmp_path, sample_configs):
    ase.io.write(tmp_path / "sample.xyz", sample_configs)

    preprocess_params = {
        "train_file": tmp_path / "sample.xyz",
        "r_max": 5.0,
        "config_type_weights": "{'Default':1.0}",
        "num_process": 2,
        "valid_fraction": 0.1,
        "h5_prefix": tmp_path / "preprocessed_",
        "compute_statistics": None,
        "seed": 42,
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
    }

    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

    cmd = (
        sys.executable
        + " "
        + str(preprocess_data)
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in preprocess_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), env=run_env, check=True)
    assert p.returncode == 0

    # Check if the output files are created
    assert (tmp_path / "preprocessed_train").is_dir()
    assert (tmp_path / "preprocessed_val").is_dir()
    assert (tmp_path / "preprocessed_statistics.json").is_file()

    # Check if the correct number of files are created
    train_files = list((tmp_path / "preprocessed_train").glob("*.h5"))
    val_files = list((tmp_path / "preprocessed_val").glob("*.h5"))
    assert len(train_files) == preprocess_params["num_process"]
    assert len(val_files) == preprocess_params["num_process"]

    # Example of checking statistics file content:
    import json

    with open(tmp_path / "preprocessed_statistics.json", "r", encoding="utf-8") as f:
        statistics = json.load(f)
    assert "atomic_energies" in statistics
    assert "avg_num_neighbors" in statistics
    assert "mean" in statistics
    assert "std" in statistics
    assert "atomic_numbers" in statistics
    assert "r_max" in statistics

    # Example of checking H5 file content:
    import h5py

    with h5py.File(train_files[0], "r") as f:
        assert "config_batch_0" in f
        config = f["config_batch_0"]["config_0"]
        assert "atomic_numbers" in config
        assert "positions" in config
        assert "energy" in config["properties"]
        assert "forces" in config["properties"]

    original_energies = [
        config.info["REF_energy"]
        for config in sample_configs[2:]
        if "REF_energy" in config.info
    ]
    original_forces = [
        config.arrays["REF_forces"]
        for config in sample_configs[2:]
        if "REF_forces" in config.arrays
    ]

    h5_energies = []
    h5_forces = []

    for train_file in train_files:
        with h5py.File(train_file, "r") as f:
            for _, batch in f.items():
                for config_key in batch.keys():
                    config = batch[config_key]
                    assert "atomic_numbers" in config
                    assert "positions" in config
                    assert "energy" in config["properties"]
                    assert "forces" in config["properties"]

                    h5_energies.append(config["properties"]["energy"][()])
                    h5_forces.append(config["properties"]["forces"][()])

    for val_file in val_files:
        with h5py.File(val_file, "r") as f:
            for _, batch in f.items():
                for config_key in batch.keys():
                    config = batch[config_key]
                    h5_energies.append(config["properties"]["energy"][()])
                    h5_forces.append(config["properties"]["forces"][()])

    print("Original energies", original_energies)
    print("H5 energies", h5_energies)
    print("Original forces", original_forces)
    print("H5 forces", h5_forces)
    original_energies.sort()
    h5_energies.sort()
    original_forces = np.concatenate(original_forces).flatten()
    h5_forces = np.concatenate(h5_forces).flatten()
    original_forces.sort()
    h5_forces.sort()

    # Compare energies and forces
    np.testing.assert_allclose(original_energies, h5_energies, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(original_forces, h5_forces, rtol=1e-5, atol=1e-8)

    print("All checks passed successfully!")


def test_preprocess_config(tmp_path, sample_configs):
    ase.io.write(tmp_path / "sample.xyz", sample_configs)

    preprocess_params = {
        "train_file": str(tmp_path / "sample.xyz"),
        "r_max": 5.0,
        "config_type_weights": "{'Default':1.0}",
        "num_process": 2,
        "valid_fraction": 0.1,
        "h5_prefix": str(tmp_path / "preprocessed_"),
        "compute_statistics": None,
        "seed": 42,
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
    }
    filename = tmp_path / "config.yaml"
    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(preprocess_params, file)

    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

    cmd = (
        sys.executable
        + " "
        + str(preprocess_data)
        + " "
        + "--config"
        + " "
        + str(filename)
    )

    p = subprocess.run(cmd.split(), env=run_env, check=True)
    assert p.returncode == 0
