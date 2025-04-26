import json
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import lmdb
import numpy as np
import orjson
import pytest
import torch
import yaml
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mace.calculators import MACECalculator


def create_test_atoms(num_atoms=5, seed=42):
    """Create random atoms for testing purposes with energy, forces, and stress."""
    # Set random seed for reproducibility
    rng = np.random.RandomState(seed)

    # Create random positions
    positions = rng.rand(num_atoms, 3) * 5.0

    # Create random atomic numbers (H, C, N, O)
    atomic_numbers = rng.choice([1, 6, 7, 8], size=num_atoms)

    # Create atoms object
    atoms = Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=np.eye(3) * 10.0,  # 10 Å periodic box
        pbc=True,
    )

    # Add random energy, forces and stress
    energy = float(rng.uniform(-15.0, -5.0))
    forces = rng.rand(num_atoms, 3) * 0.5 - 0.25  # Forces between -0.25 and 0.25 eV/Å
    stress = rng.rand(6) * 0.2 - 0.1  # Stress tensor in Voigt notation

    # Add calculator to atoms with results
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
    atoms.calc = calc

    # Mark isolated atoms with config_type
    if num_atoms == 1:
        atoms.info["config_type"] = "IsolatedAtom"

    return atoms


def create_xyz_file(atoms_list, filename):
    """Write a list of atoms to an xyz file."""
    from ase.io import write

    write(filename, atoms_list, format="extxyz")
    return filename


def create_e0s_file(e0s_dict, filename):
    """Create an E0s JSON file with isolated atom energies."""
    # Convert keys to integers since MACE expects atomic numbers as integers
    e0s_dict_int_keys = {int(k): v for k, v in e0s_dict.items()}

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(e0s_dict_int_keys, f)
    return filename


def create_h5_dataset(xyz_file, output_dir, e0s_file=None, r_max=5.0, seed=42):
    """
    Run MACE's preprocess_data.py script to convert an xyz file to h5 format.

    Args:
        xyz_file: Path to the input xyz file
        output_dir: Directory to store the preprocessed h5 files
        e0s_file: Path to the E0s file with isolated atom energies
        r_max: Cutoff radius
        seed: Random seed

    Returns:
        The output directory containing the h5 files
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find the path to the preprocess_data.py script
    preprocess_script = (
        Path(__file__).parent.parent / "mace" / "cli" / "preprocess_data.py"
    )

    # Set up command to run preprocess_data.py
    cmd = [
        sys.executable,
        str(preprocess_script),
        f"--train_file={xyz_file}",
        f"--r_max={r_max}",
        f"--h5_prefix={output_dir}/",
        f"--seed={seed}",
        "--compute_statistics",  # Generate statistics file
        "--num_process=2",  # Create 2 files for testing sharded loading
    ]

    # Add E0s file if provided
    if e0s_file:
        cmd.append(f"--E0s={e0s_file}")

    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(Path(__file__).parent.parent) + ":" + env.get("PYTHONPATH", "")
    )

    # Run the script
    print(f"Running preprocess command: {' '.join(cmd)}")
    try:
        process = subprocess.run(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        # Print output for debugging
        print("Preprocess stdout:", process.stdout.decode())
        print("Preprocess stderr:", process.stderr.decode())
    except subprocess.CalledProcessError as e:
        print("Preprocess failed with error:", e)
        print("Stdout:", e.stdout.decode() if e.stdout else "")
        print("Stderr:", e.stderr.decode() if e.stderr else "")
        raise

    return output_dir


def create_lmdb_dataset(atoms_list, folder_path, head_name="Default"):
    """Create an LMDB dataset from a list of atoms objects that MACE can read."""
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Create the LMDB database file
    db_path = os.path.join(folder_path, "data.aselmdb")

    # Initialize LMDB environment
    env = lmdb.open(
        db_path,
        map_size=1099511627776,  # 1TB
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # Open a transaction
    with env.begin(write=True) as txn:
        # Store metadata
        metadata = {"format_version": 1}
        txn.put(
            "metadata".encode("ascii"),
            zlib.compress(orjson.dumps(metadata, option=orjson.OPT_SERIALIZE_NUMPY)),
        )

        # Store nextid
        nextid = len(atoms_list) + 1
        txn.put(
            "nextid".encode("ascii"),
            zlib.compress(orjson.dumps(nextid, option=orjson.OPT_SERIALIZE_NUMPY)),
        )

        # Store deleted_ids (empty)
        txn.put(
            "deleted_ids".encode("ascii"),
            zlib.compress(orjson.dumps([], option=orjson.OPT_SERIALIZE_NUMPY)),
        )

        # Store each atom
        for i, atoms in enumerate(atoms_list):
            id_num = i + 1  # Start from 1

            # Convert atoms to dictionary
            positions = atoms.get_positions()
            cell = atoms.get_cell()

            # Create a dictionary with all necessary fields
            dct = {
                "numbers": atoms.get_atomic_numbers().tolist(),
                "positions": positions.tolist(),
                "cell": cell.tolist(),
                "pbc": atoms.get_pbc().tolist(),
                "ctime": 0.0,  # Creation time
                "mtime": 0.0,  # Modification time
                "user": "test",
                "energy": atoms.calc.results["energy"],
                "forces": atoms.calc.results["forces"].tolist(),
                "stress": atoms.calc.results["stress"].tolist(),
                "key_value_pairs": {
                    "config_type": atoms.info.get("config_type", "Default"),
                    "head": head_name,
                },
            }

            # Store the atom in LMDB
            txn.put(
                f"{id_num}".encode("ascii"),
                zlib.compress(orjson.dumps(dct, option=orjson.OPT_SERIALIZE_NUMPY)),
            )

    # Close the environment
    env.close()

    return folder_path


@pytest.mark.slow
def test_multifile_training():
    """Test training with multiple file formats per head"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Set up file paths
        xyz_file1 = os.path.join(temp_dir, "data1.xyz")
        xyz_file2 = os.path.join(temp_dir, "data2.xyz")
        iso_atoms_file = os.path.join(temp_dir, "isolated_atoms.xyz")
        h5_folder = os.path.join(temp_dir, "h5_data")
        lmdb_folder1 = os.path.join(
            temp_dir, "lmdb_data1_lmdb"
        )  # Add _lmdb suffix for LMDB recognition
        lmdb_folder2 = os.path.join(
            temp_dir, "lmdb_data2_lmdb"
        )  # Add _lmdb suffix for LMDB recognition

        config_path = os.path.join(temp_dir, "config.yaml")
        results_dir = os.path.join(temp_dir, "results")
        checkpoints_dir = os.path.join(temp_dir, "checkpoints")
        model_dir = os.path.join(temp_dir, "models")
        e0s_file = os.path.join(temp_dir, "e0s.json")

        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Set atomic numbers for z_table
        z_table_elements = [1, 6, 7, 8]  # H, C, N, O

        # Create test data for each format
        rng = np.random.RandomState(42)
        seeds = rng.randint(0, 10000, size=5)

        # Create isolated atoms for E0s (one of each element)
        isolated_atoms = []
        e0s_dict = {}
        for z in z_table_elements:
            # Create isolated atom
            atom = Atoms(
                numbers=[z], positions=[[0, 0, 0]], cell=np.eye(3) * 10.0, pbc=True
            )
            energy = float(rng.uniform(-5.0, -1.0))  # Random reference energy
            forces = np.zeros((1, 3))
            stress = np.zeros(6)
            calc = SinglePointCalculator(
                atom, energy=energy, forces=forces, stress=stress
            )
            atom.calc = calc
            atom.info["config_type"] = "IsolatedAtom"
            atom.info["REF_energy"] = energy  # Make sure energy is in the right place
            isolated_atoms.append(atom)
            e0s_dict[str(z)] = energy  # Store energy for E0s file

        # Create E0s file
        create_e0s_file(e0s_dict, e0s_file)

        # Create isolated atoms xyz file
        create_xyz_file(isolated_atoms, iso_atoms_file)

        # Create 10 atoms for each dataset
        xyz_atoms1 = [
            create_test_atoms(num_atoms=5, seed=seeds[0] + i) for i in range(10)
        ]
        xyz_atoms2 = [
            create_test_atoms(num_atoms=5, seed=seeds[1] + i) for i in range(10)
        ]

        # Create h5 data directly - first convert the xyz file to a format with REF_ keys
        for atom in xyz_atoms1:
            atom.info["REF_energy"] = atom.calc.results["energy"]
            atom.arrays["REF_forces"] = atom.calc.results["forces"]
            atom.info["REF_stress"] = atom.calc.results["stress"]

        for atom in xyz_atoms2:
            atom.info["REF_energy"] = atom.calc.results["energy"]
            atom.arrays["REF_forces"] = atom.calc.results["forces"]
            atom.info["REF_stress"] = atom.calc.results["stress"]

        # Save isolated atoms to xyz files first, then create the h5 datasets
        create_xyz_file(xyz_atoms1, xyz_file1)
        create_xyz_file(xyz_atoms2, xyz_file2)

        # Create h5 data from xyz file, using both isolated atoms and real data
        all_atoms_for_h5 = isolated_atoms + xyz_atoms2
        all_atoms_xyz = os.path.join(temp_dir, "all_atoms_for_h5.xyz")
        create_xyz_file(all_atoms_for_h5, all_atoms_xyz)
        create_h5_dataset(all_atoms_xyz, h5_folder)

        # Create LMDB datasets
        lmdb_atoms1 = [
            create_test_atoms(num_atoms=5, seed=seeds[3] + i) for i in range(10)
        ]
        lmdb_atoms2 = [
            create_test_atoms(num_atoms=5, seed=seeds[4] + i) for i in range(10)
        ]
        create_lmdb_dataset(lmdb_atoms1, lmdb_folder1, head_name="head1")
        create_lmdb_dataset(lmdb_atoms2, lmdb_folder2, head_name="head2")

        # Create config.yaml for training with proper format specification
        config = {
            "name": "multifile_test",
            "seed": 42,
            "model": "MACE",
            "hidden_irreps": "32x0e",
            "r_max": 5.0,
            "batch_size": 5,
            "max_num_epochs": 2,
            "patience": 5,
            "device": "cpu",
            "energy_weight": 1.0,
            "forces_weight": 10.0,
            "loss": "weighted",
            "optimizer": "adam",
            "default_dtype": "float64",
            "lr": 0.01,
            "swa": False,
            "work_dir": temp_dir,
            "results_dir": results_dir,
            "checkpoints_dir": checkpoints_dir,
            "model_dir": model_dir,
            "E0s": e0s_file,
            "atomic_numbers": str(z_table_elements),
            "heads": {
                "head1": {
                    "train_file": [lmdb_folder1, xyz_file1],
                    "valid_file": xyz_file1,
                    "energy_key": "REF_energy",
                    "forces_key": "REF_forces",
                    "stress_key": "REF_stress",
                },
                "head2": {
                    "train_file": [h5_folder + "/train", xyz_file2],
                    "valid_file": xyz_file2,
                    "energy_key": "REF_energy",
                    "forces_key": "REF_forces",
                    "stress_key": "REF_stress",
                },
            },
        }

        # Write config file
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        # Import the modified run_train from our local module
        run_train_script = (
            Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"
        )

        # Run training with subprocess
        cmd = [sys.executable, str(run_train_script), f"--config={config_path}"]

        # Set environment to add the current path to PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(Path(__file__).parent.parent) + ":" + env.get("PYTHONPATH", "")
        )

        # Run the process
        process = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,  # Don't raise exception on non-zero exit, we'll check manually
        )

        # Print output for debugging
        print("\n" + "=" * 40 + " STDOUT " + "=" * 40)
        print(process.stdout.decode())
        print("\n" + "=" * 40 + " STDERR " + "=" * 40)
        print(process.stderr.decode())

        # Check that process completed successfully
        assert (
            process.returncode == 0
        ), f"Training failed with error: {process.stderr.decode()}"

        # Check that model was created
        model_path = os.path.join(model_dir, "multifile_test.model")
        assert os.path.exists(model_path), f"Model was not created at {model_path}"

        # Try to load and run the model
        model = torch.load(model_path, map_location="cpu")
        assert model is not None, "Failed to load model"

        # Create a calculator
        calc = MACECalculator(model_paths=model_path, device="cpu", head="head1")

        # Run prediction on a test atom
        test_atom = create_test_atoms(num_atoms=5, seed=99999)
        test_atom.calc = calc
        energy = test_atom.get_potential_energy()
        forces = test_atom.get_forces()

        # Assert we got sensible outputs
        assert np.isfinite(energy), "Model produced non-finite energy"
        assert np.all(np.isfinite(forces)), "Model produced non-finite forces"

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


@pytest.mark.slow
def test_multiple_xyz_per_head():
    """Test training with multiple XYZ files per head for train, valid and test sets"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Set up file paths - create multiple xyz files for each dataset
        train_xyz_files = [
            os.path.join(temp_dir, f"train_data{i}.xyz") for i in range(1, 4)
        ]  # 3 train files
        valid_xyz_files = [
            os.path.join(temp_dir, f"valid_data{i}.xyz") for i in range(1, 3)
        ]  # 2 valid files
        test_xyz_files = [
            os.path.join(temp_dir, f"test_data{i}.xyz") for i in range(1, 3)
        ]  # 2 test files

        iso_atoms_file = os.path.join(temp_dir, "isolated_atoms.xyz")

        config_path = os.path.join(temp_dir, "config.yaml")
        results_dir = os.path.join(temp_dir, "results")
        checkpoints_dir = os.path.join(temp_dir, "checkpoints")
        model_dir = os.path.join(temp_dir, "models")
        e0s_file = os.path.join(temp_dir, "e0s.json")

        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Set atomic numbers for z_table
        z_table_elements = [1, 6, 7, 8]  # H, C, N, O

        # Create test data for each format
        rng = np.random.RandomState(42)
        seeds = rng.randint(0, 10000, size=10)  # More seeds for multiple files

        # Create isolated atoms for E0s (one of each element)
        isolated_atoms = []
        e0s_dict = {}
        for z in z_table_elements:
            # Create isolated atom
            atom = Atoms(
                numbers=[z], positions=[[0, 0, 0]], cell=np.eye(3) * 10.0, pbc=True
            )
            energy = float(rng.uniform(-5.0, -1.0))  # Random reference energy
            forces = np.zeros((1, 3))
            stress = np.zeros(6)
            calc = SinglePointCalculator(
                atom, energy=energy, forces=forces, stress=stress
            )
            atom.calc = calc
            atom.info["config_type"] = "IsolatedAtom"
            isolated_atoms.append(atom)
            e0s_dict[str(z)] = energy  # Store energy for E0s file

        # Create E0s file
        create_e0s_file(e0s_dict, e0s_file)

        # Create isolated atoms xyz file
        create_xyz_file(isolated_atoms, iso_atoms_file)

        # Create atoms for each train dataset - use different seeds for variety
        train_datasets = []
        for i, file in enumerate(train_xyz_files):
            # Create atoms with different seeds
            atoms = [
                create_test_atoms(num_atoms=5, seed=seeds[i] + j) for j in range(5)
            ]
            create_xyz_file(atoms, file)
            train_datasets.append(atoms)

        # Create atoms for validation datasets
        valid_datasets = []
        for i, file in enumerate(valid_xyz_files):
            atoms = [
                create_test_atoms(num_atoms=5, seed=seeds[i + 3] + j) for j in range(3)
            ]
            create_xyz_file(atoms, file)
            valid_datasets.append(atoms)

        # Create atoms for test datasets
        test_datasets = []
        for i, file in enumerate(test_xyz_files):
            atoms = [
                create_test_atoms(num_atoms=5, seed=seeds[i + 5] + j) for j in range(3)
            ]
            create_xyz_file(atoms, file)
            test_datasets.append(atoms)

        # Create config.yaml for training with multiple xyz files per dataset
        config = {
            "name": "multi_xyz_test",
            "seed": 42,
            "model": "MACE",
            "hidden_irreps": "32x0e",
            "r_max": 5.0,
            "batch_size": 5,
            "max_num_epochs": 2,
            "patience": 5,
            "device": "cpu",
            "energy_weight": 1.0,
            "forces_weight": 10.0,
            "loss": "weighted",
            "optimizer": "adam",
            "default_dtype": "float64",
            "lr": 0.01,
            "swa": False,
            "work_dir": temp_dir,
            "results_dir": results_dir,
            "checkpoints_dir": checkpoints_dir,
            "model_dir": model_dir,
            "E0s": e0s_file,
            "atomic_numbers": str(z_table_elements),
            "heads": {
                "multi_xyz_head": {
                    # Using lists of multiple xyz files for each dataset
                    "train_file": train_xyz_files,
                    "valid_file": valid_xyz_files,
                    "test_file": test_xyz_files,
                    "energy_key": "energy",
                    "forces_key": "forces",
                    "stress_key": "stress",
                },
            },
        }

        # Write config file
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        # Import the modified run_train from our local module
        run_train_script = (
            Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"
        )

        # Run training with subprocess
        cmd = [sys.executable, str(run_train_script), f"--config={config_path}"]

        # Set environment to add the current path to PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(Path(__file__).parent.parent) + ":" + env.get("PYTHONPATH", "")
        )

        # Run the process
        process = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        # Print output for debugging
        print("\n" + "=" * 40 + " STDOUT " + "=" * 40)
        print(process.stdout.decode())
        print("\n" + "=" * 40 + " STDERR " + "=" * 40)
        print(process.stderr.decode())

        # Check that process completed successfully
        assert (
            process.returncode == 0
        ), f"Training failed with error: {process.stderr.decode()}"

        # Check that model was created
        model_path = os.path.join(model_dir, "multi_xyz_test.model")
        assert os.path.exists(model_path), f"Model was not created at {model_path}"

        # Try to load and run the model
        model = torch.load(model_path, map_location="cpu")
        assert model is not None, "Failed to load model"

        # Create a calculator
        calc = MACECalculator(
            model_paths=model_path, device="cpu", head="multi_xyz_head"
        )

        # Run prediction on a test atom
        test_atom = create_test_atoms(num_atoms=5, seed=99999)
        test_atom.calc = calc
        energy = test_atom.get_potential_energy()
        forces = test_atom.get_forces()

        # Assert we got sensible outputs
        assert np.isfinite(energy), "Model produced non-finite energy"
        assert np.all(np.isfinite(forces)), "Model produced non-finite forces"

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


@pytest.mark.slow
def test_single_xyz_per_head():
    """Test training with multiple XYZ files per head for train, valid and test sets"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Set up file paths - create multiple xyz files for each dataset
        train_xyz_files = [
            os.path.join(temp_dir, f"train_data{i}.xyz") for i in range(1, 2)
        ]  # 3 train files
        valid_xyz_files = [
            os.path.join(temp_dir, f"valid_data{i}.xyz") for i in range(1, 2)
        ]  # 2 valid files
        test_xyz_files = [
            os.path.join(temp_dir, f"test_data{i}.xyz") for i in range(1, 2)
        ]  # 2 test files

        iso_atoms_file = os.path.join(temp_dir, "isolated_atoms.xyz")

        config_path = os.path.join(temp_dir, "config.yaml")
        results_dir = os.path.join(temp_dir, "results")
        checkpoints_dir = os.path.join(temp_dir, "checkpoints")
        model_dir = os.path.join(temp_dir, "models")
        e0s_file = os.path.join(temp_dir, "e0s.json")

        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Set atomic numbers for z_table
        z_table_elements = [1, 6, 7, 8]  # H, C, N, O

        # Create test data for each format
        rng = np.random.RandomState(42)
        seeds = rng.randint(0, 10000, size=10)  # More seeds for multiple files

        # Create isolated atoms for E0s (one of each element)
        isolated_atoms = []
        e0s_dict = {}
        for z in z_table_elements:
            # Create isolated atom
            atom = Atoms(
                numbers=[z], positions=[[0, 0, 0]], cell=np.eye(3) * 10.0, pbc=True
            )
            energy = float(rng.uniform(-5.0, -1.0))  # Random reference energy
            forces = np.zeros((1, 3))
            stress = np.zeros(6)
            calc = SinglePointCalculator(
                atom, energy=energy, forces=forces, stress=stress
            )
            atom.calc = calc
            atom.info["config_type"] = "IsolatedAtom"
            isolated_atoms.append(atom)
            e0s_dict[str(z)] = energy  # Store energy for E0s file

        # Create E0s file
        create_e0s_file(e0s_dict, e0s_file)

        # Create isolated atoms xyz file
        create_xyz_file(isolated_atoms, iso_atoms_file)

        # Create atoms for each train dataset - use different seeds for variety
        train_datasets = []
        for i, file in enumerate(train_xyz_files):
            # Create atoms with different seeds
            atoms = [
                create_test_atoms(num_atoms=5, seed=seeds[i] + j) for j in range(5)
            ]
            create_xyz_file(atoms, file)
            train_datasets.append(atoms)

        # Create atoms for validation datasets
        valid_datasets = []
        for i, file in enumerate(valid_xyz_files):
            atoms = [
                create_test_atoms(num_atoms=5, seed=seeds[i + 3] + j) for j in range(3)
            ]
            create_xyz_file(atoms, file)
            valid_datasets.append(atoms)

        # Create atoms for test datasets
        test_datasets = []
        for i, file in enumerate(test_xyz_files):
            atoms = [
                create_test_atoms(num_atoms=5, seed=seeds[i + 5] + j) for j in range(3)
            ]
            create_xyz_file(atoms, file)
            test_datasets.append(atoms)

        # Create config.yaml for training with multiple xyz files per dataset
        config = {
            "name": "multi_xyz_test",
            "seed": 42,
            "model": "MACE",
            "hidden_irreps": "32x0e",
            "r_max": 5.0,
            "batch_size": 5,
            "max_num_epochs": 2,
            "patience": 5,
            "device": "cpu",
            "energy_weight": 1.0,
            "forces_weight": 10.0,
            "loss": "weighted",
            "optimizer": "adam",
            "default_dtype": "float64",
            "lr": 0.01,
            "swa": False,
            "work_dir": temp_dir,
            "results_dir": results_dir,
            "checkpoints_dir": checkpoints_dir,
            "model_dir": model_dir,
            "E0s": e0s_file,
            "atomic_numbers": str(z_table_elements),
            "heads": {
                "multi_xyz_head": {
                    # Using lists of multiple xyz files for each dataset
                    "train_file": train_xyz_files,
                    "valid_file": valid_xyz_files,
                    "test_file": test_xyz_files,
                    "energy_key": "energy",
                    "forces_key": "forces",
                    "stress_key": "stress",
                },
            },
        }

        # Write config file
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        # Import the modified run_train from our local module
        run_train_script = (
            Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"
        )

        # Run training with subprocess
        cmd = [sys.executable, str(run_train_script), f"--config={config_path}"]

        # Set environment to add the current path to PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(Path(__file__).parent.parent) + ":" + env.get("PYTHONPATH", "")
        )

        # Run the process
        process = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        # Print output for debugging
        print("\n" + "=" * 40 + " STDOUT " + "=" * 40)
        print(process.stdout.decode())
        print("\n" + "=" * 40 + " STDERR " + "=" * 40)
        print(process.stderr.decode())

        # Check that process completed successfully
        assert (
            process.returncode == 0
        ), f"Training failed with error: {process.stderr.decode()}"

        # Check that model was created
        model_path = os.path.join(model_dir, "multi_xyz_test.model")
        assert os.path.exists(model_path), f"Model was not created at {model_path}"

        # Try to load and run the model
        model = torch.load(model_path, map_location="cpu")
        assert model is not None, "Failed to load model"

        # Create a calculator
        calc = MACECalculator(
            model_paths=model_path, device="cpu", head="multi_xyz_head"
        )

        # Run prediction on a test atom
        test_atom = create_test_atoms(num_atoms=5, seed=99999)
        test_atom.calc = calc
        energy = test_atom.get_potential_energy()
        forces = test_atom.get_forces()

        # Assert we got sensible outputs
        assert np.isfinite(energy), "Model produced non-finite energy"
        assert np.all(np.isfinite(forces)), "Model produced non-finite forces"

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


@pytest.mark.slow
def test_multihead_finetuning_different_formats():
    """Test multihead finetuning with different file formats for each head."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Set up file paths
        xyz_file = os.path.join(temp_dir, "finetuning_xyz.xyz")
        h5_folder = os.path.join(temp_dir, "h5_data")
        iso_atoms_file = os.path.join(temp_dir, "isolated_atoms.xyz")

        config_path = os.path.join(temp_dir, "config.yaml")
        results_dir = os.path.join(temp_dir, "results")
        checkpoints_dir = os.path.join(temp_dir, "checkpoints")
        model_dir = os.path.join(temp_dir, "models")
        e0s_file = os.path.join(temp_dir, "e0s.json")

        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Set atomic numbers for z_table
        z_table_elements = [1, 6, 7, 8]  # H, C, N, O

        # Create test data with different seeds
        rng = np.random.RandomState(42)
        seeds = rng.randint(0, 10000, size=3)

        # Create isolated atoms for E0s (one of each element)
        isolated_atoms = []
        e0s_dict = {}
        for z in z_table_elements:
            atom = Atoms(
                numbers=[z], positions=[[0, 0, 0]], cell=np.eye(3) * 10.0, pbc=True
            )
            energy = float(rng.uniform(-5.0, -1.0))
            forces = np.zeros((1, 3))
            stress = np.zeros(6)
            calc = SinglePointCalculator(
                atom, energy=energy, forces=forces, stress=stress
            )
            atom.calc = calc
            atom.info["config_type"] = "IsolatedAtom"
            atom.info["REF_energy"] = energy  # Make sure energy is in the right place
            atom.arrays["REF_forces"] = forces
            atom.info["REF_stress"] = stress
            isolated_atoms.append(atom)
            e0s_dict[str(z)] = energy

        # Create E0s file
        create_e0s_file(e0s_dict, e0s_file)

        # Create isolated atoms xyz file
        create_xyz_file(isolated_atoms, iso_atoms_file)

        # Create XYZ data for xyz_head
        xyz_atoms = [
            create_test_atoms(num_atoms=5, seed=seeds[0] + i) for i in range(30)
        ]
        # Add REF_ properties
        for atom in xyz_atoms:
            atom.info["REF_energy"] = atom.calc.results["energy"]
            atom.arrays["REF_forces"] = atom.calc.results["forces"]
            atom.info["REF_stress"] = atom.calc.results["stress"]
            atom.info["head"] = "xyz_head"  # Assign head
        create_xyz_file(xyz_atoms, xyz_file)

        # Create H5 data for h5_head
        h5_atoms = [
            create_test_atoms(num_atoms=5, seed=seeds[1] + i) for i in range(30)
        ]
        # Add REF_ properties
        for atom in h5_atoms:
            atom.info["REF_energy"] = atom.calc.results["energy"]
            atom.arrays["REF_forces"] = atom.calc.results["forces"]
            atom.info["REF_stress"] = atom.calc.results["stress"]
            atom.info["head"] = "h5_head"  # Assign head

        h5_atoms_xyz = os.path.join(temp_dir, "h5_atoms.xyz")
        create_xyz_file(h5_atoms, h5_atoms_xyz)
        # Include isolated atoms for E0s in the h5 dataset
        all_atoms_for_h5 = h5_atoms + isolated_atoms
        all_atoms_h5_xyz = os.path.join(temp_dir, "all_atoms_for_h5.xyz")
        create_xyz_file(all_atoms_for_h5, all_atoms_h5_xyz)
        create_h5_dataset(all_atoms_h5_xyz, h5_folder)

        # Create config.yaml for multihead finetuning
        heads = {
            "xyz_head": {
                "train_file": xyz_file,
                "valid_fraction": 0.2,
                "energy_key": "REF_energy",
                "forces_key": "REF_forces",
                "stress_key": "REF_stress",
                "E0s": e0s_file,
            },
            "h5_head": {
                "train_file": os.path.join(h5_folder, "train"),
                "valid_file": os.path.join(h5_folder, "val"),
                "energy_key": "REF_energy",
                "forces_key": "REF_forces",
                "stress_key": "REF_stress",
                "E0s": e0s_file,
            },
        }

        yaml_str = "heads:\n"
        for key, value in heads.items():
            yaml_str += f"  {key}:\n"
            for sub_key, sub_value in value.items():
                yaml_str += f"    {sub_key}: {sub_value}\n"

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(yaml_str)

        # Now perform multihead finetuning
        finetuning_params = {
            "name": "multihead_finetuned",
            "config": config_path,
            "foundation_model": "small",  # Use the small foundation model
            "energy_weight": 1.0,
            "forces_weight": 10.0,
            "model": "MACE",
            "hidden_irreps": "128x0e",  # Match foundation model
            "r_max": 5.0,
            "batch_size": 2,
            "max_num_epochs": 2,  # Just do a quick finetuning for test
            "device": "cpu",
            "seed": 42,
            "loss": "weighted",
            "default_dtype": "float64",
            "checkpoints_dir": checkpoints_dir,
            "model_dir": model_dir,
            "results_dir": results_dir,
            "atomic_numbers": "[" + ",".join(map(str, z_table_elements)) + "]",
            "multiheads_finetuning": True,
            "filter_type_pt": "combinations",
            "subselect_pt": "random",
            "num_samples_pt": 10,  # Small number for testing
            "force_mh_ft_lr": True,  # Force using specified learning rate
        }

        # Run finetuning
        run_train_script = (
            Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(Path(__file__).parent.parent) + ":" + env.get("PYTHONPATH", "")
        )

        cmd = [sys.executable, str(run_train_script)]
        for k, v in finetuning_params.items():
            if v is None:
                cmd.append(f"--{k}")
            else:
                cmd.append(f"--{k}={v}")

        # Run the process
        process = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        # Print output for debugging
        print("\n" + "=" * 40 + " STDOUT " + "=" * 40)
        print(process.stdout.decode())
        print("\n" + "=" * 40 + " STDERR " + "=" * 40)
        print(process.stderr.decode())

        # Check that process completed successfully
        assert (
            process.returncode == 0
        ), f"Finetuning failed with error: {process.stderr.decode()}"

        # Check that model was created
        model_path = os.path.join(model_dir, "multihead_finetuned.model")
        assert os.path.exists(model_path), f"Model was not created at {model_path}"

        # Load model and verify it has the expected heads
        model = torch.load(model_path, map_location="cpu")
        assert hasattr(model, "heads"), "Model does not have heads attribute"
        assert set(["xyz_head", "h5_head", "pt_head"]).issubset(
            set(model.heads)
        ), "Expected heads not found in model"

        # Try to run the model with both heads
        # For xyz_head
        calc_xyz = MACECalculator(
            model_paths=model_path,
            device="cpu",
            head="xyz_head",
            default_dtype="float64",
        )
        test_atom = create_test_atoms(num_atoms=5, seed=99999)
        test_atom.calc = calc_xyz
        energy_xyz = test_atom.get_potential_energy()
        forces_xyz = test_atom.get_forces()

        # For h5_head
        calc_h5 = MACECalculator(
            model_paths=model_path,
            device="cpu",
            head="h5_head",
            default_dtype="float64",
        )
        test_atom.calc = calc_h5
        energy_h5 = test_atom.get_potential_energy()
        forces_h5 = test_atom.get_forces()

        # Verify results
        assert np.isfinite(energy_xyz), "xyz_head produced non-finite energy"
        assert np.all(np.isfinite(forces_xyz)), "xyz_head produced non-finite forces"
        assert np.isfinite(energy_h5), "h5_head produced non-finite energy"
        assert np.all(np.isfinite(forces_h5)), "h5_head produced non-finite forces"

    finally:
        # Clean up
        shutil.rmtree(temp_dir)
