import os
import shutil
import tempfile
import subprocess
import sys
import yaml
from pathlib import Path
import numpy as np
import pytest
import torch

from mace.calculators import MACECalculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.atoms import Atoms


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
        pbc=True
    )
    
    # Add random energy, forces and stress
    energy = float(rng.uniform(-15.0, -5.0))
    forces = rng.rand(num_atoms, 3) * 0.5 - 0.25  # Forces between -0.25 and 0.25 eV/Å
    stress = rng.rand(6) * 0.2 - 0.1  # Stress tensor in Voigt notation
    
    # Add calculator to atoms with results
    calc = SinglePointCalculator(
        atoms, energy=energy, forces=forces, stress=stress
    )
    atoms.calc = calc
    
    return atoms


def create_xyz_file(atoms_list, filename):
    """Write a list of atoms to an xyz file."""
    from ase.io import write
    write(filename, atoms_list, format='extxyz')
    return filename

def create_h5_dataset(xyz_file, output_dir, r_max=5.0, seed=42):
    """
    Run MACE's preprocess_data.py script to convert an xyz file to h5 format.
    
    Args:
        xyz_file: Path to the input xyz file
        output_dir: Directory to store the preprocessed h5 files
        r_max: Cutoff radius
        seed: Random seed
        
    Returns:
        The output directory containing the h5 files
    """
    import os
    import subprocess
    from pathlib import Path
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the path to the preprocess_data.py script
    preprocess_script = Path(__file__).parent.parent / "mace" / "cli" / "preprocess_data.py"
    
    # Set up command to run preprocess_data.py
    cmd = [
        sys.executable,
        str(preprocess_script),
        f"--train_file={xyz_file}",
        f"--r_max={r_max}",
        f"--h5_prefix={output_dir}/",
        f"--seed={seed}",
        "--compute_statistics",  # Generate statistics file
        "--num_process=2"        # Create 2 files for testing sharded loading
    ]
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent.parent) + ":" + env.get('PYTHONPATH', '')
    
    # Run the script
    print(f"Running preprocess command: {' '.join(cmd)}")
    process = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    
    # Print output for debugging
    print("Preprocess stdout:", process.stdout.decode())
    print("Preprocess stderr:", process.stderr.decode())
    
    return output_dir


def create_lmdb_dataset(atoms_list, folder_path):
    """Create an LMDB dataset from a list of atoms objects that MACE can read."""
    import os
    import lmdb
    import zlib
    import orjson
    
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Create the LMDB database file
    db_path = os.path.join(folder_path, "data.lmdb")
    
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
            zlib.compress(orjson.dumps(metadata, option=orjson.OPT_SERIALIZE_NUMPY))
        )
        
        # Store nextid
        nextid = len(atoms_list) + 1
        txn.put(
            "nextid".encode("ascii"),
            zlib.compress(orjson.dumps(nextid, option=orjson.OPT_SERIALIZE_NUMPY))
        )
        
        # Store deleted_ids (empty)
        txn.put(
            "deleted_ids".encode("ascii"),
            zlib.compress(orjson.dumps([], option=orjson.OPT_SERIALIZE_NUMPY))
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
                    "config_type": "Default",
                    "head": "Default"
                }
            }
            
            # Store the atom in LMDB
            txn.put(
                f"{id_num}".encode("ascii"),
                zlib.compress(orjson.dumps(dct, option=orjson.OPT_SERIALIZE_NUMPY))
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
        h5_folder = os.path.join(temp_dir, "h5_data")
        lmdb_folder1 = os.path.join(temp_dir, "lmdb_data1_lmdb")  # Add _lmdb suffix for LMDB recognition
        lmdb_folder2 = os.path.join(temp_dir, "lmdb_data2_lmdb")  # Add _lmdb suffix for LMDB recognition
        
        config_path = os.path.join(temp_dir, "config.yaml")
        results_dir = os.path.join(temp_dir, "results")
        checkpoints_dir = os.path.join(temp_dir, "checkpoints")
        model_dir = os.path.join(temp_dir, "models")
        
        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Set atomic numbers for z_table
        z_table_elements = [1, 6, 7, 8]  # H, C, N, O
        
        # Create test data for each format
        rng = np.random.RandomState(42)
        seeds = rng.randint(0, 10000, size=5)
        
        # Create 10 atoms for each dataset
        xyz_atoms1 = [create_test_atoms(num_atoms=5, seed=seeds[0]+i) for i in range(10)]
        xyz_atoms2 = [create_test_atoms(num_atoms=5, seed=seeds[1]+i) for i in range(10)]
        h5_atoms = [create_test_atoms(num_atoms=5, seed=seeds[2]+i) for i in range(10)]
        lmdb_atoms1 = [create_test_atoms(num_atoms=5, seed=seeds[3]+i) for i in range(10)]
        lmdb_atoms2 = [create_test_atoms(num_atoms=5, seed=seeds[4]+i) for i in range(10)]
        
        # Create z_table for dataset creation
        from mace.tools.utils import AtomicNumberTable
        z_table = AtomicNumberTable(z_table_elements)
        
        # Save datasets using utility functions
        create_xyz_file(xyz_atoms1, xyz_file1)
        create_xyz_file(xyz_atoms2, xyz_file2)
        create_h5_dataset(xyz_file2, h5_folder)
        create_lmdb_dataset(lmdb_atoms1, lmdb_folder1)
        create_lmdb_dataset(lmdb_atoms2, lmdb_folder2)
        
        # Create E0s file (isolated atom energies)
        e0s = {1: 0.0, 6: 0.0, 7: 0.0, 8: 0.0}
        e0s_file = os.path.join(temp_dir, "e0s.json")
        import json
        with open(e0s_file, 'w') as f:
            json.dump(e0s, f)
        
        # Create config.yaml for training with proper format specification
        config = {
            "name": "multifile_test",
            "seed": 42,
            "model": "MACE",
            "hidden_irreps": "32x0e",
            "r_max": 5.0,
            "batch_size": 5,
            "max_num_epochs": 2,
            "patience": 5,  # Add patience to avoid early termination
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
                    "stress_key": "REF_stress"
                },
                "head2": {
                    "train_file": h5_folder,
                    "valid_file": xyz_file2,
                    "energy_key": "REF_energy",
                    "forces_key": "REF_forces",
                    "stress_key": "REF_stress"
                }
            }
        }
        
        # Write config file
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Import the modified run_train from our local module
        run_train_script = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"
        
        # Run training with subprocess
        cmd = [
            sys.executable,
            str(run_train_script),
            f"--config={config_path}"
        ]
        
        # Set environment to add the current path to PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent.parent) + ":" + env.get('PYTHONPATH', '')
        
        # Run the process
        process = subprocess.run(
            cmd, 
            env=env, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=False
        )
        
        # Print output for debugging
        print("\n" + "="*40 + " STDOUT " + "="*40)
        print(process.stdout.decode())
        print("\n" + "="*40 + " STDERR " + "="*40)
        print(process.stderr.decode())
        
        # Check that process completed successfully
        assert process.returncode == 0, f"Training failed with error: {process.stderr.decode()}"
        
        # Check that model was created
        model_path = os.path.join(model_dir, "multifile_test.model")
        assert os.path.exists(model_path), f"Model was not created at {model_path}"
        
        # Try to load and run the model
        model = torch.load(model_path, map_location="cpu")
        assert model is not None, "Failed to load model"
        
        # Create a calculator
        calc = MACECalculator(model_paths=model_path, device="cpu")
        
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