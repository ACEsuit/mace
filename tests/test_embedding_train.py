import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule

from mace.calculators import MACECalculator


def test_run_train_with_elec_temp(tmp_path):
    """Test run_train.py with electronic temperature embedding."""
    np.random.seed(42)

    # Create configurations with electronic temperature
    configs_with_temp = []
    for i in range(20):
        # Create a water molecule with random displacements
        water = molecule("H2O")
        water.positions += np.random.normal(0, 0.1, size=water.positions.shape)
        water.info["REF_energy"] = -10.0 + np.random.normal(0, 0.1)
        water.new_array(
            "REF_forces", np.random.normal(0, 0.1, size=water.positions.shape)
        )
        # Add electronic temperature (in K)
        water.info["elec_temp"] = 300.0 + i * 10.0  # Vary temperature
        water.info["total_charge"] = np.random.randint(-100, 100)
        water.info["total_spin"] = np.random.randint(0, 100)
        configs_with_temp.append(water)

    # Add isolated atoms
    for atom_num in [1, 8]:  # H and O
        isolated_atom = Atoms(
            symbols=[atom_num], positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=False
        )
        isolated_atom.info["REF_energy"] = 0.0
        isolated_atom.info["config_type"] = "IsolatedAtom"
        isolated_atom.info["elec_temp"] = 300.0
        isolated_atom.info["total_charge"] = 1.0
        isolated_atom.info["total_spin"] = 2.0
        configs_with_temp.append(isolated_atom)

    # Save configurations
    try:
        import ase.io

        ase.io.write(tmp_path / "fit_with_temp.xyz", configs_with_temp)
    except Exception as e:
        print(f"Error writing XYZ file: {e}")
        raise

    # Create config file with feature specs
    config_yaml = """
name: MACE_with_temp
valid_fraction: 0.1
energy_weight: 1.0
forces_weight: 10.0
model: MACE
hidden_irreps: 32x0e
r_max: 5.0
num_interactions: 3
max_num_epochs: 3
batch_size: 4
device: cpu
seed: 42
loss: weighted
energy_key: REF_energy
forces_key: REF_forces
elec_temp_key: elec_temp
total_charge_key: total_charge
total_spin_key: total_spin
embedding_specs:
    elec_temp:
        type: continuous
        per: graph
        in_dim: 1
        emb_dim: 32
    total_charge:
        type: categorical
        per: graph
        in_dim: 1
        emb_dim: 32
        num_classes: 200
        offset: 100
    total_spin:
        type: categorical
        per: graph
        in_dim: 1
        emb_dim: 32
        num_classes: 100
"""

    with open(tmp_path / "config.yaml", "w", encoding="utf-8") as f:
        f.write(config_yaml)

    # Setup model parameters for command line
    mace_params = {
        "config": str(tmp_path / "config.yaml"),
        "work_dir": str(tmp_path),
        "train_file": str(tmp_path / "fit_with_temp.xyz"),
    }

    # Run training
    run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"

    # Make sure the run_train.py script exists
    assert run_train.exists(), f"Could not find run_train.py at {run_train}"

    # Set up environment
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = (
        str(Path(__file__).parent.parent) + ":" + run_env.get("PYTHONPATH", "")
    )

    # Build command
    cmd_parts = [sys.executable, str(run_train)]
    for k, v in mace_params.items():
        cmd_parts.append(f"--{k}={v}")

    cmd = " ".join(cmd_parts)
    print(f"Running command: {cmd}")

    # Run training process
    try:
        completed_process = subprocess.run(
            cmd_parts,
            env=run_env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("STDOUT:", completed_process.stdout)
        print("STDERR:", completed_process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    # Verify the model was created
    model_path = tmp_path / "MACE_with_temp.model"
    assert model_path.exists(), f"Model file was not created at {model_path}"

    # Test the trained model with different temperatures
    calc = MACECalculator(
        model_paths=str(model_path),
        device="cpu",
        info_keys={
            "elec_temp": "elec_temp",
            "total_charge": "total_charge",
            "total_spin": "total_spin",
        },
    )

    # Create two test molecules with different temperatures
    test_mol1 = molecule("H2O")
    test_mol1.info["elec_temp"] = 300.0

    test_mol2 = molecule("H2O")
    test_mol2.positions = test_mol1.positions.copy()  # Ensure identical geometry
    test_mol2.info["elec_temp"] = 600.0

    # Get energies
    test_mol1.calc = calc
    energy1 = test_mol1.get_potential_energy()
    calc.reset()
    test_mol2.calc = calc
    energy2 = test_mol2.get_potential_energy()

    # Verify energies are different (the model responds to temperature)
    assert np.isfinite(energy1), "Energy calculation failed for temperature 300K"
    assert np.isfinite(energy2), "Energy calculation failed for temperature 600K"

    # The energies should be different if the model correctly uses temperature
    assert abs(energy1 - energy2) > 1e-6, "Model is not sensitive to temperature"

    print("Model produces different energies for different temperatures:")
    print(f"Energy at 300K: {energy1:.6f} eV")
    print(f"Energy at 600K: {energy2:.6f} eV")
    print(f"Difference: {abs(energy1 - energy2):.6f} eV")

    # Create two test molecules with different total charges
    test_mol3 = molecule("H2O")
    test_mol3.info["total_charge"] = 50.0  # Set a

    test_mol4 = molecule("H2O")
    test_mol4.positions = test_mol3.positions.copy()  # Ensure identical geometry
    test_mol4.info["total_charge"] = -50.0  # Set a

    # Get energies
    calc.reset()
    test_mol3.calc = calc
    energy3 = test_mol3.get_potential_energy()
    calc.reset()
    test_mol4.calc = calc
    energy4 = test_mol4.get_potential_energy()

    # Verify energies are different (the model responds to total charge)
    assert np.isfinite(energy3), "Energy calculation failed for total charge 50"
    assert np.isfinite(energy4), "Energy calculation failed for total charge -50"
    assert abs(energy3 - energy4) > 1e-6, "Model is not sensitive to total charge"
    print("Model produces different energies for different total charges:")
    print(f"Energy at total charge 50: {energy3:.6f} eV")
    print(f"Energy at total charge -50: {energy4:.6f} eV")
    print(f"Difference: {abs(energy3 - energy4):.6f} eV")

    return True


def test_run_train_with_atom_embed(tmp_path):
    """Test run_train.py with electronic temperature embedding."""
    np.random.seed(42)

    # Create configurations with electronic temperature
    configs_with_atom_embed = []
    for _ in range(20):
        # Create a water molecule with random displacements
        water = molecule("H2O")
        water.positions += np.random.normal(0, 0.1, size=water.positions.shape)
        water.info["REF_energy"] = -10.0 + np.random.normal(0, 0.1)
        water.new_array(
            "REF_forces", np.random.normal(0, 0.1, size=water.positions.shape)
        )
        # Add electronic temperature (in K)
        water.arrays["KEY_quant"] = np.random.normal(0.0, 0.01, size=len(water))
        water.arrays["quant_v"] = np.random.normal(0.0, 0.01, size=(len(water), 2))
        water.arrays["cat"] = np.random.randint(2, size=len(water))
        configs_with_atom_embed.append(water)

    # Add isolated atoms
    for atom_num in [1, 8]:  # H and O
        isolated_atom = Atoms(
            symbols=[atom_num], positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=False
        )
        isolated_atom.info["REF_energy"] = 0.0
        isolated_atom.info["config_type"] = "IsolatedAtom"
        water.arrays["KEY_quant"] = np.zeros(1)
        water.arrays["quant_v"] = np.zeros((1, 2))
        water.arrays["cat"] = np.zeros(1).astype(int)
        configs_with_atom_embed.append(isolated_atom)

    # Save configurations
    try:
        import ase.io

        ase.io.write(tmp_path / "fit_with_atom_embed.xyz", configs_with_atom_embed)
    except Exception as e:
        print(f"Error writing XYZ file: {e}")
        raise

    # Create config file with feature specs
    config_yaml = """
name: MACE_with_atom_embed
valid_fraction: 0.1
energy_weight: 1.0
forces_weight: 10.0
model: MACE
hidden_irreps: 32x0e
r_max: 5.0
num_interactions: 3
max_num_epochs: 3
batch_size: 4
device: cpu
seed: 42
loss: weighted
energy_key: REF_energy
forces_key: REF_forces
embedding_specs:
    cat:
        type: categorical
        per: atom
        in_dim: 1
        emb_dim: 32
        num_classes: 2
    quant:
        key: KEY_quant
        type: continuous
        per: atom
        in_dim: 1
        emb_dim: 32
    quant_v:
        key: quant_v
        type: continuous
        per: atom
        in_dim: 2
        emb_dim: 32
"""

    with open(tmp_path / "config.yaml", "w", encoding="utf-8") as f:
        f.write(config_yaml)

    # Setup model parameters for command line
    mace_params = {
        "config": str(tmp_path / "config.yaml"),
        "work_dir": str(tmp_path),
        "train_file": str(tmp_path / "fit_with_atom_embed.xyz"),
    }

    # Run training
    run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"

    # Make sure the run_train.py script exists
    assert run_train.exists(), f"Could not find run_train.py at {run_train}"

    # Set up environment
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = (
        str(Path(__file__).parent.parent) + ":" + run_env.get("PYTHONPATH", "")
    )

    # Build command
    cmd_parts = [sys.executable, str(run_train)]
    for k, v in mace_params.items():
        cmd_parts.append(f"--{k}={v}")

    cmd = " ".join(cmd_parts)
    print(f"Running command: {cmd}")

    # Run training process
    try:
        completed_process = subprocess.run(
            cmd_parts,
            env=run_env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("STDOUT:", completed_process.stdout)
        print("STDERR:", completed_process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    # Verify the model was created
    model_path = tmp_path / "MACE_with_atom_embed.model"
    assert model_path.exists(), f"Model file was not created at {model_path}"

    # Test the trained model with different temperatures
    calc = MACECalculator(
        model_paths=str(model_path),
        device="cpu",
        arrays_keys={
            "quant": "quant",
            "quant_v": "quant_v",
            "cat": "cat",
        },
    )

    for vary_prop in ["cat", "quant", "quant_v"]:
        # Create two test molecules with different cat values
        test_mol1 = molecule("H2O")
        test_mol1.new_array("quant", np.zeros(len(test_mol1)))
        test_mol1.new_array("quant_v", np.zeros((len(test_mol1), 2)))
        test_mol1.new_array("cat", np.zeros(len(test_mol1)).astype(int))

        test_mol2 = molecule("H2O")
        test_mol2.positions = test_mol1.positions.copy()  # Ensure identical geometry
        test_mol2.new_array("quant", np.zeros(len(test_mol2)))
        test_mol2.new_array("quant_v", np.zeros((len(test_mol2), 2)))
        test_mol2.new_array("cat", np.zeros(len(test_mol2)).astype(int))

        test_mol2.arrays[vary_prop][0] = 1

        # Get energies
        calc.reset()
        test_mol1.calc = calc
        energy1 = test_mol1.get_potential_energy()
        calc.reset()
        test_mol2.calc = calc
        energy2 = test_mol2.get_potential_energy()

        # Verify energies are different (the model responds to temperature)
        assert np.isfinite(energy1), f"Energy calculation failed for {vary_prop} {test_mol1.arrays[vary_prop]}"
        assert np.isfinite(energy2), f"Energy calculation failed for {vary_prop} {test_mol2.arrays[vary_prop]}"

        # The energies should be different if the model correctly uses temperature
        assert abs(energy1 - energy2) > 1e-6, "Model is not sensitive to {vary_prop"

        print(f"Model produces different energies for different {vary_prop}:")
        print(f"Energy at {test_mol1.arrays[vary_prop]}: {energy1:.6f} eV")
        print(f"Energy at {test_mol2.arrays[vary_prop]}: {energy2:.6f} eV")
        print(f"Difference: {abs(energy1 - energy2):.6f} eV")
