import os
import subprocess
import sys
from pathlib import Path
import tempfile
import json

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms
import torch
from ase import build

from mace.calculators.mace import MACECalculator

run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"


class HeadEmbeddingTracker:
    """Tracks head embedding weights during training"""

    def __init__(self, log_file):
        self.log_file = log_file
        self.weights_history = []

    def save_weights(self, weights):
        """Save weights to file as JSON"""
        weights_np = weights.detach().cpu().numpy()
        with open(self.log_file, "a") as f:
            json.dump({"weights": weights_np.tolist()}, f)
            f.write("\n")

    def load_history(self):
        """Load weights history from file"""
        weights = []
        with open(self.log_file, "r") as f:
            for line in f:
                weights.append(np.array(json.loads(line)["weights"]))
        return weights


def get_model_head_embedding_weights(model_path):
    """Load model and return head embedding weights"""
    model = torch.load(model_path)
    if hasattr(model, "head_embedding"):
        return model.head_embedding.linear.weight.detach().clone()
    return None


@pytest.fixture(name="multihead_configs")
def fixture_multihead_configs():
    # Create water molecule using ASE
    water = build.molecule("H2O")

    # Create two sets of configs with different heads
    fit_configs_head1 = []
    fit_configs_head2 = []

    # First add isolated atoms for both heads
    for z in [1, 8]:  # H and O
        atom = Atoms(numbers=[z], positions=[[0, 0, 0]], cell=[6] * 3)
        atom.info["REF_energy"] = 0.0
        atom.info["config_type"] = "IsolatedAtom"

        # Add to both heads
        atom1 = atom.copy()
        atom1.info["head"] = "head1"
        fit_configs_head1.append(atom1)

        atom2 = atom.copy()
        atom2.info["head"] = "head2"
        fit_configs_head2.append(atom2)

    np.random.seed(42)
    for _ in range(20):
        # Head 1 configs
        c1 = water.copy()
        c1.positions += np.random.normal(0.1, size=c1.positions.shape)
        c1.info["REF_energy"] = np.random.normal(0.1)
        c1.new_array("REF_forces", np.random.normal(0.1, size=c1.positions.shape))
        c1.info["REF_stress"] = np.random.normal(0.1, size=6)
        c1.info["head"] = "head1"
        fit_configs_head1.append(c1)

        # Head 2 configs - slightly different distribution
        c2 = water.copy()
        c2.positions += np.random.normal(
            0.2, size=c2.positions.shape
        )  # Different noise level
        c2.info["REF_energy"] = np.random.normal(0.2)  # Different energy scale
        c2.new_array("REF_forces", np.random.normal(0.2, size=c2.positions.shape))
        c2.info["REF_stress"] = np.random.normal(0.2, size=6)
        c2.info["head"] = "head2"
        fit_configs_head2.append(c2)

    return fit_configs_head1, fit_configs_head2


def test_head_embedding_training(tmp_path, multihead_configs):
    """Test that head embedding parameters change during training"""

    fit_configs_head1, fit_configs_head2 = multihead_configs

    # Save configs to files
    ase.io.write(tmp_path / "fit_head1.xyz", fit_configs_head1)
    ase.io.write(tmp_path / "fit_head2.xyz", fit_configs_head2)

    # Create config yaml with two heads
    heads = {
        "head1": {"train_file": str(tmp_path / "fit_head1.xyz")},
        "head2": {"train_file": str(tmp_path / "fit_head2.xyz")},
    }
    yaml_str = "heads:\n"
    for key, value in heads.items():
        yaml_str += f"  {key}:\n"
        for sub_key, sub_value in value.items():
            yaml_str += f"    {sub_key}: {sub_value}\n"

    with open(tmp_path / "config.yaml", "w", encoding="utf-8") as file:
        file.write(yaml_str)

    # Create tracker for head embedding weights
    weights_file = tmp_path / "head_embedding_weights.jsonl"
    tracker = HeadEmbeddingTracker(weights_file)

    # Training parameters
    mace_params = {
        "name": "MACE",
        "valid_fraction": 0.1,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "model": "MACE",
        "hidden_irreps": "128x0e",
        "r_max": 3.5,
        "batch_size": 5,
        "max_num_epochs": 10,
        "device": "cpu",
        "seed": 42,
        "loss": "weighted",
        "default_dtype": "float64",
        "head_emb_dim": 8,  # Enable head embedding
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "config": str(tmp_path / "config.yaml"),
        "eval_interval": 1,  # Evaluate every epoch to save weights
        "save_all_checkpoints": None,
    }

    # Set up environment
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)

    # Run training
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

    # Before training - get initial weights
    model_path = tmp_path / "MACE.model"

    p = subprocess.run(cmd.split(), env=run_env, check=True)
    assert p.returncode == 0

    # Load checkpoints and compare weights
    checkpoint_files = sorted(list((tmp_path).glob("MACE_run-42_epoch-*.pt")))
    weights_list = []
    node_weights_list = []

    for checkpoint_file in checkpoint_files:
        checkpoint = torch.load(checkpoint_file)
        if "model" in checkpoint:
            model_state = checkpoint["model"]
            # Extract head embedding weights
            head_emb_weight = {
                k: v for k, v in model_state.items() if "head_embedding" in k
            }
            node_emb_weight = {
                k: v for k, v in model_state.items() if "node_embedding" in k
            }
            if head_emb_weight:
                weights = next(iter(head_emb_weight.values()))
                weights_list.append(weights.detach().cpu().numpy())
            if node_emb_weight:
                node_emb_weights = next(iter(node_emb_weight.values()))
                node_weights_list.append(node_emb_weights.detach().cpu().numpy())

    # Check that we have weights from multiple epochs
    assert len(weights_list) > 1, "Not enough checkpoints found"

    # Check if weights are changing between epochs
    weights_changed = False
    for i in range(1, len(weights_list)):
        if not np.allclose(weights_list[i], weights_list[i - 1], rtol=1e-5, atol=1e-5):
            weights_changed = True
            break

    # print("\nHead embedding weight changes between epochs:")
    # for i in range(1, len(weights_list)):
    #     print("weights_list[i]", weights_list[i])
    #     print("weights_list[i-1]", weights_list[i - 1])
    #     diff = np.abs(weights_list[i] - weights_list[i - 1]).mean()
    #     print(f"Epoch {i} -> {i+1}: Mean absolute change = {diff:.6f}")
    # print("\n Node embedding weights changed during epochs")

    # for i in range(1, len(node_weights_list)):
    #     print("node_weights_list[i]", node_weights_list[i])
    #     print("node_weights_list[i-1]", node_weights_list[i - 1])
    #     diff = np.abs(node_weights_list[i] - node_weights_list[i - 1]).mean()
    #     print(f"Epoch {i} -> {i+1}: Mean absolute change = {diff:.6f}")

    assert weights_changed, "Head embedding weights did not change during training"

    # Final model tests
    calc = MACECalculator(model_path, device="cpu")
    test_atoms = build.molecule("H2O")
    test_atoms_clone = build.molecule("H2O")

    # Test head1
    test_atoms.info["head"] = "head1"
    test_atoms.calc = calc
    e1 = test_atoms.get_potential_energy()
    print("=========================================")
    # Test head2
    calc = MACECalculator(model_path, device="cpu")
    test_atoms_clone.info["head"] = "head2"
    test_atoms_clone.calc = calc
    e2 = test_atoms_clone.get_potential_energy()

    # Verify we get different energies for different heads
    assert not np.allclose(
        e1, e2, rtol=1e-3
    ), "Head embedding not affecting predictions"
