"""Tests for MACE-MDP (AtomicDielectricMACE) fine-tuning support.

Covers:
  - test_mdp_finetune_updates_params: verifies that a short fine-tuning run
    completes without error and that at least some model parameters change.
  - test_mdp_finetune_wrong_model_type_raises: verifies that passing a non-MDP
    model type with --finetune_dipoles_polarizabilities raises an error.
"""
import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from ase.atoms import Atoms

from tests.helpers import REPO_ROOT

run_train = REPO_ROOT / "mace" / "cli" / "run_train.py"


# ---------------------------------------------------------------------------
# Shared fixture (same water-molecule setup as test_run_train_dipole_polar.py)
# ---------------------------------------------------------------------------

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
        c.new_array("REF_forces", np.random.normal(0.1, size=c.positions.shape))
        c.info["REF_stress"] = np.random.normal(0.1, size=6)
        c.info["REF_dipoles"] = np.random.normal(0.1, size=3)
        c.info["REF_polarizability"] = np.random.normal(0.1, size=(3, 3))
        fit_configs.append(c)

    return fit_configs


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_subprocess(params: dict, check: bool = True) -> subprocess.CompletedProcess:
    run_env = os.environ.copy()
    sys.path.insert(0, str(REPO_ROOT))
    run_env["PYTHONPATH"] = os.pathsep.join(sys.path)

    cmd = [sys.executable, str(run_train)] + [
        f"--{k}={v}" if v is not None else f"--{k}" for k, v in params.items()
    ]
    return subprocess.run(cmd, env=run_env, check=check, capture_output=True)


# ---------------------------------------------------------------------------
# Common training params for an AtomicDielectricMACE model
# ---------------------------------------------------------------------------

_BASE_MDP_PARAMS = {
    "valid_fraction": 0.05,
    "model": "AtomicDielectricMACE",
    "r_max": 3.5,
    "max_L": 1,
    "num_channels": 4,
    "batch_size": 5,
    "max_num_epochs": 3,
    "device": "cpu",
    "seed": 5,
    "loss": "dipole_polar",
    "error_table": "DipolePolarRMSE",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "dipole_key": "REF_dipoles",
    "polarizability_key": "REF_polarizability",
    "eval_interval": 1,
    "use_reduced_cg": False,
    "compute_polarizability": True,
    "MLP_irreps": "4x0e+4x1o",
    "dipole_weight": 1.0,
    "polarizability_weight": 1.0,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mdp_finetune_updates_params(fitting_configs, tmp_path):
    """
    1. Train a tiny AtomicDielectricMACE model from scratch as the 'foundation'.
    2. Fine-tune it with --finetune_dipoles_polarizabilities=True (naive: all params trainable).
    3. Verify the fine-tuned model exists and at least some parameters changed.
    """
    # Write data
    train_file = tmp_path / "train.xyz"
    ase.io.write(str(train_file), fitting_configs)

    # --- Stage 1: train foundation model ---
    foundation_params = dict(
        _BASE_MDP_PARAMS,
        name="mdp_foundation",
        train_file=str(train_file),
        valid_file=str(train_file),
        model_dir=str(tmp_path / "foundation"),
        checkpoints_dir=str(tmp_path / "foundation_ckpts"),
    )
    _run_subprocess(foundation_params)

    foundation_model_path = tmp_path / "foundation" / "mdp_foundation.model"
    assert foundation_model_path.exists(), "Foundation model not saved"

    # --- Stage 2: fine-tune ---
    ft_params = dict(
        _BASE_MDP_PARAMS,
        name="mdp_finetuned",
        train_file=str(train_file),
        valid_file=str(train_file),
        model_dir=str(tmp_path / "finetuned"),
        checkpoints_dir=str(tmp_path / "finetuned_ckpts"),
        foundation_model=str(foundation_model_path),
        finetune_dipoles_polarizabilities=True,
        multiheads_finetuning=False,
    )
    _run_subprocess(ft_params)

    ft_model_path = tmp_path / "finetuned" / "mdp_finetuned.model"
    assert ft_model_path.exists(), "Fine-tuned model not saved"

    # Verify at least some parameters changed after fine-tuning
    foundation = torch.load(str(foundation_model_path), map_location="cpu", weights_only=False)
    finetuned = torch.load(str(ft_model_path), map_location="cpu", weights_only=False)

    foundation_sd = foundation.state_dict()
    finetuned_sd = finetuned.state_dict()

    any_changed = any(
        not torch.allclose(foundation_sd[k], finetuned_sd[k])
        for k in foundation_sd
        if k in finetuned_sd and foundation_sd[k].shape == finetuned_sd[k].shape
    )
    assert any_changed, "No parameters changed after fine-tuning — training may not have run"


def test_mdp_finetune_wrong_model_type_raises(fitting_configs, tmp_path):
    """Passing --finetune_dipoles_polarizabilities with a non-MDP model should fail."""
    train_file = tmp_path / "train.xyz"
    ase.io.write(str(train_file), fitting_configs)

    # Create a dummy file to satisfy --foundation_model
    dummy_foundation = tmp_path / "dummy.model"
    dummy_foundation.write_text("placeholder")

    params = {
        "name": "bad_ft",
        "model": "MACE",  # wrong model type
        "train_file": str(train_file),
        "valid_file": str(train_file),
        "model_dir": str(tmp_path),
        "r_max": 3.5,
        "batch_size": 5,
        "max_num_epochs": 1,
        "device": "cpu",
        "seed": 5,
        "foundation_model": str(dummy_foundation),
        "finetune_dipoles_polarizabilities": True,
        "multiheads_finetuning": False,
    }
    result = _run_subprocess(params, check=False)
    assert result.returncode != 0, (
        "Expected non-zero exit when using wrong model type with finetune_dipoles_polarizabilities"
    )
    stderr = result.stderr.decode("utf-8", errors="replace")
    assert "AtomicDielectricMACE" in stderr, (
        f"Expected error mentioning AtomicDielectricMACE, got:\n{stderr}"
    )
