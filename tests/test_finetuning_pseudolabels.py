import json
import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms


run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"


def _make_small_real_head_dataset(tmp_path: Path):
    """Create a tiny synthetic dataset for a non-pt head (e.g., DFT)."""
    # Two isolated atoms to define E0s and a few random molecules
    configs = [
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3),
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3),
    ]
    configs[0].info["REF_energy"] = 0.0
    configs[0].info["config_type"] = "IsolatedAtom"
    configs[1].info["REF_energy"] = 0.0
    configs[1].info["config_type"] = "IsolatedAtom"

    rng = np.random.default_rng(5)
    for _ in range(4):
        at = Atoms(
            numbers=[8, 1, 1],
            positions=rng.normal(scale=0.1, size=(3, 3)),
            cell=[5, 5, 5],
            pbc=[True, True, True],
        )
        at.info["REF_energy"] = float(rng.normal(0.0))
        at.arrays["REF_forces"] = rng.normal(0.0, 1.0, size=(3, 3))
        at.info["REF_stress"] = rng.normal(0.0, 1.0, size=6)
        configs.append(at)

    real_path = tmp_path / "fit_real_head.xyz"
    ase.io.write(real_path, configs)
    return real_path


def _write_heads_yaml(tmp_path: Path, real_path: Path):
    heads = {"DFT": {"train_file": str(real_path)}}
    yaml_str = "heads:\n"
    for key, value in heads.items():
        yaml_str += f"  {key}:\n"
        for sub_key, sub_value in value.items():
            yaml_str += f"    {sub_key}: {sub_value}\n"
    filename = tmp_path / "config.yaml"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(yaml_str)
    return filename


def _run_and_collect_pt_metrics(name: str, mace_params: dict):
    # ensure this repo is on PYTHONPATH for the subprocess
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)

    cmd = [sys.executable, str(run_train)]
    for k, v in mace_params.items():
        if v is None:
            cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}={v}")

    completed = subprocess.run(cmd, env=run_env, text=True, check=True)
    assert completed.returncode == 0

    # Parse initial eval metrics from results log
    tag = f"{name}_run-{mace_params.get('seed', 123)}_train.txt"
    results_path = Path(mace_params["results_dir"]) / tag
    assert results_path.exists(), f"Results log not found: {results_path}"

    pt_initial = None
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                rec.get("mode") == "eval"
                and rec.get("head") == "pt_head"
                and rec.get("epoch") is None
            ):
                pt_initial = rec
                break

    assert pt_initial is not None, "Initial pt_head eval metrics not found in log"
    return pt_initial


def _assert_near_zero_metrics(metrics: dict, atol_loss=1e-5, atol_e=1e-4, atol_f=1e-3):
    # Loss and RMSEs should be ~0 because labels are pseudolabels from foundation model
    assert float(metrics.get("loss", 0.0)) <= atol_loss
    # Only check keys if present
    if "rmse_e_per_atom" in metrics:
        assert float(metrics["rmse_e_per_atom"]) <= atol_e
    if "rmse_f" in metrics:
        assert float(metrics["rel_rmse_f"]) <= atol_f
    if "rmse_stress" in metrics:
        assert float(metrics["rmse_stress"]) <= atol_f


@pytest.mark.timeout(120)
def test_initial_metrics_replay_head_mh0(tmp_path):
    real_path = _make_small_real_head_dataset(tmp_path)
    cfg_path = _write_heads_yaml(tmp_path, real_path)

    name = "ft_mh0_pseudo"
    mace_params = {
        "name": name,
        "seed": 42,
        "device": "cpu",
        "default_dtype": "float64",
        "results_dir": str(tmp_path),
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "config": cfg_path,
        "loss": "weighted",
        "error_table": "PerAtomRMSE",
        "batch_size": 2,
        "valid_batch_size": 1,
        "valid_fraction": 0.3,
        "foundation_model": "mh-0",
        "foundation_head": "omat_pbe",
        "multiheads_finetuning": True,
        "pseudolabel_replay": True,
        "pt_train_file": "mp",
        "num_samples_pt": 3,
        "subselect_pt": "random",
        "filter_type_pt": "none",
        "force_mh_ft_lr": True,
        "max_num_epochs": 0,
    }

    metrics = _run_and_collect_pt_metrics(name, mace_params)
    _assert_near_zero_metrics(metrics)


@pytest.mark.timeout(120)
def test_initial_metrics_replay_head_mh1(tmp_path):
    real_path = _make_small_real_head_dataset(tmp_path)
    cfg_path = _write_heads_yaml(tmp_path, real_path)

    name = "ft_mh1_pseudo"
    mace_params = {
        "name": name,
        "seed": 43,
        "device": "cpu",
        "default_dtype": "float64",
        "results_dir": str(tmp_path),
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "config": cfg_path,
        "loss": "weighted",
        "error_table": "PerAtomRMSE",
        "batch_size": 2,
        "valid_batch_size": 1,
        "valid_fraction": 0.3,
        "foundation_model": "mh-1",
        "foundation_head": "omat_pbe",
        "multiheads_finetuning": True,
        "pseudolabel_replay": True,
        "pt_train_file": "mp",
        "num_samples_pt": 3,
        "subselect_pt": "random",
        "filter_type_pt": "none",
        "force_mh_ft_lr": True,
        "max_num_epochs": 0,
    }

    metrics = _run_and_collect_pt_metrics(name, mace_params)
    _assert_near_zero_metrics(metrics)


@pytest.mark.timeout(180)
def test_initial_metrics_replay_head_omol(tmp_path):
    real_path = _make_small_real_head_dataset(tmp_path)
    cfg_path = _write_heads_yaml(tmp_path, real_path)

    name = "ft_omol_pseudo"
    mace_params = {
        "name": name,
        "seed": 44,
        "device": "cpu",
        "default_dtype": "float64",
        "results_dir": str(tmp_path),
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "config": cfg_path,
        "loss": "weighted",
        "error_table": "PerAtomRMSE",
        "batch_size": 2,
        "valid_batch_size": 2,
        "valid_fraction": 0.3,
        "foundation_model": "mace_omol",
        "multiheads_finetuning": True,
        "pseudolabel_replay": True,
        "pt_train_file": "mp",
        "num_samples_pt": 5,
        "subselect_pt": "random",
        "filter_type_pt": "none",
        "force_mh_ft_lr": True,
        "max_num_epochs": 0,
    }

    metrics = _run_and_collect_pt_metrics(name, mace_params)
    _assert_near_zero_metrics(metrics)
