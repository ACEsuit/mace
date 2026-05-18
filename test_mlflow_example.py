"""Quick smoke test for MLflow integration in MACE training."""

import os
import sys
import tempfile
from pathlib import Path

import ase.io
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

# Build a small fitting dataset
def make_fitting_configs():
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
        fit_configs.append(c)
    return fit_configs


def main():
    tmp_path = Path(tempfile.mkdtemp())
    mlflow_dir = Path("/home/sudhar46/mace/mlflow_runs")
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    db_uri = f"sqlite:///{mlflow_dir}/mlflow.db"
    fitting_configs = make_fitting_configs()

    xyz_path = tmp_path / "fit.xyz"
    ase.io.write(xyz_path, fitting_configs)

    run_train = Path(__file__).parent / "mace" / "cli" / "run_train.py"

    params = {
        "name": "MACE_mlflow_test",
        "valid_fraction": "0.1",
        "energy_weight": "1.0",
        "forces_weight": "10.0",
        "stress_weight": "1.0",
        "model": "MACE",
        "hidden_irreps": "16x0e",
        "r_max": "3.5",
        "batch_size": "5",
        "max_num_epochs": "5",
        "device": "cpu",
        "seed": "5",
        "loss": "stress",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
        "eval_interval": "2",
        "train_file": str(xyz_path),
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        # MLflow options
        "mlflow": None,           # flag, no value
        "mlflow_experiment": "mace_smoke_test",
        "mlflow_run_name": "test_run_1",
        "mlflow_tracking_uri": db_uri,
    }

    cmd = [sys.executable, str(run_train)]
    for k, v in params.items():
        cmd.append(f"--{k}")
        if v is not None:
            cmd.append(str(v))

    print("Running command:")
    print(" ".join(cmd))
    print()

    import subprocess
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n✅ Training completed successfully!")
        # Check MLflow run was recorded
        import mlflow
        mlflow.set_tracking_uri(db_uri)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("mace_smoke_test")
        if exp:
            runs = client.search_runs(exp.experiment_id)
            print(f"MLflow experiment: {exp.name}")
            print(f"Number of runs: {len(runs)}")
            if runs:
                r = runs[0]
                print(f"Run ID: {r.info.run_id}")
                print(f"Status: {r.info.status}")
                print(f"Logged metrics: {list(r.data.metrics.keys())}")
                print(f"Logged params (sample): {dict(list(r.data.params.items())[:5])}")
        else:
            print("⚠️  No MLflow experiment found — check tracking URI.")
    else:
        print(f"\n❌ Training failed with return code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
