import json
import subprocess
import sys

import numpy as np
import pytest
from ase.io import read

from examples.reference_workflow import nvt, prepare


@pytest.mark.parametrize("kind", ["md22", "ani1x", "water"])
def test_train_evaluate_and_run_nvt(tmp_path, kind):
    config = prepare(kind, tmp_path)
    prefix = [
        sys.executable,
        "-c",
        "import sys; from mace_core.cli import main; "
        "raise SystemExit(main(sys.argv[1:]))",
    ]
    subprocess.run(
        [*prefix, "train", "--config", str(config), "--device", "cpu"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    models = [
        path
        for path in tmp_path.rglob("*.json")
        if path.with_suffix(".safetensors").exists()
        and json.loads(path.read_text()).get("format") == "mace-v1-checkpoint"
    ]
    assert models
    model = models[0]
    subprocess.run(
        [
            *prefix,
            "eval",
            "--model",
            str(model),
            "--configs",
            str(tmp_path / "synthetic.xyz"),
            "--output",
            str(tmp_path / "predictions.xyz"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert len(read(tmp_path / "predictions.xyz", ":")) == len(
        read(tmp_path / "synthetic.xyz", ":")
    )
    if kind == "water":
        result = nvt(model, tmp_path / "synthetic.xyz", tmp_path / "nvt.json")
        assert result["steps"] == 5 and np.isfinite(result["energy_eV"])
        assert np.isfinite(result["positions_A"]).all()
