"""MAG-1 ships magnetic training in the base native installation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_magnetic_training_and_reload_need_no_legacy_optional_dependency(tmp_path):
    root = Path(__file__).resolve().parents[2]
    probe = r"""
import importlib.abc
import sys
from pathlib import Path

blocked = {"mace", "e3nn", "sphericart", "torch_geometric"}
class RefuseOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("unexpected optional dependency: " + fullname)
sys.meta_path.insert(0, RefuseOptionalImports())
import torch
from test_mace_torch_magnetic_model import configuration, graph, CLUSTER, moments
from mace_torch.cli.run_train import run
from mace_torch.deploy.loader import load_deployed
torch.set_default_dtype(torch.float64)
trained = run(configuration(Path(sys.argv[1])))
loaded = load_deployed(trained.checkpoint_path)
inputs = graph(CLUSTER, moments(4))
before = trained.engine(inputs, compute=("forces", "magforces"))
after = loaded.engine(inputs, compute=("forces", "magforces"))
for name in ("total_energy", "forces", "magforces"):
    actual, expected = after.get(name), before.get(name)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
assert torch.count_nonzero(after.extras["magforces"])
assert not blocked & {name.split(".")[0] for name in sys.modules}
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "packages/mace-torch/tests"), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
