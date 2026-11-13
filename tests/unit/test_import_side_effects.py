"""Loading a checkpoint must not weaken PyTorch's process-wide safe default.

``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`` is read by ``torch.load`` for the whole
interpreter: setting it at import time silently turns arbitrary unpickling back
on for every caller, including code that never asked MACE for a checkpoint.
MACE's model files are whole pickles, so the escape hatch belongs on the load
sites that genuinely need it, which pass ``weights_only=False`` explicitly.
"""

import os
import subprocess
import sys
from pathlib import Path

from tests.golden.harness import MODELS_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_VAR = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"


def test_importing_mace_does_not_set_the_weights_only_escape_hatch():
    # Start from a clean environment: the point is what the import itself does.
    env = {key: value for key, value in os.environ.items() if key != ENV_VAR}
    code = (
        "import os\n"
        "import mace\n"
        f"assert {ENV_VAR!r} not in os.environ, 'import mace set {ENV_VAR}'\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def test_a_pickled_mace_checkpoint_still_loads_without_the_escape_hatch(
    monkeypatch,
):
    monkeypatch.delenv(ENV_VAR, raising=False)

    from mace.calculators.mace import MACECalculator

    calculator = MACECalculator(
        model_paths=str(MODELS_DIR / "tiny_scaleshift.model"),
        device="cpu",
        default_dtype="float32",
    )
    assert len(calculator.models) == 1
