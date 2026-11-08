"""Isolated execution against the pinned external electrostatics oracle."""

import os
import subprocess
import sys
from pathlib import Path

PINS = {
    "elec3-reference": "784a510339aacc4412b61b6926ef1a92bca387c0",
    "elec3-legacy": "9e223a1dd2e20eacd8e6bd551775adc7bd3a0682",
    "elec3-longrange": "e0eb8945c21e203656f4e0bbece5d1da488b6753",
}


def run_capture(module: str):
    root = Path(__file__).resolve().parents[3]
    sources = Path(os.environ.get("MACE_ELEC3_SOURCES", str(root / "scripts")))
    for name, revision in PINS.items():
        checkout = sources / name
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision:
            raise RuntimeError(f"{checkout}: expected {revision}, got {actual}")
        subprocess.run(
            ["git", "-C", str(checkout), "diff", "--exit-code", "HEAD"], check=True
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        str(sources / name)
        for name in ("elec3-legacy", "elec3-reference", "elec3-longrange")
    )
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    code = (
        "import mace_scf; import sys; "
        f"sys.path.insert(0,{str(root)!r}); "
        f"from {module} import capture; capture()"
    )
    subprocess.run([sys.executable, "-P", "-c", code], cwd=root, env=env, check=True)
