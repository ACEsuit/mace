"""The legacy converter's boundary: what crosses it, and what ships.

The converter is two halves. The extraction script runs against the legacy
package and writes files; everything on the v1 side reads those files and
nothing else. These checks hold the line from both sides, and check that the
script a user needs actually reaches them in the wheel.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "mace_torch/legacy-extractor/extract_legacy.py"


def test_the_v1_side_imports_no_legacy_module():
    """In a fresh interpreter, so nothing another test imported can hide it."""
    probe = (
        "import sys\n"
        "import mace_torch.cli.convert_legacy\n"
        "import mace_torch.deploy.legacy, mace_torch.deploy.neutral_io\n"
        "import mace_torch.deploy.reference\n"
        "loaded = sorted(m for m in sys.modules if m == 'mace' or m.startswith('mace.'))\n"
        "print(loaded)\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_the_reference_tolerance_is_the_harness_row():
    """One bound for a committed float64 reference, whichever way the check is
    reached: the harness's table or the converter's default."""
    pytest.importorskip("mace_torch")
    from mace_torch.deploy.reference import REFERENCE_ATOL, REFERENCE_RTOL

    from tests.golden.harness import tolerance

    row = tolerance("fp64_cpu_reference")
    assert (REFERENCE_ATOL, REFERENCE_RTOL) == (row.atol, row.rtol)


@pytest.mark.skipif(shutil.which("uv") is None, reason="building the wheel needs uv")
def test_the_wheel_ships_the_extraction_script(tmp_path):
    """A user with only v1 installed converts through the script in the wheel.
    A packaging rule that dropped it would leave the packaged mode with
    nothing to run, and nothing else would notice."""
    completed = subprocess.run(
        [
            "uv",
            "build",
            "--no-build-isolation", "--python", sys.executable,
            "--wheel",
            "--quiet",
            str(REPO_ROOT / "packages" / "mace-torch"),
            "-o",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    (wheel,) = tmp_path.glob("*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    assert SCRIPT in names, sorted(name for name in names if "deploy" in name)
    # Runtime artifacts are loaded explicitly. In particular the retired ANI-CC
    # pickle must not arrive through wheel contents.
    assert not any(name.endswith((".model", ".pt", ".safetensors")) for name in names)
    assert not any("ani500k" in name or "torch_geometric" in name for name in names)

    # Inspect the built artifact: declaring an extra in the source tree alone
    # does not establish that a user installing the wheel receives it.
    from packaging.requirements import Requirement

    with zipfile.ZipFile(wheel) as archive:
        metadata_name, = [name for name in names if name.endswith(".dist-info/METADATA")]
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    for extra, dependency in {
        "fpsample": "fpsample",
        "cueq": "cuequivariance-torch",
        "cueq-cuda-12": "cuequivariance-ops-torch-cu12",
        "cueq-cuda-13": "cuequivariance-ops-torch-cu13",
        "oeq": "openequivariance",
    }.items():
        assert extra in metadata.get_all("Provides-Extra", [])
        optional = [item for item in requirements if item.marker is not None
                    and item.marker.evaluate({"extra": extra})
                    and not item.marker.evaluate({"extra": ""})]
        assert [item.name for item in optional] == [dependency]
        if extra.startswith("cueq-cuda"):
            assert "0.11.0" not in optional[0].specifier
    assert "magnetic" not in metadata.get_all("Provides-Extra", [])
    assert not {"sphericart-torch", "torch-geometric"} & {item.name for item in requirements}
