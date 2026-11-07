"""Real native OpenMM MD and geometry updates against frozen legacy."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from tests.golden.harness import FP32, FP64_CPU_REFERENCE
from tests.integrations.libmace._harness import geometry_text
from tests.parity.test_libmace import native_package
from tests.parity.test_fm_convert_legacy import GOLDEN, FIXTURES

pytestmark = [pytest.mark.bin_openmm_native, pytest.mark.slow]


def run(package, frames, directory, platform="Reference", steps=0):
    env = os.environ.copy()
    if sys.platform == "darwin":
        env["DYLD_PRINT_LIBRARIES"] = "1"
    else:
        env["LD_DEBUG"] = "libs"
        env["LD_DEBUG_OUTPUT"] = str(directory / "libraries")
    result = subprocess.run(
        [os.environ["MACE_OPENMM_DRIVER"], str(package), platform, str(steps)],
        input=geometry_text(frames),
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    (directory / "native.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr[-6000:]
    trace = result.stderr + "".join(
        p.read_text(errors="replace") for p in directory.glob("libraries.*")
    )
    assert "libmace" in trace and "libOpenMM" in trace
    assert not re.search(
        r"/(?:libpython[0-9]|libtorch_python|Python\.framework/.*/Python\b)", trace
    )
    return [
        np.fromstring(line[7:], sep=" ")
        for line in result.stdout.splitlines()
        if line.startswith("RESULT ")
    ]


def compare(rows, frames, name, tolerance):
    from mace.calculators import MACECalculator

    calculator = MACECalculator(
        model_paths=GOLDEN / "models" / f"{name}.model", default_dtype="float64"
    )
    assert len(rows) == len(frames)
    maxima = {"energy": 0.0, "forces": 0.0}
    for values, source in zip(rows, frames, strict=True):
        atoms = source.copy()
        assert values.size == 1 + 6 * len(atoms)
        atoms.positions = values[1 : 1 + 3 * len(atoms)].reshape(-1, 3)
        atoms.calc = calculator
        for key, actual, expected in (
            ("energy", values[0], atoms.get_potential_energy()),
            ("forces", values[1 + 3 * len(atoms) :].reshape(-1, 3), atoms.get_forces()),
        ):
            np.testing.assert_allclose(
                actual, expected, atol=tolerance.atol, rtol=tolerance.rtol
            )
            maxima[key] = max(maxima[key], float(np.max(np.abs(actual - expected))))
    print(f"OpenMM native {name}: {maxima}")


@pytest.mark.parametrize("fixture", ["water_cluster", "triclinic_bulk"])
def test_native_openmm_md_and_geometry(
    native_package, fixture, tmp_path, fp64, isolated
):
    name, _, package = native_package
    atoms = read(FIXTURES / f"{fixture}.xyz")
    compare(
        run(package, [atoms], tmp_path, steps=4), [atoms] * 5, name, FP64_CPU_REFERENCE
    )
    frames = []
    for distance in (0.96, 6.0, 0.82):
        frame = atoms.copy()
        frame.positions[1] = frame.positions[0] + [distance, 0, 0]
        frame.set_cell(np.asarray(frame.cell) * (1 + 0.01 * len(frames)))
        frames.append(frame)
    compare(run(package, frames, tmp_path), frames, name, FP64_CPU_REFERENCE)


@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_openmm_cuda(dtype, tmp_path, fp64, isolated):
    from mace_torch.deploy.bundle import export_lammps

    source = (
        Path(__file__).resolve().parents[2]
        / "packages/mace-jax/tests/fixtures/tiny.json"
    )
    package = (
        export_lammps(
            source,
            tmp_path / "model",
            halo_exchange=True,
            native_capacity=(32, 512),
            backend="cueq",
            device="cuda",
            dtype=dtype,
        )
        / "native.pt2"
    )
    row = FP32 if dtype == "float32" else FP64_CPU_REFERENCE
    for fixture in ("water_cluster", "triclinic_bulk"):
        atoms = read(FIXTURES / f"{fixture}.xyz")
        compare(
            run(package, [atoms], tmp_path, platform="CUDA", steps=4),
            [atoms] * 5,
            "tiny_scaleshift",
            row,
        )
