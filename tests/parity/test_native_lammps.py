"""Native LAMMPS trajectories with atom migration and changing neighbors."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units
from mace_torch.deploy.bundle import export_lammps
from mace_torch.deploy.manifest import DeployManifest

from tests.golden.harness import FP32, FP64_CPU_REFERENCE

ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "packages/mace-jax/tests/fixtures/tiny.json"
INPUT = ROOT / "tests/integrations/lammps/native_trajectory.in"


def _trajectory(directory):
    thermo_text = (
        (directory / "log")
        .read_text()
        .rsplit("c_stress[6]", 1)[1]
        .split("Loop time")[0]
    )
    thermo = np.array(
        [
            list(map(float, line.split()))
            for line in thermo_text.splitlines()
            if line.strip() and line.split()[0].isdigit()
        ]
    )
    frames = (directory / "atoms.dump").read_text().split("ITEM: TIMESTEP\n")[1:]
    assert len(frames) == len(thermo) == 15
    for i, frame in enumerate(frames):
        data = np.loadtxt(frame.split("c_neighbors\n")[1].splitlines(), ndmin=2)
        assert i == int(frame.splitlines()[0]) == int(thermo[i, 0])
        atoms = Atoms(
            numbers=np.array([0, 1, 6, 8])[data[:, 1].astype(int)],
            positions=data[:, 3:6],
            cell=[20, 16, 16],
            pbc=True,
        )
        yield atoms, data, thermo[i]


@pytest.mark.slow
@pytest.mark.parametrize(
    "device,backend",
    [
        ("cpu", "reference"),
        pytest.param("cuda", "cueq", marks=[pytest.mark.gpu, pytest.mark.cueq]),
    ],
)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_native_trajectory_after_neighbor_changes_and_migration(
    device, backend, dtype, tmp_path, fp64, isolated
):
    from mace.calculators import MACECalculator as FrozenCalculator

    lmp, plugin = os.getenv("MACE_NATIVE_LMP"), os.getenv("MACE_NATIVE_PLUGIN")
    if not (lmp or plugin):
        pytest.skip(
            "set MACE_NATIVE_LMP and MACE_NATIVE_PLUGIN for the MPI integration"
        )
    assert lmp and Path(lmp).is_file() and plugin and Path(plugin).is_file()
    bundle = export_lammps(
        TINY,
        tmp_path / f"tiny-{dtype}.mace-deploy",
        halo_exchange=True,
        dtype=dtype,
        native_capacity=(32, 100),
        device=device,
        backend=backend,
    )
    assert DeployManifest.read(bundle / "deploy.json").target == "lammps-native"
    reference = FrozenCalculator(
        model_paths=ROOT / "tests/golden/models/tiny_scaleshift.model",
        default_dtype="float64",
        device="cpu",
    )
    row = FP64_CPU_REFERENCE if dtype == "float64" else FP32
    serial_positions = []
    for ranks in (1, 2):
        directory = tmp_path / f"ranks-{ranks}"
        directory.mkdir()
        launch = [os.getenv("MPIEXEC", "mpiexec"), "-n", str(ranks), "env"]
        if sys.platform == "darwin":
            launch += ["DYLD_PRINT_LIBRARIES=1"]
        else:
            launch += ["LD_DEBUG=libs", f"LD_DEBUG_OUTPUT={directory}/libs"]
        cmd = [
            *launch,
            lmp,
            "-var",
            "ranks",
            str(ranks),
            "-var",
            "plugin",
            plugin,
            "-var",
            "model",
            str(bundle / "native.pt2"),
            "-var",
            "out",
            str(directory),
            "-in",
            str(INPUT),
            "-log",
            str(directory / "log"),
        ]
        result = subprocess.run(
            cmd, cwd=directory, text=True, capture_output=True, timeout=180
        )
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        libs = result.stderr + "".join(f.read_text() for f in directory.glob("libs.*"))
        assert "libtorch_cpu" in libs
        assert not re.search(
            r"/(?:libpython[0-9]|libtorch_python|Python\.framework/.*/Python\b)", libs
        )
        if backend == "cueq":
            assert "libcue_ops.so" in libs
        counts, owners = [], []
        errors = {}
        for i, (atoms, data, thermo) in enumerate(_trajectory(directory)):
            reference.calculate(atoms)
            expected = reference.results
            for key, value in (
                ("energy", thermo[1]),
                ("forces", data[:, 6:9]),
                ("energies", data[:, 9]),
                ("stress", -thermo[[2, 3, 4, 7, 6, 5]] * units.bar),
            ):
                np.testing.assert_allclose(
                    value,
                    expected[key],
                    atol=row.atol,
                    rtol=row.rtol,
                    err_msg=f"{key}: step {i} on {ranks} ranks",
                )
                errors[key] = max(
                    errors.get(key, 0.0), float(np.max(np.abs(value - expected[key])))
                )
            counts.append(int(sum(data[:, 10])))
            owners.append(int(data[0, 2]))
            if ranks == 1:
                serial_positions.append(atoms.positions.copy())
            else:
                np.testing.assert_allclose(
                    atoms.positions,
                    serial_positions[i],
                    atol=1e-10 if dtype == "float64" else row.atol,
                    rtol=0,
                )
        print(
            f"{backend} {dtype} ranks={ranks} maxima={errors} "
            f"neighbors={counts} owners={owners}"
        )
        assert min(counts) < counts[0] and max(
            counts[counts.index(min(counts)) :]
        ) > min(counts)
        if ranks == 2:
            assert owners[0] == 0 and owners[-1] == 1
