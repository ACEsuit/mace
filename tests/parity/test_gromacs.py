"""Actual GROMACS native MD/rerun against frozen legacy, without Python in mdrun."""

import numpy as np
import pytest
from ase.io import read
from tests.golden.harness import FP64_CPU_REFERENCE
from tests.integrations.gromacs._harness import prepare, run, write_g96
from tests.parity.test_fm_convert_legacy import GOLDEN, FIXTURES
from tests.parity.test_libmace import native_package, virial_of

pytestmark = [pytest.mark.bin_gromacs, pytest.mark.slow]


def compare(result, frames, name, row=FP64_CPU_REFERENCE):
    from mace.calculators import MACECalculator

    oracle = MACECalculator(
        model_paths=GOLDEN / "models" / f"{name}.model", default_dtype="float64"
    )
    maxima = {}
    assert len(result["energy"]) == len(frames)
    for step, original in enumerate(frames):
        atoms = original.copy()
        atoms.positions = result["positions"][step]
        atoms.set_cell(result["cell"][step])
        oracle.calculate(atoms)
        for key in ("energy", "forces", *(["virial"] if "virial" in result else [])):
            expected = (
                virial_of(oracle, atoms) if key == "virial" else oracle.results[key]
            )
            actual = result[key][step]
            np.testing.assert_allclose(actual, expected, atol=row.atol, rtol=row.rtol)
            maxima[key] = max(
                maxima.get(key, 0), float(np.max(np.abs(actual - expected)))
            )
    print(f"GROMACS {name}: maxima {maxima}")


@pytest.mark.parametrize("fixture", ["water_cluster", "triclinic_bulk"])
def test_native_gromacs_trajectory(native_package, fixture, tmp_path, fp64, isolated):
    name, _, package = native_package
    atoms = read(FIXTURES / f"{fixture}.xyz")
    if not atoms.pbc.all():
        atoms.set_cell([20, 20, 20])
        atoms.pbc = True
    prepare(tmp_path, atoms, package)
    if fixture == "triclinic_bulk":
        with (tmp_path / "run.mdp").open("a") as file:
            file.write("deform = 0.01 0.02 0.03 0 0.01 0.01\n")
    result = run(tmp_path)
    compare(result, [atoms] * 5, name)


def test_native_gromacs_changed_neighbours_and_cell(
    native_package, tmp_path, fp64, isolated
):
    name, _, package = native_package
    atoms = read(FIXTURES / "water_cluster.xyz")[:3]
    atoms.set_cell([[20, 0, 0], [2, 21, 0], [1, 3, 22]])
    atoms.pbc = True
    frames = []
    for index, distance in enumerate((0.96, 6.0, 0.82)):
        frame = atoms.copy()
        frame.positions[1] = frame.positions[0] + [distance, 0, 0]
        frame.set_cell(atoms.cell * (1 + index * 0.02))
        frames.append(frame)
    prepare(tmp_path, atoms, package)
    write_g96(tmp_path / "scan.g96", frames)
    result = run(tmp_path, rerun="scan.g96")
    compare(result, frames, name)


def migration_frames():
    atoms = read(FIXTURES / "water_cluster.xyz")
    atoms.set_cell([20, 20, 20])
    atoms.pbc = True
    frames = []
    for shift in (0, 9, 13):
        frame = atoms.copy()
        frame.positions[:, 0] += shift
        frame.wrap()
        frames.append(frame)
    return frames


def test_native_gromacs_two_rank_migration(native_package, tmp_path, fp64, isolated):
    name, _, package = native_package
    frames = migration_frames()
    prepare(tmp_path, frames[0], package)
    write_g96(tmp_path / "migration.g96", frames)
    result = run(tmp_path, rerun="migration.g96", ranks=2)
    compare(result, frames, name)


@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_gromacs_cueq(dtype, tmp_path, fp64, isolated):
    from mace_torch.deploy.bundle import export_lammps
    from pathlib import Path
    from tests.golden.harness import FP32

    checkpoint = (
        Path(__file__).resolve().parents[2]
        / "packages/mace-jax/tests/fixtures/tiny.json"
    )
    package = (
        export_lammps(
            checkpoint,
            tmp_path / "native",
            halo_exchange=True,
            native_capacity=(32, 512),
            backend="cueq",
            device="cuda",
            dtype=dtype,
        )
        / "native.pt2"
    )
    frames = migration_frames()
    for ranks in (1, 2):
        directory = tmp_path / f"ranks-{ranks}"
        prepare(directory, frames[0], package)
        write_g96(directory / "migration.g96", frames)
        result = run(directory, ranks=ranks)
        compare(
            result,
            [frames[0]] * 5,
            "tiny_scaleshift",
            row=FP32 if dtype == "float32" else FP64_CPU_REFERENCE,
        )
        result = run(directory, rerun="migration.g96", ranks=ranks)
        compare(
            result,
            frames,
            "tiny_scaleshift",
            row=FP32 if dtype == "float32" else FP64_CPU_REFERENCE,
        )
