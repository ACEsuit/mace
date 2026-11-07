"""Native C++/C and Fortran libmace against the frozen anchors."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from mace_torch.deploy.bundle import export_lammps
from mace_torch.deploy.neutral_io import import_neutral
from mace_torch.train import write_model

from tests.golden.harness import FP32, FP64_CPU_REFERENCE
from tests.integrations.libmace._harness import geometry_text, run_driver, unpack
from tests.parity.test_fm_convert_legacy import (
    CATALOGUE,
    FIXTURES,
    GOLDEN,
    extract_here,
    reference,
)

pytestmark = [pytest.mark.bin_libmace, pytest.mark.slow]


@pytest.fixture(scope="module", params=["tiny_scaleshift", "tiny_mace"])
def native_package(request, tmp_path_factory):
    import torch

    directory = tmp_path_factory.mktemp("libmace")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        name = request.param
        neutral = extract_here(
            GOLDEN / "models" / f"{name}.model", directory / "neutral"
        )
        imported = import_neutral(neutral, CATALOGUE)
        checkpoint = write_model(directory / "v1", imported.engine, imported.metadata)
        bundle = export_lammps(
            checkpoint,
            halo_exchange=True,
            native_capacity=(32, 512),
        )
        yield name, checkpoint, bundle / "native.pt2"
    finally:
        torch.set_default_dtype(previous)


def frames_for(name):
    goldens = json.loads(reference(name).read_text())["fixtures"]
    frames = [read(FIXTURES / f"{fixture}.xyz") for fixture in goldens]
    # Change neighbour topology and atom count within the same exported capacity.
    water = read(FIXTURES / "water_cluster.xyz")[:3]
    water.pbc = False
    for distance in (0.96, 6.0, 0.82):
        moved = water.copy()
        moved.positions[1] = moved.positions[0] + [distance, 0, 0]
        frames.append(moved)
    expanded = read(FIXTURES / "triclinic_bulk.xyz")
    expanded.set_cell(
        np.asarray(expanded.cell) @ np.diag([1.03, 0.98, 1.04]), scale_atoms=True
    )
    frames.append(expanded)
    return frames, goldens


def virial_of(calculator, atoms):
    batch = calculator._atoms_to_batch(atoms)
    output = calculator.models[0](batch.to_dict(), compute_virials=True)
    matrix = output["virials"][0].detach().numpy()
    matrix = 0.5 * (matrix + matrix.T)
    return matrix[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]


@pytest.mark.parametrize("interface", ["cpp", "c", "fortran"])
def test_libmace_matches_legacy_and_committed_goldens(
    native_package, interface, tmp_path, fp64, isolated
):
    from mace_torch.calculators import MACECalculator

    from mace.calculators import MACECalculator as FrozenCalculator

    name, checkpoint, package = native_package
    frames, goldens = frames_for(name)
    frozen = FrozenCalculator(
        model_paths=GOLDEN / "models" / f"{name}.model", default_dtype="float64"
    )
    current = MACECalculator(checkpoint)
    ranks = 2 if interface == "fortran" else 1
    results = run_driver(package, frames, tmp_path, interface=interface, ranks=ranks)
    row = FP64_CPU_REFERENCE
    errors = {}
    for values in results.values():
        for index, (atoms, values_at_step) in enumerate(
            zip(frames, values, strict=True)
        ):
            actual = unpack(values_at_step, atoms)
            frozen.calculate(atoms)
            current.calculate(atoms)
            for key in ("energy", "energies", "forces"):
                for expected in (frozen.results[key], current.results[key]):
                    np.testing.assert_allclose(
                        actual[key], expected, atol=row.atol, rtol=row.rtol
                    )
                    errors[key] = max(
                        errors.get(key, 0),
                        float(np.max(np.abs(actual[key] - expected))),
                    )
            if index % 3 == 2:
                np.testing.assert_array_equal(actual["virial"], 0)
            else:
                expected_virial = virial_of(frozen, atoms)
                np.testing.assert_allclose(
                    actual["virial"], expected_virial, atol=row.atol, rtol=row.rtol
                )
                errors["virial"] = max(
                    errors.get("virial", 0),
                    float(np.max(np.abs(actual["virial"] - expected_virial))),
                )
            # libmace reports a virial, so a zero-volume slab needs no invented
            # volume. Stress comparisons use the physical nondegenerate cells.
            if index % 3 != 2 and atoms.pbc.any() and atoms.cell.rank == 3:
                stress = -actual["virial"][[0, 1, 2, 5, 4, 3]] / atoms.get_volume()
                np.testing.assert_allclose(
                    stress, frozen.results["stress"], atol=row.atol, rtol=row.rtol
                )
                np.testing.assert_allclose(
                    stress, current.results["stress"], atol=row.atol, rtol=row.rtol
                )
            if index < len(goldens):
                expected = list(goldens.values())[index]["outputs"]
                for key in ("energy", "forces"):
                    np.testing.assert_allclose(
                        actual[key],
                        expected[key]["value"],
                        atol=row.atol,
                        rtol=row.rtol,
                    )
    print(f"libmace {name} {interface} ranks={ranks}: maxima {errors}")


def test_libmace_rejects_unknown_elements_and_capacity_overflow(
    native_package, tmp_path
):
    _, _, package = native_package
    frames, _ = frames_for("tiny_scaleshift")
    unknown = frames[0].copy()
    unknown.numbers[0] = 118
    from ase import Atoms

    oversized = Atoms("H32", positions=np.arange(96).reshape(32, 3), cell=[100] * 3)
    dense = Atoms("H31", positions=np.arange(93).reshape(31, 3) * 0.01, cell=[100] * 3)
    for atoms, message in (
        (unknown, "unknown atomic number"),
        (oversized, "node slots"),
        (dense, "edge slots"),
    ):
        result = subprocess.run(
            [os.environ["MACE_LIBMACE_DRIVER"], str(package)],
            input=geometry_text([atoms]),
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert result.returncode != 0
        assert message in result.stderr


@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_libmace_cueq_matches_legacy(dtype, tmp_path, fp64, isolated):
    from mace.calculators import MACECalculator as FrozenCalculator

    root = Path(__file__).resolve().parents[2]
    package = (
        export_lammps(
            root / "packages/mace-jax/tests/fixtures/tiny.json",
            tmp_path / "gpu.mace-deploy",
            halo_exchange=True,
            dtype=dtype,
            backend="cueq",
            device="cuda",
            native_capacity=(32, 512),
        )
        / "native.pt2"
    )
    frames, _ = frames_for("tiny_scaleshift")
    reference_model = FrozenCalculator(
        model_paths=GOLDEN / "models/tiny_scaleshift.model", default_dtype="float64"
    )
    row = FP32 if dtype == "float32" else FP64_CPU_REFERENCE
    errors = {}
    for interface, ranks in (("c", 1), ("fortran", 2)):
        results = run_driver(
            package, frames, tmp_path / interface, interface=interface, ranks=ranks
        )
        for values in results.values():
            for index, (atoms, data) in enumerate(zip(frames, values, strict=True)):
                reference_model.calculate(atoms)
                actual = unpack(data, atoms)
                for key in ("energy", "energies", "forces"):
                    expected = reference_model.results[key]
                    np.testing.assert_allclose(
                        actual[key], expected, atol=row.atol, rtol=row.rtol
                    )
                    errors[key] = max(
                        errors.get(key, 0),
                        float(np.max(np.abs(actual[key] - expected))),
                    )
                if index % 3 == 2:
                    np.testing.assert_array_equal(actual["virial"], 0)
                else:
                    expected = virial_of(reference_model, atoms)
                    np.testing.assert_allclose(
                        actual["virial"], expected, atol=row.atol, rtol=row.rtol
                    )
                    errors["virial"] = max(
                        errors.get("virial", 0),
                        float(np.max(np.abs(actual["virial"] - expected))),
                    )
                if index % 3 != 2 and atoms.pbc.any() and atoms.cell.rank == 3:
                    stress = -actual["virial"][[0, 1, 2, 5, 4, 3]] / atoms.get_volume()
                    np.testing.assert_allclose(
                        stress,
                        reference_model.results["stress"],
                        atol=row.atol,
                        rtol=row.rtol,
                    )
    print(f"libmace cuEq {dtype}: maxima {errors}")
