"""The native Polar contract includes reciprocal forces and cell derivatives."""

import os

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from ase.neighborlist import neighbor_list
from mace_core.config.precision import PrecisionConfig
from mace_torch.calculators import MACECalculator, NativeMACECalculator
from mace_torch.deploy.bundle import export_lammps
from mace_torch.deploy.loader import load_deployed
from mace_torch.deploy.periodic import periodic_export_inputs
from mace_torch.serialization import write_model
from test_mace_torch_polar_training import WATER, built, polar_configuration

from tests.golden.harness import FP32, FP64_CPU_REFERENCE


@pytest.fixture(params=["full_periodic", "per_structure"])
def polar_checkpoint(tmp_path, request):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        config = polar_configuration(
            tmp_path, profile=request.param, interaction_first="plain"
        )
        _, model = built(config)
        yield write_model(tmp_path / "polar", model.engine, model.metadata)
    finally:
        torch.set_default_dtype(previous)


def water_cell():
    atoms = Atoms(
        "OH2",
        positions=WATER + np.array([40.0, 20.0, 10.0]),
        cell=[[6, 0, 0], [0.3, 6.5, 0], [-0.2, 0.1, 7]],
        pbc=True,
    )
    atoms.info.update(
        total_charge=0.25, total_spin=1.5, external_field=np.array([0.01, -0.02, 0.005])
    )
    return atoms


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_periodic_native_matches_energy_force_and_strain_derivatives(
    polar_checkpoint, dtype
):
    deployed = load_deployed(polar_checkpoint, precision=PrecisionConfig(model=dtype))
    atoms = water_cell()
    model, variants = periodic_export_inputs(
        deployed, atoms, nodes=3, edges=128, device="cpu"
    )
    calc = MACECalculator(polar_checkpoint, default_dtype=dtype)
    row = FP64_CPU_REFERENCE if dtype == "float64" else FP32
    for sample in variants:
        sample = tuple(x.detach().requires_grad_(x.is_floating_point()) for x in sample)
        energy, site, forces, virial = model(*sample)
        atoms.positions = sample[3].detach().numpy()
        # The calculator uses the conventional file keys for these inputs.
        atoms.info.update(charge=0.25, spin=1.5)
        calc.calculate(atoms)
        for actual, expected in (
            (energy, calc.results["energy"]),
            (site, calc.results["energies"]),
            (forces, calc.results["forces"]),
            (
                virial,
                -atoms.get_volume()
                * calc.results["stress"][[0, 5, 4, 5, 1, 3, 4, 3, 2]].reshape(3, 3),
            ),
        ):
            np.testing.assert_allclose(
                actual.detach().numpy(), expected, atol=row.atol, rtol=row.rtol
            )
        assert not np.isclose(float(site.sum()), float(energy), atol=1e-8, rtol=0)


def test_periodic_export_captures_all_outputs_and_fixes_system(
    polar_checkpoint, tmp_path, monkeypatch
):
    atoms = water_cell()
    structure = tmp_path / "water.xyz"
    write(structure, atoms)
    captured = {}

    def capture(program, *, package_path, inductor_configs):
        captured.update(inductor_configs["aot_inductor.metadata"])
        assert len(program.graph_signature.output_specs) == 4

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", capture)
    export_lammps(
        polar_checkpoint,
        tmp_path / "native",
        halo_exchange=False,
        structure=structure,
        native_capacity=(3, 128),
    )
    assert captured["mace_evaluation"] == "periodic_system"
    assert captured["mace_comm_width"] == "0"
    assert float(captured["mace_total_charge"]) == 0.25
    np.testing.assert_array_equal(
        np.fromstring(captured["mace_cell"], sep=" ").reshape(3, 3), atoms.cell
    )


@pytest.mark.slow
@pytest.mark.bin_libmace
def test_periodic_native_cpp_matches_python_and_refuses_changed_cell(
    polar_checkpoint, tmp_path
):
    import subprocess

    from tests.integrations.libmace._harness import geometry_text, run_driver, unpack

    if not os.environ.get("MACE_LIBMACE_DRIVER"):
        pytest.skip("set MACE_LIBMACE_DRIVER for the compiled native contract")
    atoms = water_cell()
    structure = tmp_path / "water.xyz"
    write(structure, atoms)
    bundle = export_lammps(
        polar_checkpoint,
        tmp_path / "compiled",
        halo_exchange=False,
        structure=structure,
        native_capacity=(3, 128),
    )
    shifted = atoms.copy()
    shifted.positions[1] += [0.01, 0.02, -0.03]
    changed_neighbors = atoms.copy()
    changed_neighbors.positions[1] += [1.1, -0.4, 0.2]
    cutoff = load_deployed(polar_checkpoint).r_max
    assert len(neighbor_list("i", atoms, cutoff)) != len(
        neighbor_list("i", changed_neighbors, cutoff)
    )
    frames = [atoms, changed_neighbors, shifted]
    for interface in ("cpp", "c"):
        results = run_driver(
            bundle / "native.pt2",
            frames,
            tmp_path / interface,
            interface=interface,
        )
        for index, (frame, result) in enumerate(zip(frames, results[0], strict=True)):
            frame.info.update(charge=0.25, spin=1.5)
            calc = MACECalculator(polar_checkpoint)
            calc.calculate(frame)
            actual = unpack(result, frame)
            for key in ("energy", "energies", "forces"):
                np.testing.assert_allclose(
                    actual[key],
                    calc.results[key],
                    atol=FP64_CPU_REFERENCE.atol,
                    rtol=FP64_CPU_REFERENCE.rtol,
                )
            # The shared driver explicitly disables virials on every third frame.
            expected_virial = (
                -frame.get_volume() * calc.results["stress"][[0, 1, 2, 5, 4, 3]]
                if index % 3 != 2
                else np.zeros(6)
            )
            np.testing.assert_allclose(
                actual["virial"],
                expected_virial,
                atol=FP64_CPU_REFERENCE.atol,
                rtol=FP64_CPU_REFERENCE.rtol,
            )
    atoms.cell[0, 0] *= 1.001
    rejected = subprocess.run(
        [os.environ["MACE_LIBMACE_DRIVER"], str(bundle / "native.pt2")],
        input=geometry_text([atoms]),
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0 and "fixed cell" in rejected.stderr


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.bin_openmm_native
@pytest.mark.parametrize("precision", ["double", "mixed"])
def test_openmm_periodic_device_coordinates(
    polar_checkpoint, tmp_path, monkeypatch, precision
):
    from tests.parity.test_openmm_native import run

    monkeypatch.setenv("MACE_OPENMM_PRECISION", precision)
    atoms = water_cell()
    structure = tmp_path / "water.xyz"
    write(structure, atoms)
    bundle = export_lammps(
        polar_checkpoint,
        tmp_path / "cuda-native",
        halo_exchange=False,
        structure=structure,
        native_capacity=(3, 128),
        backend="cueq",
        device="cuda",
        dtype="float32",
    )
    frames = [atoms.copy() for _ in range(3)]
    frames[1].positions[1] += [1.1, -0.4, 0.2]
    frames[2].positions += np.asarray(atoms.cell[0]) * 2
    rows = run(bundle / "native.pt2", frames, tmp_path, platform="CUDA")
    assert "MACE_OPENMM_DEVICE_IO cuda" in (tmp_path / "native.log").read_text()
    calc = MACECalculator(polar_checkpoint, default_dtype="float32")
    for frame, values in zip(frames, rows, strict=True):
        frame.positions = values[1 : 1 + 3 * len(frame)].reshape(-1, 3)
        frame.info.update(charge=0.25, spin=1.5)
        calc.calculate(frame)
        for actual, expected in (
            (values[0], calc.results["energy"]),
            (values[1 + 3 * len(frame) :].reshape(-1, 3), calc.results["forces"]),
        ):
            np.testing.assert_allclose(actual, expected, atol=FP32.atol, rtol=FP32.rtol)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.cueq
def test_lammps_kokkos_periodic_triclinic(polar_checkpoint, tmp_path):
    from tests.benchmarks.deployment.polar_engines import lammps

    if not (os.getenv("MACE_NATIVE_LMP") or os.getenv("MACE_NATIVE_PLUGIN")):
        pytest.skip("set MACE_NATIVE_LMP and MACE_NATIVE_PLUGIN for Kokkos parity")
    atoms = water_cell()
    atoms.wrap()
    structure = tmp_path / "water.xyz"
    write(structure, atoms)
    bundle = export_lammps(
        polar_checkpoint,
        tmp_path / "cuda-native",
        halo_exchange=False,
        structure=structure,
        native_capacity=(3, 128),
        backend="cueq",
        device="cuda",
        dtype="float32",
    )
    values, frames = lammps(
        tmp_path / "trajectory", atoms, bundle / "native.pt2", steps=3
    )
    calc = MACECalculator(polar_checkpoint, default_dtype="float32")
    for actual, frame in zip(values, frames, strict=True):
        frame.info.update(charge=0.25, spin=1.5)
        calc.calculate(frame)
        expected = {
            **calc.results,
            "virial": -frame.get_volume() * calc.results["stress"][[0, 1, 2, 5, 4, 3]],
        }
        for name, value in actual.items():
            np.testing.assert_allclose(
                value, expected[name], atol=FP32.atol, rtol=FP32.rtol, err_msg=name
            )


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.cueq
def test_ase_native_periodic_coordinates_and_fixed_inputs(polar_checkpoint, tmp_path):
    library = os.getenv("MACE_CUDA_NEIGHBORS_LIBRARY")
    if not library:
        pytest.skip("set MACE_CUDA_NEIGHBORS_LIBRARY for native ASE parity")
    atoms = water_cell()
    structure = tmp_path / "water.xyz"
    write(structure, atoms)
    bundle = export_lammps(
        polar_checkpoint,
        tmp_path / "cuda-native",
        halo_exchange=False,
        structure=structure,
        native_capacity=(3, 128),
        backend="cueq",
        device="cuda",
        dtype="float32",
    )
    compiled = NativeMACECalculator(bundle / "native.pt2", neighbor_library=library)
    reference = MACECalculator(polar_checkpoint, default_dtype="float32")
    for displacement in (0.0, 0.1, 1.1):
        frame = atoms.copy()
        frame.positions[1] += [displacement, -0.4, 0.2]
        frame.info.update(charge=0.25, spin=1.5)
        compiled.calculate(frame)
        reference.calculate(frame)
        for name in ("energy", "energies", "forces", "stress"):
            np.testing.assert_allclose(
                compiled.results[name],
                reference.results[name],
                atol=FP32.atol,
                rtol=FP32.rtol,
                err_msg=name,
            )
    # Changing info alone must invalidate/reject ASE's cached result.
    frame.calc = compiled
    frame.info["charge"] = 1.0
    with pytest.raises(ValueError, match="fixes charge"):
        frame.get_potential_energy()
    frame.info["charge"] = 0.25
    frame.cell[0, 0] += 0.01
    with pytest.raises(ValueError, match="fixed cell"):
        frame.get_potential_energy()
