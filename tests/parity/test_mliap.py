"""V1 ML-IAP against frozen models and the actual ML-IAP Cython callback."""

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms, units
from mace_torch.deploy.bundle import export_lammps
from mace_torch.deploy.mliap import MACEMLIAPUnified

from tests.golden.harness import FP32, FP64_CPU_REFERENCE
from tests.parity.test_native_lammps import ROOT, TINY, _trajectory
from tests.retirement.anchors import model_checkpoint


class DeviceData(SimpleNamespace):
    __module__ = "mliap_unified_couple_kokkos"

    def forward_exchange(self, send, receive, width):
        receive.copy_(send[torch.arange(self.ntotal, device=send.device) % self.nlocal])
        self.forward_calls += 1

    def reverse_exchange(self, send, receive, width):
        receive.zero_()
        receive[: self.nlocal] = send[: self.nlocal] + send[self.nlocal :]
        self.reverse_calls += 1

    def update_pair_forces_gpu(self, forces):
        assert forces.dtype == torch.float64 and forces.is_contiguous()
        self.pair_forces = forces.detach().cpu().numpy().copy()


@pytest.mark.bin_lammps
def test_real_mliap_accumulates_ghost_centered_pair_forces():
    """Pin the external writeback ABI independently of the model's derivatives."""
    from lammps import lammps
    from lammps.mliap import activate_mliappy, load_unified
    from lammps.mliap.mliap_unified_abc import MLIAPUnified

    class PairForceProbe(MLIAPUnified):
        def __init__(self):
            super().__init__(
                element_types=["H"], ndescriptors=1, nparams=1, rcutfac=1.0
            )

        def compute_gradients(self, data):
            pass

        def compute_descriptors(self, data):
            pass

        def compute_forces(self, data):
            centers, neighbors = np.asarray(data.pair_i), np.asarray(data.pair_j)
            edges = np.flatnonzero((centers >= data.nlocal) & (neighbors < data.nlocal))
            assert len(edges), "the probe needs a ghost-centered edge to a local atom"
            pair_forces = np.zeros((data.npairs, 3))
            pair_forces[edges[0]] = [0.5, -0.25, 0.75]
            self.forces = np.zeros((data.ntotal, 3))
            np.add.at(self.forces, centers, pair_forces)
            np.add.at(self.forces, neighbors, -pair_forces)
            data.eatoms = np.zeros(data.nlocal)
            data.energy = 0.0
            data.update_pair_forces(pair_forces)

    probe = PairForceProbe()
    with lammps(cmdargs=["-log", "none", "-screen", "none"]) as lmp:
        activate_mliappy(lmp)
        load_unified(probe)
        for command in (
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            "atom_modify map array",
            "region box block 0 4 0 4 0 4",
            "create_box 1 box",
            "mass 1 1.008",
            "create_atoms 1 single 0.25 2 2",
            "create_atoms 1 single 3.5 2 2",
            "pair_style mliap unified EXISTS 1",
            "pair_coeff * * H",
            "run 0",
        ):
            lmp.command(command)
        tags = lmp.numpy.extract_atom("id")[: len(probe.forces)]
        expected = np.zeros((2, 3))
        np.add.at(expected, tags - 1, probe.forces)
        order = np.argsort(tags[:2])
        np.testing.assert_array_equal(
            lmp.numpy.extract_atom("f")[:2][order],
            expected,
            err_msg=(
                "LAMMPS dropped ghost-centered pair forces. Rebuild with "
                "stable_22Jul2025_update4 (611ca3b7f8525ba802373b04f9e3d632d515f7e3); "
                "the version date alone does not identify the corrected coupling."
            ),
        )


def _bundle(model, directory, **options):
    return export_lammps(model, directory / "bundle", **options)


@pytest.mark.parametrize("scale_shift", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize(
    "backend,device",
    [
        ("reference", "cpu"),
        pytest.param("cueq", "cuda", marks=[pytest.mark.gpu, pytest.mark.cueq]),
    ],
)
def test_callback_pair_forces_and_halo_adjoint_match_eager(
    fp64, isolated, tmp_path, monkeypatch, scale_shift, dtype, layers, backend, device
):
    from mace_torch.calculators import MACECalculator

    monkeypatch.setenv("MACE_ALLOW_CPU", "true")
    checkpoint = model_checkpoint(tmp_path, scale_shift, layers)
    bundle = _bundle(
        checkpoint,
        tmp_path,
        halo_exchange=True,
        compile_regions=False,
        dtype=dtype,
        backend=backend,
    )
    adapter = MACEMLIAPUnified(bundle)
    reference = MACECalculator(checkpoint)
    row = FP64_CPU_REFERENCE if dtype == "float64" else FP32
    for distance in (0.9, 1.4, 6.0):
        atoms = Atoms("HOH", positions=[[distance, 0, 0], [0, 0, 0], [0, 1, 0]])
        graph, _ = reference._graph(atoms, padded=False)
        j, i = graph["edge_index"]
        vectors = graph["positions"][j] - graph["positions"][i] - graph["shifts"]
        order = torch.arange(len(j) - 1, -1, -1)
        data = DeviceData(
            nlocal=3,
            ntotal=6,
            nlistatoms=3,
            natomneigh=3,
            npairs=len(j),
            elems=graph["node_attrs"].argmax(-1).repeat(2).to(device),
            pair_i=i[order].to(device),
            pair_j=(j[order] + 3).to(device),
            rij=vectors[order].to(device=device, dtype=getattr(torch, dtype)),
            iatoms=torch.arange(3, device=device),
            eatoms=torch.empty(3, dtype=torch.float64, device=device),
            forward_calls=0,
            reverse_calls=0,
        )
        adapter.compute_forces(data)
        assert data.forward_calls == data.reverse_calls == layers - 1
        forces = np.zeros((3, 3))
        np.add.at(forces, data.pair_i.cpu().numpy() % 3, data.pair_forces)
        np.add.at(forces, data.pair_j.cpu().numpy() % 3, -data.pair_forces)
        reference.calculate(atoms)
        for actual, expected in (
            (data.energy, reference.results["energy"]),
            (data.eatoms.cpu().numpy(), reference.results["energies"]),
            (forces, reference.results["forces"]),
        ):
            np.testing.assert_allclose(actual, expected, atol=row.atol, rtol=row.rtol)


@pytest.mark.bin_lammps
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("layers,halo", [(1, True), (2, False)])
@pytest.mark.parametrize("isolated_atom", [False, True])
def test_real_mliap_energy_forces_and_stress(
    fp64, isolated, tmp_path, dtype, layers, halo, isolated_atom, compiled=False
):
    from lammps import lammps
    from lammps.mliap import activate_mliappy, load_unified
    from mace_torch.calculators import MACECalculator as FrozenCalculator

    legacy = model_checkpoint(tmp_path, True, layers)
    bundle = _bundle(
        legacy, tmp_path, halo_exchange=halo, compile_regions=compiled, dtype=dtype
    )
    adapter = MACEMLIAPUnified(bundle)
    reference = FrozenCalculator(model_paths=legacy, device="cpu")
    row = FP64_CPU_REFERENCE if dtype == "float64" else FP32
    atoms = Atoms(
        "HOH",
        positions=[[0.2, 3, 3], [7.5, 3, 3], [7.5, 4, 3]],
        cell=[8, 8, 8],
        pbc=True,
    )
    if isolated_atom:
        atoms = Atoms("H", positions=[[4, 4, 4]], cell=[8, 8, 8], pbc=True)
    with lammps(cmdargs=["-log", "none", "-screen", "none"]) as lmp:
        activate_mliappy(lmp)
        load_unified(adapter)
        for command in (
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            "atom_modify map array",
            "region box block 0 8 0 8 0 8",
            "create_box 2 box",
            "mass 1 1.008",
            "mass 2 15.999",
        ):
            lmp.command(command)
        for number, position in zip(atoms.numbers, atoms.positions, strict=True):
            lmp.command(
                f"create_atoms {1 if number == 1 else 2} single "
                + " ".join(map(str, position))
            )
        lmp.command(f"pair_style mliap unified EXISTS {0 if halo else 1}")
        lmp.command("pair_coeff * * H O")
        lmp.command("compute site all pe/atom")
        lmp.command("thermo_style custom step pe pxx pyy pzz pyz pxz pxy")
        for x in (0.2, 1.0, 4.0):
            atoms.positions[0, 0] = x
            lmp.command(f"set atom 1 x {x}")
            lmp.command("run 0")
            reference.calculate(atoms)
            tags = lmp.numpy.extract_atom("id")[: len(atoms)]
            order = np.argsort(tags)
            site = lmp.numpy.extract_compute("site", 1, 1)[order].copy()
            forces = lmp.numpy.extract_atom("f")[: len(atoms)][order].copy()
            stress = (
                -np.array(
                    [
                        lmp.get_thermo(k)
                        for k in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")
                    ]
                )
                * units.bar
            )
            for key, value in (
                ("energy", lmp.get_thermo("pe")),
                ("energies", site),
                ("forces", forces),
                ("stress", stress),
            ):
                np.testing.assert_allclose(
                    value,
                    reference.results[key],
                    atol=row.atol,
                    rtol=row.rtol,
                    err_msg=f"x={x}, {key}",
                )
    if compiled:
        assert all(
            layer.forward.compiled is not None
            for layer in (
                *adapter.deployed.model.backbone.interactions,
                *adapter.deployed.model.backbone.products,
            )
        )


@pytest.mark.bin_lammps
def test_real_mliap_compiled_cpu(fp64, isolated, tmp_path):
    test_real_mliap_energy_forces_and_stress(
        fp64, isolated, tmp_path, "float64", 2, False, False, compiled=True
    )


@pytest.mark.gpu
@pytest.mark.cueq
@pytest.mark.bin_lammps
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_kokkos_mpi_neighbor_changes_and_atom_migration(
    fp64, isolated, tmp_path, dtype, compiled
):
    from mace_torch.calculators import MACECalculator as FrozenCalculator

    if not os.getenv("MACE_MLIAP_KOKKOS"):
        pytest.skip("set MACE_MLIAP_KOKKOS=1 with CUDA KOKKOS/ML-IAP")
    bundle = export_lammps(
        TINY,
        tmp_path / "bundle",
        halo_exchange=True,
        backend="cueq",
        dtype=dtype,
        compile_regions=compiled,
    )
    reference = FrozenCalculator(
        model_paths=ROOT / "tests/retirement/artifacts/tiny_scaleshift.json",
        precision="float64",
        device="cpu",
    )
    row = FP64_CPU_REFERENCE if dtype == "float64" else FP32
    for ranks in (1, 2):
        directory = tmp_path / f"ranks-{ranks}"
        directory.mkdir()
        command = [
            os.getenv("MPIEXEC", "mpiexec"),
            "-n",
            str(ranks),
            sys.executable,
            str(ROOT / "tests/integrations/lammps/mliap_trajectory.py"),
            str(bundle),
            str(directory),
            "--ranks",
            str(ranks),
            "--gpus",
            str(min(ranks, torch.cuda.device_count())),
        ]
        capture = directory / "runtime.log"
        with capture.open("w") as stream:
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=240
            )
        diagnostic = capture.read_text()
        assert result.returncode == 0, diagnostic[-9000:]
        assert "layer compilation failed" not in diagnostic
        counts, owners, errors = [], [], {}
        for step, (atoms, data, thermo) in enumerate(_trajectory(directory)):
            reference.calculate(atoms)
            for key, value in (
                ("energy", thermo[1]),
                ("forces", data[:, 6:9]),
                ("energies", data[:, 9]),
                ("stress", -thermo[[2, 3, 4, 7, 6, 5]] * units.bar),
            ):
                expected = reference.results[key]
                np.testing.assert_allclose(
                    value,
                    expected,
                    atol=row.atol,
                    rtol=row.rtol,
                    err_msg=f"{key} step={step} ranks={ranks}",
                )
                errors[key] = max(
                    errors.get(key, 0.0), float(np.max(np.abs(value - expected)))
                )
            counts.append(int(sum(data[:, 10])))
            owners.append(int(data[0, 2]))
        print(f"ML-IAP {dtype} compiled={compiled} ranks={ranks} maxima={errors}")
        assert min(counts) < counts[0] and max(
            counts[counts.index(min(counts)) :]
        ) > min(counts)
        if ranks == 2:
            assert owners[0] == 0 and owners[-1] == 1
