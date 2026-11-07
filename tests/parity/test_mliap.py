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
from mace_torch.train import write_model

from tests.golden.harness import FP32, FP64_CPU_REFERENCE
from tests.integrations.lammps._harness import model_batch
from tests.parity.test_fm_convert_families import converted
from tests.parity.test_native_lammps import ROOT, TINY, _trajectory
from tests.parity.test_output_variants import legacy_model


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


def _bundle(model, directory, **options):
    imported, _ = converted(model, directory)
    checkpoint = write_model(directory / "v1", imported.engine, imported.metadata)
    return export_lammps(checkpoint, directory / "bundle", **options)


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
def test_callback_pair_forces_and_halo_adjoint_match_legacy(
    fp64, isolated, tmp_path, monkeypatch, scale_shift, dtype, layers, backend, device
):
    from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE

    monkeypatch.setenv("MACE_ALLOW_CPU", "true")
    legacy = legacy_model(scale_shift, num_interactions=layers)
    bundle = _bundle(
        legacy,
        tmp_path,
        halo_exchange=True,
        compile_regions=False,
        dtype=dtype,
        backend=backend,
    )
    current = MACEMLIAPUnified(bundle)
    frozen = LAMMPS_MLIAP_MACE(legacy.to(dtype=getattr(torch, dtype)))
    row = FP64_CPU_REFERENCE if dtype == "float64" else FP32
    for distance in (0.9, 1.4, 6.0):
        atoms = Atoms("HOH", positions=[[distance, 0, 0], [0, 0, 0], [0, 1, 0]])
        graph = model_batch(legacy, atoms)
        j, i = graph["edge_index"]
        vectors = graph["positions"][j] - graph["positions"][i] - graph["shifts"]
        # Deliberately interleave senders; both couplings receive exactly this order.
        order = torch.arange(len(j) - 1, -1, -1)
        values = []
        for adapter, where in ((frozen, "cpu"), (current, device)):
            data = DeviceData(
                nlocal=3,
                ntotal=6,
                nlistatoms=3,
                natomneigh=3,
                npairs=len(j),
                elems=graph["node_attrs"].argmax(-1).repeat(2).to(where),
                pair_i=i[order].to(where),
                pair_j=(j[order] + 3).to(where),
                rij=vectors[order].to(where),
                iatoms=torch.arange(3, device=where),
                eatoms=torch.empty(3, dtype=torch.float64, device=where),
                forward_calls=0,
                reverse_calls=0,
            )
            adapter.compute_forces(data)
            assert data.forward_calls == data.reverse_calls == layers - 1
            values.append((data.energy, data.eatoms.cpu().numpy(), data.pair_forces))
        for actual, expected in zip(values[1], values[0], strict=True):
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

    from mace.calculators import MACECalculator as FrozenCalculator

    legacy = legacy_model(True, num_interactions=layers)
    bundle = _bundle(
        legacy, tmp_path, halo_exchange=halo, compile_regions=compiled, dtype=dtype
    )
    adapter = MACEMLIAPUnified(bundle)
    reference = FrozenCalculator(models=legacy, default_dtype="float64", device="cpu")
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
                    err_msg=key,
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
    from mace.calculators import MACECalculator as FrozenCalculator

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
        model_paths=ROOT / "tests/golden/models/tiny_scaleshift.model",
        default_dtype="float64",
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
