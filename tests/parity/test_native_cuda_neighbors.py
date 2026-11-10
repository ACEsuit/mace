"""CUDA cell lists reproduce the exact directed periodic neighbour set."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list

pytestmark = [pytest.mark.gpu, pytest.mark.cueq, pytest.mark.bin_libmace]


def invoke(atoms, cutoff, capacity=65536):
    driver = Path(os.environ["MACE_LIBMACE_DRIVER"]).with_name(
        "mace_cuda_neighbor_driver"
    )
    assert driver.is_file(), "build libmace with CUDA and the native test drivers"
    geometry = f"{len(atoms)} {cutoff}\n"
    geometry += " ".join(map(str, atoms.cell.array.ravel())) + "\n"
    geometry += " ".join(map(str, atoms.positions.ravel())) + "\n"
    return subprocess.run(
        [driver, str(capacity)],
        input=geometry,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    "kind", ["bulk", "triclinic", "rotated", "many_images", "cutoff"]
)
def test_periodic_device_neighbors_match_ase_and_repeat(kind):
    rng = np.random.default_rng(27)
    cell = np.array([[8.0, 0, 0], [2.1, 7.0, 0], [-1.3, 1.4, 6.0]])
    cutoff = 2.3
    if kind == "bulk":
        cell = np.diag([8.0, 7.0, 6.0])
    elif kind == "many_images":
        cell *= 0.15
    positions = rng.uniform(-2, 3, (29, 3)) @ cell
    if kind == "rotated":
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        cell = cell @ rotation
        positions = positions @ rotation
    elif kind == "cutoff":
        cell = np.eye(3) * 10
        positions = np.array(
            [[0.0, 0, 0], [cutoff, 0, 0], [0, cutoff - 1e-9, 0], [0, 0, cutoff + 1e-9]]
        )
    atoms = Atoms(
        numbers=np.ones(len(positions), dtype=int),
        positions=positions,
        cell=cell,
        pbc=True,
    )
    i, j, shifts = neighbor_list("ijS", atoms, cutoff)
    expected = sorted(zip(i, j, *shifts.T, strict=True))
    result = invoke(atoms, cutoff)
    assert result.returncode == 0, result.stderr
    actual = np.loadtxt(result.stdout.splitlines(), dtype=np.int64, ndmin=2)
    assert sorted(map(tuple, actual)) == expected
    again = invoke(atoms, cutoff)
    assert again.returncode == 0, again.stderr
    assert again.stdout == result.stdout


def test_device_neighbor_capacity_fails_before_using_truncated_edges():
    atoms = Atoms("HH", positions=[[0, 0, 0], [0.5, 0, 0]], cell=[3, 3, 3], pbc=True)
    result = invoke(atoms, 2.0, capacity=1)
    assert result.returncode != 0
    assert "capacity exceeded" in result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["triclinic", "empty", "no_edges"])
def test_torch_provider_matches_host_edge_set(kind):
    import torch
    from mace_core.neighbors import search_geometry
    from mace_torch.neighbors.native_cuda import NativeCudaBackend, NativeCudaSettings

    library = Path(os.environ["MACE_LIBMACE_DRIVER"]).with_name(
        "libmace_cuda_neighbors.so"
    )
    provider = NativeCudaBackend(NativeCudaSettings(library=library))
    atoms = Atoms(
        "HHO",
        positions=[[0, 0, 0], [0.7, 0.1, 0], [1.3, 0, 0]],
        cell=[[4, 0, 0], [1, 4, 0], [0, 0, 4]],
        pbc=True,
    )
    if kind == "empty":
        atoms = atoms[:0]
    cutoff = 0.1 if kind == "no_edges" else 2.0
    geometry = search_geometry(atoms.positions, cutoff, atoms.pbc, atoms.cell)
    points = torch.tensor(atoms.positions, dtype=torch.float64, device="cuda")
    actual = provider.search(points, cutoff, geometry)
    expected = neighbor_list("ijS", atoms, cutoff)
    rows = torch.cat((actual.edge_index.T, actual.unit_shifts), dim=1)
    assert rows.device.type == "cuda"
    assert sorted(map(tuple, rows.cpu().numpy())) == sorted(
        zip(expected[0], expected[1], *expected[2].T, strict=True)
    )
    torch.testing.assert_close(actual.shifts, actual.unit_shifts.double() @ actual.cell)
