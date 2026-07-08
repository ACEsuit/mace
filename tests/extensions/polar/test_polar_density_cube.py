from __future__ import annotations

import argparse
import json

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io.cube import read_cube_data

from mace.cli import polar_density_cube
from mace.cli.polar_density_cube import (
    GRAPH_LONGRANGE_AVAILABLE,
    PotentialInterpolator,
    RealSpaceDensityInterpolator,
    _select_backend,
    coefficient_charge,
    coefficient_dipole,
    cube_quality_metrics,
    make_grid,
    run,
    select_multipoles,
    voxel_volume,
    write_cube_file,
)


def _slab_atoms():
    return Atoms(
        numbers=[1, 1],
        positions=[[1.0, 1.0, 3.0], [2.5, 2.0, 4.0]],
        cell=np.diag([5.0, 5.0, 8.0]),
        pbc=[True, True, False],
    )


def _molecule_atoms(pbc):
    return Atoms(
        numbers=[8, 1, 1],
        positions=[[2.0, 2.0, 2.0], [2.9, 2.0, 2.0], [1.8, 2.8, 2.0]],
        cell=np.diag([6.0, 6.0, 6.0]),
        pbc=pbc,
    )


def _centered_grid(atoms, grid):
    nx, ny, nz = grid
    frac = np.stack(
        np.meshgrid(
            (np.arange(nx) + 0.5) / nx,
            (np.arange(ny) + 0.5) / ny,
            (np.arange(nz) + 0.5) / nz,
            indexing="ij",
        ),
        axis=-1,
    )
    return frac @ atoms.cell.array


def _voxel_volume(atoms, density):
    return voxel_volume(atoms, density)


def _integrated_charge(density, atoms):
    return float(np.sum(density) * _voxel_volume(atoms, density))


def _integrated_dipole(density, coords, atoms):
    return np.sum(density[..., None] * coords, axis=(0, 1, 2)) * _voxel_volume(
        atoms, density
    )


def _coefficient_charge(multipoles):
    return coefficient_charge(multipoles)


def _coefficient_dipole(atoms, multipoles):
    return coefficient_dipole(atoms, multipoles)


def test_select_multipoles_from_calculator_results():
    density_coefficients = np.array([[0.2], [0.8]])
    spin_charge_density = np.array([[[0.7], [0.1]], [[0.3], [0.9]]])
    results = {
        "density_coefficients": density_coefficients,
        "spin_charge_density": spin_charge_density,
    }

    np.testing.assert_allclose(select_multipoles(results, "charge"), [[0.2], [0.8]])
    np.testing.assert_allclose(select_multipoles(results, "alpha"), [[0.7], [0.3]])
    np.testing.assert_allclose(select_multipoles(results, "beta"), [[0.1], [0.9]])
    np.testing.assert_allclose(select_multipoles(results, "spin"), [[0.6], [-0.6]])


@pytest.mark.parametrize(
    ("pbc", "expected"),
    [
        ([False, False, False], "realspace"),
        ([True, False, True], "realspace"),
        ([True, True, False], "fourier"),
        ([True, True, True], "fourier"),
    ],
)
def test_auto_backend_selection(pbc, expected):
    assert _select_backend(_molecule_atoms(pbc), "auto") == expected
    assert _select_backend(_molecule_atoms(pbc), "realspace") == "realspace"


def test_realspace_monopole_matches_analytic_gaussian():
    sigma = 0.5
    atoms = Atoms(
        numbers=[1],
        positions=[[1.0, 1.0, 1.0]],
        cell=np.diag([4.0, 4.0, 4.0]),
        pbc=[False, False, False],
    )
    coords = np.array([[[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]]])

    interpolator = RealSpaceDensityInterpolator(
        sigma=sigma,
        multipoles_max_l=0,
        cutoff_factor=8.0,
    )
    density, _, _ = interpolator(atoms, np.array([[2.0]]), coords)

    norm = 1.0 / ((2.0 * np.pi) ** 1.5 * sigma**3)
    expected = np.array([2.0 * norm, 2.0 * norm * np.exp(-0.5 / sigma**2)])
    np.testing.assert_allclose(density.reshape(-1), expected, rtol=1e-12)


def test_realspace_dipole_matches_analytic_gaussian_derivative():
    sigma = 1.0
    atoms = Atoms(
        numbers=[1],
        positions=[[0.0, 0.0, 0.0]],
        cell=np.diag([4.0, 4.0, 4.0]),
        pbc=[False, False, False],
    )
    coords = np.array([[[[1.0, 0.0, 0.0]]]])

    interpolator = RealSpaceDensityInterpolator(
        sigma=sigma,
        multipoles_max_l=1,
        cutoff_factor=8.0,
    )
    density, _, _ = interpolator(atoms, np.array([[0.0, 0.0, 0.0, 1.0]]), coords)

    norm = 1.0 / ((2.0 * np.pi) ** 1.5 * sigma**3)
    expected = norm * np.exp(-0.5) / sigma**2
    np.testing.assert_allclose(density.item(), expected, rtol=1e-12)


def test_realspace_cube_integrates_to_coefficient_charge_and_dipole():
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[4.0, 4.0, 4.0], [5.0, 4.0, 4.0]],
        cell=np.diag([9.0, 9.0, 9.0]),
        pbc=[False, False, False],
    )
    multipoles = np.array(
        [
            [0.25, 0.01, -0.02, 0.03],
            [-0.10, -0.03, 0.01, -0.02],
        ]
    )
    coords = _centered_grid(atoms, (56, 56, 56))
    density, _, _ = RealSpaceDensityInterpolator(
        sigma=0.55,
        multipoles_max_l=1,
        cutoff_factor=8.0,
        chunk_size=4096,
    )(atoms, multipoles, coords)

    np.testing.assert_allclose(
        _integrated_charge(density, atoms),
        _coefficient_charge(multipoles),
        atol=2e-4,
    )
    np.testing.assert_allclose(
        _integrated_dipole(density, coords, atoms),
        _coefficient_dipole(atoms, multipoles),
        atol=2e-3,
    )

    metrics = cube_quality_metrics(atoms, density, coords, multipoles)
    assert abs(metrics["charge_error"]) < 2e-4
    assert metrics["dipole_error_norm"] < 2e-3
    assert metrics["boundary_max_abs"] < 1e-5
    assert metrics["density_min"] < metrics["density_max"]
    assert metrics["density_l2"] > 0.0


def test_cube_quality_metrics_detect_box_boundary_density():
    small_box = Atoms(
        numbers=[1],
        positions=[[1.0, 1.0, 1.0]],
        cell=np.diag([2.0, 2.0, 2.0]),
        pbc=[False, False, False],
    )
    large_box = Atoms(
        numbers=[1],
        positions=[[3.0, 3.0, 3.0]],
        cell=np.diag([6.0, 6.0, 6.0]),
        pbc=[False, False, False],
    )
    multipoles = np.array([[1.0]])
    interpolator = RealSpaceDensityInterpolator(
        sigma=0.5,
        multipoles_max_l=0,
        cutoff_factor=8.0,
        chunk_size=4096,
    )

    small_coords = _centered_grid(small_box, (24, 24, 24))
    large_coords = _centered_grid(large_box, (24, 24, 24))
    small_density, _, _ = interpolator(small_box, multipoles, small_coords)
    large_density, _, _ = interpolator(large_box, multipoles, large_coords)

    small_metrics = cube_quality_metrics(
        small_box, small_density, small_coords, multipoles
    )
    large_metrics = cube_quality_metrics(
        large_box, large_density, large_coords, multipoles
    )
    assert small_metrics["boundary_max_abs"] > large_metrics["boundary_max_abs"]
    assert small_metrics["boundary_max_abs"] > 1e-3
    assert large_metrics["boundary_max_abs"] < 1e-5


def test_realspace_spin_channel_integrals_match_coefficients():
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[3.5, 3.5, 3.5], [4.4, 3.5, 3.5]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=[False, False, False],
    )
    results = {
        "density_coefficients": np.array([[0.6], [0.4]]),
        "spin_charge_density": np.array(
            [
                [[0.45], [0.15]],
                [[0.25], [0.15]],
            ]
        ),
    }
    coords = _centered_grid(atoms, (48, 48, 48))
    interpolator = RealSpaceDensityInterpolator(
        sigma=0.5,
        multipoles_max_l=0,
        cutoff_factor=8.0,
        chunk_size=4096,
    )

    for quantity in ("alpha", "beta", "spin", "charge"):
        multipoles = select_multipoles(results, quantity)
        density, _, _ = interpolator(atoms, multipoles, coords)
        np.testing.assert_allclose(
            _integrated_charge(density, atoms),
            _coefficient_charge(multipoles),
            atol=2e-4,
        )


def test_realspace_integrated_charge_converges_with_grid():
    atoms = Atoms(
        numbers=[1],
        positions=[[2.7, 3.2, 2.9]],
        cell=np.diag([6.0, 6.0, 6.0]),
        pbc=[False, False, False],
    )
    multipoles = np.array([[1.25]])
    interpolator = RealSpaceDensityInterpolator(
        sigma=0.3,
        multipoles_max_l=0,
        cutoff_factor=8.0,
        chunk_size=4096,
    )

    errors = []
    for n in (8, 12, 20):
        coords = _centered_grid(atoms, (n, n, n))
        density, _, _ = interpolator(atoms, multipoles, coords)
        errors.append(abs(_integrated_charge(density, atoms) - 1.25))

    assert errors[1] < errors[0]
    assert errors[2] < 1e-6


@pytest.mark.polar
@pytest.mark.skipif(
    not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
)
def test_potential_interpolator_writes_cube(tmp_path):
    atoms = _slab_atoms()
    coords = make_grid(atoms, (5, 4, 6))
    multipoles = np.array([[0.4], [-0.2]])

    interpolator = PotentialInterpolator(
        sigma=1.0,
        multipoles_max_l=0,
        kspace_cutoff=3.0,
        device="cpu",
    )
    density, potential, potential_corrected = interpolator(
        atoms,
        multipoles,
        external_field=np.zeros(3),
        fermi_level=0.0,
        coords=coords,
    )

    assert density.shape == (5, 4, 6)
    assert potential.shape == (5, 4, 6)
    assert potential_corrected.shape == (5, 4, 6)
    assert np.all(np.isfinite(density))
    assert np.all(np.isfinite(potential))
    assert np.all(np.isfinite(potential_corrected))

    cube_path = tmp_path / "density.cube"
    write_cube_file(cube_path, atoms, density, "synthetic density")
    cube_data, cube_atoms = read_cube_data(cube_path)

    assert cube_data.shape == density.shape
    assert len(cube_atoms) == len(atoms)
    np.testing.assert_allclose(cube_data, density, atol=1e-8)


@pytest.mark.polar
@pytest.mark.skipif(
    not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
)
def test_fourier_and_realspace_monopole_densities_agree_for_full_pbc():
    atoms = Atoms(
        numbers=[1],
        positions=[[2.0, 2.0, 2.0]],
        cell=np.diag([4.0, 4.0, 4.0]),
        pbc=[True, True, True],
    )
    multipoles = np.array([[1.0]])
    coords = _centered_grid(atoms, (12, 12, 12))

    fourier_density, _, _ = PotentialInterpolator(
        sigma=0.8,
        multipoles_max_l=0,
        kspace_cutoff=7.0,
        device="cpu",
    )(atoms, multipoles, np.zeros(3), 0.0, coords)
    realspace_density, _, _ = RealSpaceDensityInterpolator(
        sigma=0.8,
        multipoles_max_l=0,
        cutoff_factor=6.0,
        chunk_size=4096,
    )(atoms, multipoles, coords)

    np.testing.assert_allclose(
        _integrated_charge(fourier_density, atoms), 1.0, atol=3e-3
    )
    np.testing.assert_allclose(
        _integrated_charge(realspace_density, atoms), 1.0, atol=3e-3
    )
    relative_l2 = np.linalg.norm(fourier_density - realspace_density) / np.linalg.norm(
        realspace_density
    )
    assert relative_l2 < 0.08


class _FakeCoulombEnergy:
    density_smearing_width = 0.8
    density_max_l = 1
    kspace_cutoff = 3.0


class _FakeModel:
    coulomb_energy = _FakeCoulombEnergy()


class _FakePolarCalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self):
        super().__init__()
        self.models = [_FakeModel()]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "density_coefficients": np.array(
                [[-0.4, 0.01, 0.02, 0.03], [0.4, -0.01, 0.0, 0.02]]
            ),
            "spin_charge_density": np.array(
                [
                    [[0.2, 0.0, 0.0, 0.01], [0.6, 0.0, 0.0, -0.01]],
                    [[0.5, 0.0, 0.0, 0.02], [0.1, 0.0, 0.0, -0.02]],
                ]
            ),
        }


def test_cli_run_writes_nonperiodic_cube_end_to_end(tmp_path, monkeypatch):
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]],
        cell=np.diag([5.0, 5.0, 5.0]),
        pbc=[False, False, False],
    )
    configs = tmp_path / "input.xyz"
    output = tmp_path / "spin.cube"
    report = tmp_path / "quality.json"
    ase.io.write(configs, atoms)
    monkeypatch.setattr(
        polar_density_cube, "mace_polar", lambda **_: _FakePolarCalculator()
    )

    written = run(
        argparse.Namespace(
            configs=str(configs),
            model="fake-polar",
            output=str(output),
            index=0,
            quantity="spin",
            grid=(6, 5, 4),
            device="cpu",
            default_dtype="float32",
            sigma=None,
            kspace_cutoff=None,
            backend="auto",
            realspace_cutoff_factor=5.0,
            chunk_size=11,
            subtract_total_charge=False,
            external_field=None,
            fermi_level=None,
            write_potential=False,
            quality_report=str(report),
        )
    )

    assert written == [output, report]
    cube_data, cube_atoms = read_cube_data(output)
    assert cube_data.shape == (6, 5, 4)
    assert len(cube_atoms) == len(atoms)
    assert np.all(np.isfinite(cube_data))

    expected_calc = _FakePolarCalculator()
    expected_calc.calculate(atoms)
    expected_multipoles = select_multipoles(expected_calc.results, "spin")
    expected_density, _, _ = RealSpaceDensityInterpolator(
        sigma=0.8,
        multipoles_max_l=1,
        cutoff_factor=5.0,
        chunk_size=11,
    )(atoms, expected_multipoles, make_grid(atoms, (6, 5, 4)))
    np.testing.assert_allclose(cube_data, expected_density, atol=1e-8)

    with open(report, "r", encoding="utf-8") as fin:
        quality = json.load(fin)
    assert quality["backend"] == "realspace"
    assert quality["grid"] == [6, 5, 4]
    expected_metrics = cube_quality_metrics(
        atoms, expected_density, make_grid(atoms, (6, 5, 4)), expected_multipoles
    )
    np.testing.assert_allclose(
        quality["quantities"]["spin"]["charge_error"],
        expected_metrics["charge_error"],
    )
    assert "boundary_max_abs" in quality["quantities"]["spin"]


@pytest.mark.parametrize("pbc", ([False, False, False], [True, False, True]))
def test_realspace_interpolator_supports_arbitrary_periodicity(tmp_path, pbc):
    atoms = _molecule_atoms(pbc)
    coords = make_grid(atoms, (6, 5, 4))
    multipoles = np.array(
        [
            [-0.4, 0.01, 0.02, 0.03],
            [0.2, -0.01, 0.0, 0.02],
            [0.2, 0.0, -0.02, -0.01],
        ]
    )

    interpolator = RealSpaceDensityInterpolator(
        sigma=0.8,
        multipoles_max_l=1,
        cutoff_factor=4.0,
        chunk_size=17,
    )
    density, potential, potential_corrected = interpolator(atoms, multipoles, coords)

    assert density.shape == (6, 5, 4)
    assert potential is None
    assert potential_corrected is None
    assert np.all(np.isfinite(density))

    cube_path = tmp_path / "realspace_density.cube"
    write_cube_file(cube_path, atoms, density, "real-space density")
    cube_data, cube_atoms = read_cube_data(cube_path)

    assert cube_data.shape == density.shape
    assert len(cube_atoms) == len(atoms)
