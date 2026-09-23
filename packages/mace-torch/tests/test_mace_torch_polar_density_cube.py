"""Writing a charge-aware model's density as cube files.

The frozen tree's twelve cases, against the rewrite's interpolators and command
line: the selection of a density, of an interpolation, the real-space density
against closed forms and against the charge and dipole it has to integrate to,
the quality metrics, the reciprocal-space density and potential, the two
interpolations against each other, and the command line end to end. One more
runs the command line on a trained model rather than a stand-in.
"""

from __future__ import annotations

import argparse
import json

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io.cube import read_cube_data
from conftest import fp64_only
from mace_torch.cli import polar_density_cube
from mace_torch.electrostatics.density_cube import (
    FourierDensity,
    RealSpaceDensity,
    coefficient_charge,
    coefficient_dipole,
    cube_quality_metrics,
    make_grid,
    select_backend,
    select_multipoles,
    voxel_volume,
    write_cube_file,
)


def slab_atoms():
    return Atoms(
        numbers=[1, 1],
        positions=[[1.0, 1.0, 3.0], [2.5, 2.0, 4.0]],
        cell=np.diag([5.0, 5.0, 8.0]),
        pbc=[True, True, False],
    )


def molecule_atoms(pbc):
    return Atoms(
        numbers=[8, 1, 1],
        positions=[[2.0, 2.0, 2.0], [2.9, 2.0, 2.0], [1.8, 2.8, 2.0]],
        cell=np.diag([6.0, 6.0, 6.0]),
        pbc=pbc,
    )


def centered_grid(atoms, grid):
    nx, ny, nz = grid
    fractional = np.stack(
        np.meshgrid(
            (np.arange(nx) + 0.5) / nx,
            (np.arange(ny) + 0.5) / ny,
            (np.arange(nz) + 0.5) / nz,
            indexing="ij",
        ),
        axis=-1,
    )
    return fractional @ atoms.cell.array


def integrated_charge(density, atoms):
    return float(np.sum(density) * voxel_volume(atoms, density))


def integrated_dipole(density, coords, atoms):
    return np.sum(density[..., None] * coords, axis=(0, 1, 2)) * voxel_volume(
        atoms, density
    )


def arguments(**overrides) -> argparse.Namespace:
    settings = {
        "index": 0,
        "quantity": "spin",
        "grid": (6, 5, 4),
        "device": "cpu",
        "default_dtype": "float64",
        "sigma": None,
        "kspace_cutoff": None,
        "backend": "auto",
        "realspace_cutoff_factor": 5.0,
        "chunk_size": 11,
        "subtract_total_charge": False,
        "external_field": None,
        "fermi_level": None,
        "write_potential": False,
        "quality_report": None,
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


# ---------------------------------------------------------------------------
# Choosing what to write, and how
# ---------------------------------------------------------------------------


def test_select_multipoles_from_calculator_results():
    results = {
        "density_coefficients": np.array([[0.2], [0.8]]),
        "spin_charge_density": np.array([[[0.7], [0.1]], [[0.3], [0.9]]]),
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
    assert select_backend(molecule_atoms(pbc), "auto") == expected
    assert select_backend(molecule_atoms(pbc), "realspace") == "realspace"


# ---------------------------------------------------------------------------
# The real-space density
# ---------------------------------------------------------------------------


@fp64_only
def test_realspace_monopole_matches_analytic_gaussian():
    sigma = 0.5
    atoms = Atoms(numbers=[1], positions=[[1.0, 1.0, 1.0]], cell=np.diag([4.0] * 3))
    coords = np.array([[[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]]])
    density, _, _ = RealSpaceDensity(
        sigma=sigma, multipoles_max_l=0, cutoff_factor=8.0
    )(atoms, np.array([[2.0]]), coords)
    norm = 1.0 / ((2.0 * np.pi) ** 1.5 * sigma**3)
    expected = np.array([2.0 * norm, 2.0 * norm * np.exp(-0.5 / sigma**2)])
    np.testing.assert_allclose(density.reshape(-1), expected, rtol=1e-12)


@fp64_only
def test_realspace_dipole_matches_analytic_gaussian_derivative():
    atoms = Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]], cell=np.diag([4.0] * 3))
    coords = np.array([[[[1.0, 0.0, 0.0]]]])
    density, _, _ = RealSpaceDensity(sigma=1.0, multipoles_max_l=1, cutoff_factor=8.0)(
        atoms, np.array([[0.0, 0.0, 0.0, 1.0]]), coords
    )
    norm = 1.0 / (2.0 * np.pi) ** 1.5
    np.testing.assert_allclose(density.item(), norm * np.exp(-0.5), rtol=1e-12)


@fp64_only
def test_realspace_cube_integrates_to_coefficient_charge_and_dipole():
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[4.0, 4.0, 4.0], [5.0, 4.0, 4.0]],
        cell=np.diag([9.0] * 3),
    )
    multipoles = np.array([[0.25, 0.01, -0.02, 0.03], [-0.10, -0.03, 0.01, -0.02]])
    coords = centered_grid(atoms, (56, 56, 56))
    density, _, _ = RealSpaceDensity(
        sigma=0.55, multipoles_max_l=1, cutoff_factor=8.0, chunk_size=4096
    )(atoms, multipoles, coords)
    np.testing.assert_allclose(
        integrated_charge(density, atoms), coefficient_charge(multipoles), atol=2e-4
    )
    np.testing.assert_allclose(
        integrated_dipole(density, coords, atoms),
        coefficient_dipole(atoms, multipoles),
        atol=2e-3,
    )
    metrics = cube_quality_metrics(atoms, density, coords, multipoles)
    assert abs(metrics["charge_error"]) < 2e-4
    assert metrics["dipole_error_norm"] < 2e-3
    assert metrics["boundary_max_abs"] < 1e-5
    assert metrics["density_min"] < metrics["density_max"]
    assert metrics["density_l2"] > 0.0


@fp64_only
def test_cube_quality_metrics_detect_box_boundary_density():
    small = Atoms(numbers=[1], positions=[[1.0] * 3], cell=np.diag([2.0] * 3))
    large = Atoms(numbers=[1], positions=[[3.0] * 3], cell=np.diag([6.0] * 3))
    multipoles = np.array([[1.0]])
    interpolate = RealSpaceDensity(
        sigma=0.5, multipoles_max_l=0, cutoff_factor=8.0, chunk_size=4096
    )
    reports = []
    for atoms in (small, large):
        coords = centered_grid(atoms, (24, 24, 24))
        density, _, _ = interpolate(atoms, multipoles, coords)
        reports.append(cube_quality_metrics(atoms, density, coords, multipoles))
    assert reports[0]["boundary_max_abs"] > reports[1]["boundary_max_abs"]
    assert reports[0]["boundary_max_abs"] > 1e-3
    assert reports[1]["boundary_max_abs"] < 1e-5


@fp64_only
def test_realspace_spin_channel_integrals_match_coefficients():
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[3.5, 3.5, 3.5], [4.4, 3.5, 3.5]],
        cell=np.diag([8.0] * 3),
    )
    results = {
        "density_coefficients": np.array([[0.6], [0.4]]),
        "spin_charge_density": np.array([[[0.45], [0.15]], [[0.25], [0.15]]]),
    }
    coords = centered_grid(atoms, (48, 48, 48))
    interpolate = RealSpaceDensity(
        sigma=0.5, multipoles_max_l=0, cutoff_factor=8.0, chunk_size=4096
    )
    for quantity in ("alpha", "beta", "spin", "charge"):
        multipoles = select_multipoles(results, quantity)
        density, _, _ = interpolate(atoms, multipoles, coords)
        np.testing.assert_allclose(
            integrated_charge(density, atoms), coefficient_charge(multipoles), atol=2e-4
        )


@fp64_only
def test_realspace_integrated_charge_converges_with_grid():
    atoms = Atoms(numbers=[1], positions=[[2.7, 3.2, 2.9]], cell=np.diag([6.0] * 3))
    interpolate = RealSpaceDensity(
        sigma=0.3, multipoles_max_l=0, cutoff_factor=8.0, chunk_size=4096
    )
    errors = []
    for size in (8, 12, 20):
        coords = centered_grid(atoms, (size, size, size))
        density, _, _ = interpolate(atoms, np.array([[1.25]]), coords)
        errors.append(abs(integrated_charge(density, atoms) - 1.25))
    assert errors[1] < errors[0]
    assert errors[2] < 1e-6


@pytest.mark.parametrize("pbc", ([False, False, False], [True, False, True]))
def test_realspace_interpolator_supports_arbitrary_periodicity(tmp_path, pbc):
    atoms = molecule_atoms(pbc)
    multipoles = np.array(
        [
            [-0.4, 0.01, 0.02, 0.03],
            [0.2, -0.01, 0.0, 0.02],
            [0.2, 0.0, -0.02, -0.01],
        ]
    )
    density, potential, corrected = RealSpaceDensity(
        sigma=0.8, multipoles_max_l=1, cutoff_factor=4.0, chunk_size=17
    )(atoms, multipoles, make_grid(atoms, (6, 5, 4)))
    assert density.shape == (6, 5, 4)
    assert potential is None and corrected is None
    assert np.all(np.isfinite(density))
    path = tmp_path / "realspace_density.cube"
    write_cube_file(path, atoms, density, "real-space density")
    data, written = read_cube_data(path)
    assert data.shape == density.shape and len(written) == len(atoms)


# ---------------------------------------------------------------------------
# The reciprocal-space density
# ---------------------------------------------------------------------------


@fp64_only
def test_potential_interpolator_writes_cube(tmp_path):
    atoms = slab_atoms()
    density, potential, corrected = FourierDensity(
        sigma=1.0, multipoles_max_l=0, kspace_cutoff=3.0
    )(
        atoms,
        np.array([[0.4], [-0.2]]),
        external_field=np.zeros(3),
        fermi_level=0.0,
        coords=make_grid(atoms, (5, 4, 6)),
    )
    for array in (density, potential, corrected):
        assert array.shape == (5, 4, 6) and np.all(np.isfinite(array))
    path = tmp_path / "density.cube"
    write_cube_file(path, atoms, density, "synthetic density")
    data, written = read_cube_data(path)
    assert data.shape == density.shape and len(written) == len(atoms)
    np.testing.assert_allclose(data, density, atol=1e-8)


@fp64_only
def test_fourier_and_realspace_monopole_densities_agree_for_full_pbc():
    atoms = Atoms(numbers=[1], positions=[[2.0] * 3], cell=np.diag([4.0] * 3), pbc=True)
    multipoles = np.array([[1.0]])
    coords = centered_grid(atoms, (12, 12, 12))
    fourier, _, _ = FourierDensity(sigma=0.8, multipoles_max_l=0, kspace_cutoff=7.0)(
        atoms, multipoles, np.zeros(3), 0.0, coords
    )
    direct, _, _ = RealSpaceDensity(
        sigma=0.8, multipoles_max_l=0, cutoff_factor=6.0, chunk_size=4096
    )(atoms, multipoles, coords)
    np.testing.assert_allclose(integrated_charge(fourier, atoms), 1.0, atol=3e-3)
    np.testing.assert_allclose(integrated_charge(direct, atoms), 1.0, atol=3e-3)
    assert np.linalg.norm(fourier - direct) / np.linalg.norm(direct) < 0.08


def test_the_reciprocal_space_density_declines_a_molecule():
    with pytest.raises(ValueError, match="real-space"):
        FourierDensity(sigma=1.0, multipoles_max_l=0, kspace_cutoff=3.0)(
            molecule_atoms([False] * 3),
            np.zeros((3, 1)),
            np.zeros(3),
            0.0,
            np.zeros((1, 1, 1, 3)),
        )


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


class _StandIn(Calculator):
    implemented_properties = ("energy",)

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


@fp64_only
def test_cli_run_writes_nonperiodic_cube_end_to_end(tmp_path, monkeypatch):
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]],
        cell=np.diag([5.0] * 3),
    )
    configs, output, report = (
        tmp_path / "input.xyz",
        tmp_path / "spin.cube",
        tmp_path / "quality.json",
    )
    ase.io.write(configs, atoms)
    monkeypatch.setattr(
        polar_density_cube,
        "load_calculator",
        lambda *_: (_StandIn(), (0.8, 1, 3.0)),
    )
    written = polar_density_cube.run(
        arguments(
            configs=str(configs),
            model="stand-in",
            output=str(output),
            quality_report=str(report),
        )
    )
    assert written == [output, report]
    data, written_atoms = read_cube_data(output)
    assert data.shape == (6, 5, 4) and len(written_atoms) == len(atoms)

    stand_in = _StandIn()
    stand_in.calculate(atoms)
    multipoles = select_multipoles(stand_in.results, "spin")
    grid = make_grid(atoms, (6, 5, 4))
    expected, _, _ = RealSpaceDensity(
        sigma=0.8, multipoles_max_l=1, cutoff_factor=5.0, chunk_size=11
    )(atoms, multipoles, grid)
    np.testing.assert_allclose(data, expected, atol=1e-8)
    quality = json.loads(report.read_text())
    assert quality["backend"] == "realspace" and quality["grid"] == [6, 5, 4]
    np.testing.assert_allclose(
        quality["quantities"]["spin"]["charge_error"],
        cube_quality_metrics(atoms, expected, grid, multipoles)["charge_error"],
    )
    assert "boundary_max_abs" in quality["quantities"]["spin"]


@fp64_only
def test_the_command_line_writes_a_trained_model_s_densities(tmp_path):
    """Every quantity and both potentials, through ``main``, for a slab."""
    from mace_torch.train import run_train_stage
    from test_mace_torch_polar_training import built, polar_configuration

    config = polar_configuration(tmp_path, profile="z_slab")
    config = config.model_copy(
        update={
            "electrostatics": config.electrostatics.model_copy(
                update={"slab_normal": 2}
            )
        }
    )
    _, model = built(config)
    run_train_stage(config, model, checkpoint_path=tmp_path / "polar")
    water = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    atoms = Atoms(
        "OH2", positions=water + 2.0, cell=np.diag([5.0, 5.0, 10.0]), pbc=(1, 1, 0)
    )
    atoms.info.update(charge=0.0, spin=1.0)
    configs = tmp_path / "slab.xyz"
    ase.io.write(configs, atoms)
    prefix = tmp_path / "density.cube"
    status = polar_density_cube.main(
        [
            "--configs", str(configs),
            "--model", str(tmp_path / "polar.json"),
            "--output", str(prefix),
            "--quantity", "all",
            "--grid", "6", "6", "10",
            "--default_dtype", "float64",
            "--write_potential",
            "--quality_report", str(tmp_path / "quality.json"),
        ]
    )  # fmt: skip
    assert status == 0
    for quantity in ("charge", "spin", "alpha", "beta"):
        for suffix in ("", "_potential", "_potential_corrected"):
            data, _ = read_cube_data(tmp_path / f"density_{quantity}{suffix}.cube")
            assert data.shape == (6, 6, 10) and np.all(np.isfinite(data))
    quality = json.loads((tmp_path / "quality.json").read_text())
    assert quality["backend"] == "fourier"
    np.testing.assert_allclose(
        quality["quantities"]["charge"]["coefficient_charge"], 0.0, atol=1e-10
    )


def test_the_command_line_keeps_the_frozen_tree_s_eighteen_flags():
    flags = {
        option
        for action in polar_density_cube.build_parser()._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    assert flags == {
        "--configs", "--model", "--output", "--index", "--quantity", "--grid",
        "--device", "--default_dtype", "--sigma", "--kspace_cutoff", "--backend",
        "--realspace_cutoff_factor", "--chunk_size", "--subtract_total_charge",
        "--external_field", "--fermi_level", "--write_potential",
        "--quality_report",
    }  # fmt: skip
