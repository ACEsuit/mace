# Copyright (c) MACE contributors
# Script for exporting MACE-Polar density coefficients to Gaussian cube files
# This program is distributed under the MIT License (see MIT.md)
# pylint: disable=wrong-import-position
import argparse
import json
import os
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mace-matplotlib")
)

import ase.io
import numpy as np
import torch
from ase.io.cube import write_cube

from mace.calculators.foundations_models import mace_polar

try:
    from graph_longrange.features import (
        apply_coulomb_kernel_batch,
        assemble_fourier_series_batch,
    )
    from graph_longrange.gto_utils import GTOBasis, gto_basis_kspace_cutoff
    from graph_longrange.kspace import (
        compute_k_vectors_flat,
        evaluate_fourier_series_at_points_flat,
    )
    from graph_longrange.utils import FIELD_CONSTANT

    GRAPH_LONGRANGE_AVAILABLE = True
    GRAPH_LONGRANGE_IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:
    GRAPH_LONGRANGE_AVAILABLE = False
    GRAPH_LONGRANGE_IMPORT_ERROR = exc


@contextmanager
def use_dtype(dtype=torch.float64):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)


def make_grid(atoms, grid: Sequence[int]) -> np.ndarray:
    if len(grid) != 3:
        raise ValueError("grid must contain three integers")
    nx, ny, nz = (int(n) for n in grid)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("grid sizes must be positive")
    frac = np.stack(
        np.meshgrid(
            np.linspace(0.0, 1.0, nx, endpoint=False),
            np.linspace(0.0, 1.0, ny, endpoint=False),
            np.linspace(0.0, 1.0, nz, endpoint=False),
            indexing="ij",
        ),
        axis=-1,
    )
    return frac @ atoms.cell.array


def select_multipoles(results: dict, quantity: str) -> np.ndarray:
    if quantity == "charge":
        return np.asarray(results["density_coefficients"])

    spin_charge_density = np.asarray(results["spin_charge_density"])
    if quantity == "alpha":
        return spin_charge_density[:, 0, :]
    if quantity == "beta":
        return spin_charge_density[:, 1, :]
    if quantity == "spin":
        return spin_charge_density[:, 0, :] - spin_charge_density[:, 1, :]
    raise ValueError(f"Unknown density quantity: {quantity}")


def write_cube_file(path: str | Path, atoms, data: np.ndarray, comment: str) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        write_cube(fout, atoms, data=np.asarray(data), comment=comment)


def voxel_volume(atoms, density: np.ndarray) -> float:
    return float(abs(np.linalg.det(atoms.cell.array)) / np.prod(density.shape))


def coefficient_charge(multipoles: np.ndarray) -> float:
    multipoles = np.asarray(multipoles)
    return float(np.sum(multipoles[:, 0]))


def coefficient_dipole(atoms, multipoles: np.ndarray) -> np.ndarray:
    multipoles = np.asarray(multipoles)
    charges = multipoles[:, 0]
    dipoles = np.zeros((multipoles.shape[0], 3))
    if multipoles.shape[1] > 1:
        dipoles = multipoles[:, 1:4][:, [2, 0, 1]]
    return np.sum(atoms.positions * charges[:, None], axis=0) + np.sum(dipoles, axis=0)


def cube_boundary_max_abs(density: np.ndarray) -> float:
    faces = [
        density[0, :, :],
        density[-1, :, :],
        density[:, 0, :],
        density[:, -1, :],
        density[:, :, 0],
        density[:, :, -1],
    ]
    return float(max(np.max(np.abs(face)) for face in faces))


def cube_quality_metrics(
    atoms,
    density: np.ndarray,
    coords: np.ndarray,
    multipoles: Optional[np.ndarray] = None,
) -> dict:
    density = np.asarray(density)
    coords = np.asarray(coords)
    dvol = voxel_volume(atoms, density)
    charge = float(np.sum(density) * dvol)
    dipole = np.sum(density[..., None] * coords, axis=(0, 1, 2)) * dvol
    metrics = {
        "voxel_volume": dvol,
        "integrated_charge": charge,
        "integrated_dipole": dipole,
        "density_min": float(np.min(density)),
        "density_max": float(np.max(density)),
        "density_l2": float(np.sqrt(np.sum(density * density) * dvol)),
        "boundary_max_abs": cube_boundary_max_abs(density),
    }
    if multipoles is not None:
        coeff_charge = coefficient_charge(multipoles)
        coeff_dipole = coefficient_dipole(atoms, multipoles)
        metrics.update(
            {
                "coefficient_charge": coeff_charge,
                "coefficient_dipole": coeff_dipole,
                "charge_error": charge - coeff_charge,
                "dipole_error": dipole - coeff_dipole,
                "dipole_error_norm": float(np.linalg.norm(dipole - coeff_dipole)),
            }
        )
    return metrics


class PotentialInterpolator:
    def __init__(
        self,
        sigma=2.0,
        multipoles_max_l=1,
        kspace_cutoff_factor=None,
        kspace_cutoff=None,
        device="cpu",
        subtract_total_charge=False,
    ):
        if not GRAPH_LONGRANGE_AVAILABLE:
            raise ImportError(
                "graph_longrange is required for polar density cube export"
            ) from GRAPH_LONGRANGE_IMPORT_ERROR

        warnings.warn(
            "PotentialInterpolator supports fully periodic cells and slab cells "
            "with pbc = [True, True, False]. Use RealSpaceDensityInterpolator "
            "for other periodicities.",
            stacklevel=2,
        )
        cutoff_given = kspace_cutoff is not None
        factor_given = kspace_cutoff_factor is not None
        if cutoff_given == factor_given:
            raise ValueError(
                "Provide exactly one of kspace_cutoff or kspace_cutoff_factor"
            )
        if factor_given:
            kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
                [sigma],
                multipoles_max_l,
            )

        self.kspace_cutoff = kspace_cutoff
        self.max_l = multipoles_max_l
        self.sigma = sigma
        self.device = device
        self.dtype = torch.float64
        self.subtract_total_charge = subtract_total_charge

        with use_dtype(self.dtype):
            self.density_basis = GTOBasis(
                max_l=multipoles_max_l,
                sigmas=[sigma],
                kspace_cutoff=self.kspace_cutoff,
                normalize="multipoles",
            ).to(self.device)

    @staticmethod
    def _total_charge_dipole(
        multipoles: torch.Tensor, node_positions: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_charge = torch.zeros(1, dtype=multipoles.dtype, device=multipoles.device)
        total_charge.index_add_(0, batch, multipoles[:, 0])
        dipole_q = torch.zeros((1, 3), dtype=multipoles.dtype, device=multipoles.device)
        dipole_q.index_add_(0, batch, node_positions * multipoles[:, 0].unsqueeze(-1))
        if multipoles.shape[-1] > 1:
            dipole_p = torch.zeros_like(dipole_q)
            dipole_p.index_add_(0, batch, multipoles[:, 1:4])
            dipole_q = dipole_q + dipole_p[:, [2, 0, 1]]
        return total_charge, dipole_q

    def _prepare_inputs(self, atoms, atomic_multipoles, coords):
        output_shape = coords.shape[:-1]
        sample_points = torch.tensor(
            coords, dtype=self.dtype, device=self.device
        ).reshape(-1, 3)
        node_positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device
        ).reshape(-1, 3)
        cell = torch.tensor(
            np.asarray(atoms.get_cell().array),
            dtype=self.dtype,
            device=self.device,
        ).reshape(1, 3, 3)
        pbc = torch.tensor(
            atoms.get_pbc(), dtype=torch.bool, device=self.device
        ).reshape(1, 3)
        batch = torch.zeros(
            node_positions.shape[0], dtype=torch.long, device=self.device
        )
        volume = torch.det(cell)
        multipoles = torch.tensor(
            atomic_multipoles, dtype=self.dtype, device=self.device
        ).reshape(node_positions.size(0), -1)
        return (
            output_shape,
            sample_points,
            node_positions,
            cell,
            pbc,
            batch,
            volume,
            multipoles,
        )

    def _compute_kspace(self, cell):
        rcell = 2 * np.pi * torch.linalg.inv(cell).transpose(-1, -2)
        return compute_k_vectors_flat(
            cutoff=self.kspace_cutoff, cell_vectors=cell, r_cell_vectors=rcell
        )

    def __call__(
        self,
        atoms,
        atomic_multipoles,
        external_field,
        fermi_level: float,
        coords,
    ):
        if coords.shape[-1] != 3:
            raise ValueError("coords must have shape (..., 3)")
        external_field = np.asarray(external_field, dtype=float)
        if external_field.shape != (3,):
            raise ValueError("external_field must have shape (3,)")

        (
            output_shape,
            sample_points,
            node_positions,
            cell,
            pbc,
            batch,
            volume,
            multipoles,
        ) = self._prepare_inputs(atoms, atomic_multipoles, coords)

        full_pbc = torch.tensor([[True, True, True]], device=pbc.device)
        z_slab_pbc = torch.tensor([[True, True, False]], device=pbc.device)
        if not (torch.all(pbc == full_pbc) or torch.all(pbc == z_slab_pbc)):
            raise ValueError(
                "PotentialInterpolator only supports pbc = [True, True, True] or "
                "pbc = [True, True, False]."
            )

        if self.subtract_total_charge:
            total_charge, _ = self._total_charge_dipole(
                multipoles, node_positions, batch
            )
            multipoles[:, 0] -= total_charge[0] / multipoles.shape[0]

        k_vectors, k_norm2, k_vector_batch, k0_mask = self._compute_kspace(cell)

        inner_products = torch.matmul(k_vectors, node_positions.t())
        k_mask = k_vector_batch[:, None] == batch[None, :]
        k_mask_f = k_mask.to(dtype=inner_products.dtype)
        cosines = torch.cos(inner_products) * k_mask_f
        sines = torch.sin(inner_products) * k_mask_f

        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        volume_per_k = volume.reshape(-1)[k_vector_batch]
        density = assemble_fourier_series_batch(
            source_feats=multipoles,
            cosines=cosines,
            sines=sines,
            density_basis_fs=density_basis_fs,
            volume_per_k=volume_per_k,
        )
        potential = apply_coulomb_kernel_batch(
            k_norm2=k_norm2,
            density=density,
        )

        _, total_dipole = self._total_charge_dipole(multipoles, node_positions, batch)
        correction_field_z = (
            FIELD_CONSTANT
            * total_dipole[:, 2]
            / volume
            * (~pbc[:, 2]).to(total_dipole.dtype)
        )

        sample_batch = torch.zeros(
            sample_points.shape[0], dtype=torch.long, device=self.device
        )
        samples_density = evaluate_fourier_series_at_points_flat(
            k_vectors=k_vectors,
            k_vector_batch=k_vector_batch,
            fourier_coefficients=density,
            sample_points=sample_points,
            sample_batch=sample_batch,
            k0_mask=k0_mask,
        )
        samples_potential = (
            evaluate_fourier_series_at_points_flat(
                k_vectors=k_vectors,
                k_vector_batch=k_vector_batch,
                fourier_coefficients=potential,
                sample_points=sample_points,
                sample_batch=sample_batch,
                k0_mask=k0_mask,
            )
            + fermi_level
        )
        samples_potential_corrected = (
            samples_potential
            + correction_field_z[sample_batch] * sample_points[:, 2]
            + external_field[2] * sample_points[:, 2]
        )

        density_np = samples_density.detach().cpu().numpy().reshape(output_shape)
        potential_np = samples_potential.detach().cpu().numpy().reshape(output_shape)
        corrected_np = (
            samples_potential_corrected.detach().cpu().numpy().reshape(output_shape)
        )
        return density_np, potential_np, corrected_np


class RealSpaceDensityInterpolator:
    def __init__(
        self,
        sigma=2.0,
        multipoles_max_l=1,
        device="cpu",
        cutoff_factor=6.0,
        chunk_size=65536,
    ):
        if multipoles_max_l > 1:
            raise NotImplementedError(
                "Real-space cube export currently supports monopoles and dipoles "
                "(multipoles_max_l <= 1)."
            )
        self.sigma = float(sigma)
        self.max_l = int(multipoles_max_l)
        self.device = device
        self.dtype = torch.float64
        self.cutoff = float(cutoff_factor) * self.sigma
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def _image_shifts(self, atoms) -> torch.Tensor:
        cell = np.asarray(atoms.cell.array, dtype=float)
        pbc = np.asarray(atoms.pbc, dtype=bool)
        ranges = []
        for axis in range(3):
            if not pbc[axis]:
                ranges.append(range(0, 1))
                continue
            length = np.linalg.norm(cell[axis])
            nmax = int(np.ceil(self.cutoff / length)) if length > 0.0 else 0
            ranges.append(range(-nmax, nmax + 1))

        shifts = []
        for i in ranges[0]:
            for j in ranges[1]:
                for k in ranges[2]:
                    shifts.append(i * cell[0] + j * cell[1] + k * cell[2])
        return torch.tensor(np.asarray(shifts), dtype=self.dtype, device=self.device)

    @staticmethod
    def _dipoles_to_cartesian(multipoles: torch.Tensor) -> torch.Tensor:
        dipoles = torch.zeros(
            (multipoles.shape[0], 3),
            dtype=multipoles.dtype,
            device=multipoles.device,
        )
        if multipoles.shape[1] > 1:
            dipoles = multipoles[:, 1:4][:, [2, 0, 1]]
        return dipoles

    def __call__(self, atoms, atomic_multipoles, coords):
        if coords.shape[-1] != 3:
            raise ValueError("coords must have shape (..., 3)")
        output_shape = coords.shape[:-1]
        sample_points = torch.tensor(
            coords, dtype=self.dtype, device=self.device
        ).reshape(-1, 3)
        node_positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device
        ).reshape(-1, 3)
        multipoles = torch.tensor(
            atomic_multipoles, dtype=self.dtype, device=self.device
        ).reshape(node_positions.shape[0], -1)
        if multipoles.shape[1] > 4:
            raise NotImplementedError(
                "Real-space cube export currently supports up to 4 multipole "
                "coefficients per atom (l <= 1)."
            )

        charges = multipoles[:, 0]
        dipoles = self._dipoles_to_cartesian(multipoles)
        shifts = self._image_shifts(atoms)
        sigma2 = self.sigma**2
        norm = 1.0 / ((2.0 * np.pi) ** 1.5 * self.sigma**3)
        cutoff2 = self.cutoff**2

        density = torch.empty(
            sample_points.shape[0], dtype=self.dtype, device=self.device
        )
        image_positions = (node_positions[None, :, :] + shifts[:, None, :]).reshape(
            -1, 3
        )
        image_charges = charges.repeat(shifts.shape[0])
        image_dipoles = dipoles.repeat(shifts.shape[0], 1)

        for start in range(0, sample_points.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, sample_points.shape[0])
            diff = sample_points[start:stop, None, :] - image_positions[None, :, :]
            r2 = torch.sum(diff * diff, dim=-1)
            gaussian = norm * torch.exp(-0.5 * r2 / sigma2)
            gaussian = torch.where(
                r2 <= cutoff2,
                gaussian,
                torch.zeros_like(gaussian),
            )
            values = gaussian @ image_charges
            if multipoles.shape[1] > 1:
                dipole_term = torch.sum(diff * image_dipoles[None, :, :], dim=-1)
                values = values + torch.sum(gaussian * dipole_term / sigma2, dim=-1)
            density[start:stop] = values

        return density.detach().cpu().numpy().reshape(output_shape), None, None


def _model_density_settings(
    calc, sigma: Optional[float], kspace_cutoff: Optional[float]
):
    model = calc.models[0]
    coulomb_energy = model.coulomb_energy
    sigma = (
        float(sigma)
        if sigma is not None
        else float(coulomb_energy.density_smearing_width)
    )
    kspace_cutoff = (
        float(kspace_cutoff)
        if kspace_cutoff is not None
        else float(coulomb_energy.kspace_cutoff)
    )
    return sigma, int(coulomb_energy.density_max_l), kspace_cutoff


def _parse_vector(values: Optional[Sequence[float]], default) -> np.ndarray:
    if values is None:
        return np.asarray(default, dtype=float)
    if len(values) != 3:
        raise ValueError("Expected three values")
    return np.asarray(values, dtype=float)


def _select_backend(atoms, backend: str) -> str:
    if backend != "auto":
        return backend
    pbc = tuple(bool(x) for x in atoms.pbc)
    if pbc in ((True, True, True), (True, True, False)):
        return "fourier"
    return "realspace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export MACE-Polar charge/spin density coefficients to cube files.",
    )
    parser.add_argument("--configs", required=True, help="input XYZ/EXTXYZ file")
    parser.add_argument(
        "--model",
        default="polar-1-m",
        help="Polar model name, local path, or URL accepted by mace_polar",
    )
    parser.add_argument("--output", required=True, help="output cube path or prefix")
    parser.add_argument("--index", type=int, default=0, help="configuration index")
    parser.add_argument(
        "--quantity",
        choices=["charge", "spin", "alpha", "beta", "all"],
        default="spin",
        help="density quantity to write",
    )
    parser.add_argument(
        "--grid",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        default=(80, 80, 160),
        help="cube grid dimensions",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--default_dtype", default="float32", choices=["float32", "float64"]
    )
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--kspace_cutoff", type=float, default=None)
    parser.add_argument(
        "--backend",
        choices=["auto", "fourier", "realspace"],
        default="auto",
        help="interpolation backend; auto uses realspace for unsupported Fourier PBC",
    )
    parser.add_argument(
        "--realspace_cutoff_factor",
        type=float,
        default=6.0,
        help="real-space image cutoff in units of sigma",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=65536,
        help="number of grid points per real-space evaluation chunk",
    )
    parser.add_argument(
        "--subtract_total_charge",
        action="store_true",
        help="subtract uniform per-atom monopole charge before interpolation",
    )
    parser.add_argument(
        "--external_field",
        nargs=3,
        type=float,
        default=None,
        help="external field for potential correction; defaults to atoms.info or zero",
    )
    parser.add_argument(
        "--fermi_level",
        type=float,
        default=None,
        help="fermi level for potential output; defaults to atoms.info or zero",
    )
    parser.add_argument(
        "--write_potential",
        action="store_true",
        help="also write potential and corrected-potential cube files",
    )
    parser.add_argument(
        "--quality_report",
        default=None,
        help="optional JSON path for cube quality metrics",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> list[Path]:
    atoms = ase.io.read(args.configs, index=args.index)
    calc = mace_polar(
        model=args.model,
        device=args.device,
        default_dtype=args.default_dtype,
    )
    atoms.calc = calc
    atoms.get_potential_energy()

    sigma, multipoles_max_l, kspace_cutoff = _model_density_settings(
        calc, args.sigma, args.kspace_cutoff
    )
    backend = _select_backend(atoms, args.backend)
    if backend == "fourier":
        interpolator = PotentialInterpolator(
            sigma=sigma,
            multipoles_max_l=multipoles_max_l,
            kspace_cutoff=kspace_cutoff,
            device=args.device,
            subtract_total_charge=args.subtract_total_charge,
        )
    else:
        if args.write_potential:
            raise NotImplementedError(
                "--write_potential is currently available only with "
                "--backend fourier."
            )
        interpolator = RealSpaceDensityInterpolator(
            sigma=sigma,
            multipoles_max_l=multipoles_max_l,
            device=args.device,
            cutoff_factor=args.realspace_cutoff_factor,
            chunk_size=args.chunk_size,
        )
    coords = make_grid(atoms, args.grid)
    external_field = _parse_vector(
        args.external_field, atoms.info.get("external_field", np.zeros(3))
    )
    fermi_level = (
        float(args.fermi_level)
        if args.fermi_level is not None
        else float(atoms.info.get("fermi_level", 0.0))
    )

    quantities = (
        ["charge", "spin", "alpha", "beta"]
        if args.quantity == "all"
        else [args.quantity]
    )
    output = Path(args.output)
    written = []
    quality_reports = {}
    for quantity in quantities:
        multipoles = select_multipoles(calc.results, quantity)
        if backend == "fourier":
            density, potential, corrected = interpolator(
                atoms, multipoles, external_field, fermi_level, coords
            )
        else:
            density, potential, corrected = interpolator(atoms, multipoles, coords)
        if args.quantity == "all":
            density_path = output.with_name(f"{output.stem}_{quantity}.cube")
        else:
            density_path = output
        write_cube_file(
            density_path,
            atoms,
            density,
            comment=f"MACE-Polar {quantity} density",
        )
        written.append(density_path)
        metrics = cube_quality_metrics(atoms, density, coords, multipoles)
        quality_reports[quantity] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in metrics.items()
        }

        if args.write_potential:
            potential_path = density_path.with_name(
                f"{density_path.stem}_potential.cube"
            )
            corrected_path = density_path.with_name(
                f"{density_path.stem}_potential_corrected.cube"
            )
            write_cube_file(
                potential_path,
                atoms,
                potential,
                comment=f"MACE-Polar {quantity} potential",
            )
            write_cube_file(
                corrected_path,
                atoms,
                corrected,
                comment=f"MACE-Polar {quantity} corrected potential",
            )
            written.extend([potential_path, corrected_path])
    if args.quality_report is not None:
        report_path = Path(args.quality_report)
        with open(report_path, "w", encoding="utf-8") as fout:
            json.dump(
                {
                    "backend": backend,
                    "grid": list(args.grid),
                    "quantities": quality_reports,
                },
                fout,
                indent=2,
            )
        written.append(report_path)
    return written


def main() -> None:
    written = run(parse_args())
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
