"""A charge-aware model's density, sampled on a grid and written as a cube.

The model's density is a sum of Gaussian multipoles, one per atom. This turns
the coefficients it reports into values on a grid over the cell, by one of two
interpolators, and measures how faithful the grid is: whether it integrates
back to the charge and the dipole the coefficients carry, and whether the
density is still nonzero on the faces of the box.

**Two interpolators, for two kinds of cell.** :class:`FourierDensity` sums the
Gaussians in reciprocal space, which is exact for a periodic cell and gives the
electrostatic potential alongside the density; it takes a crystal or a slab
periodic along x and y. :class:`RealSpaceDensity` sums them directly, over
the periodic images within a cutoff, for any periodicity; it gives the density
only, and monopoles and dipoles only. :func:`select_backend` picks the first
where it applies and the second otherwise.

**Conventions.** Positions and the grid are in Angstrom, the density in
elementary charges per cubic Angstrom and the potential in volts. The
coefficients are in the model's component order, a charge and then a dipole as
``(y, z, x)``, per atom.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from ase import Atoms
from ase.io.cube import write_cube

from mace_torch.electrostatics.reference.features import (
    apply_coulomb_kernel_batch,
    assemble_fourier_series_batch,
    compute_coulomb_factor,
)
from mace_torch.electrostatics.reference.gto_utils import (
    GTOBasis,
    gto_basis_kspace_cutoff,
)
from mace_torch.electrostatics.reference.kspace import (
    compute_k_vectors_flat,
    evaluate_fourier_series_at_points_flat,
)
from mace_torch.electrostatics.reference.utils import FIELD_CONSTANT

__all__ = [
    "QUANTITIES",
    "FourierDensity",
    "Quantity",
    "RealSpaceDensity",
    "coefficient_charge",
    "coefficient_dipole",
    "cube_boundary_max_abs",
    "cube_quality_metrics",
    "make_grid",
    "select_backend",
    "select_multipoles",
    "voxel_volume",
    "write_cube_file",
]

#: Which density is written: the total, one spin channel, or their difference.
Quantity = Literal["charge", "spin", "alpha", "beta"]

QUANTITIES: tuple[str, ...] = ("charge", "spin", "alpha", "beta")

#: The periodicities reciprocal space takes: a crystal and a z-normal slab.
_FOURIER_PERIODICITIES = ((True, True, True), (True, True, False))


def make_grid(atoms: Atoms, grid: Sequence[int]) -> np.ndarray:
    """``[nx, ny, nz, 3]`` Cartesian points spanning the cell, origin included
    and the far faces excluded, as a cube file lays them out."""
    if len(grid) != 3:
        raise ValueError(f"grid is {list(grid)}; it is three sizes, nx ny nz.")
    sizes = [int(size) for size in grid]
    if any(size <= 0 for size in sizes):
        raise ValueError(f"grid is {sizes}; every size has to be positive.")
    fractional = np.stack(
        np.meshgrid(
            *(np.linspace(0.0, 1.0, size, endpoint=False) for size in sizes),
            indexing="ij",
        ),
        axis=-1,
    )
    return fractional @ atoms.cell.array


def select_multipoles(results: Mapping[str, Any], quantity: str) -> np.ndarray:
    """``[n_atoms, components]``: the coefficients of one density.

    Args:
        results: A calculator's results, with ``density_coefficients`` and
            ``spin_charge_density``.
        quantity: One of :data:`QUANTITIES`. ``spin`` is alpha minus beta.
    """
    if quantity == "charge":
        return np.asarray(results["density_coefficients"])
    channels = np.asarray(results["spin_charge_density"])
    if quantity == "alpha":
        return channels[:, 0, :]
    if quantity == "beta":
        return channels[:, 1, :]
    if quantity == "spin":
        return channels[:, 0, :] - channels[:, 1, :]
    raise ValueError(
        f"{quantity!r} is not a density this writes. They are {list(QUANTITIES)}."
    )


def select_backend(atoms: Atoms, backend: str) -> str:
    """``fourier`` for a crystal or a z-normal slab under ``auto``, else
    ``realspace``; an explicit choice is kept."""
    if backend != "auto":
        return backend
    periodicity = tuple(bool(flag) for flag in atoms.pbc)
    return "fourier" if periodicity in _FOURIER_PERIODICITIES else "realspace"


def write_cube_file(path: str | Path, atoms: Atoms, data: np.ndarray, comment: str):
    """Write ``data`` on the cell as a Gaussian cube file."""
    with open(path, "w", encoding="utf-8") as handle:
        write_cube(handle, atoms, data=np.asarray(data), comment=comment)


def voxel_volume(atoms: Atoms, density: np.ndarray) -> float:
    """The volume each grid point stands for, in cubic Angstrom."""
    return float(abs(np.linalg.det(atoms.cell.array)) / np.prod(density.shape))


def coefficient_charge(multipoles: np.ndarray) -> float:
    """The total charge the coefficients carry."""
    return float(np.sum(np.asarray(multipoles)[:, 0]))


def coefficient_dipole(atoms: Atoms, multipoles: np.ndarray) -> np.ndarray:
    """The total dipole the coefficients carry, ``[3]`` in e Angstrom:
    each charge at its atom, plus each atom's own dipole."""
    multipoles = np.asarray(multipoles)
    dipoles = np.zeros((multipoles.shape[0], 3))
    if multipoles.shape[1] > 1:
        dipoles = multipoles[:, 1:4][:, [2, 0, 1]]
    return np.sum(atoms.positions * multipoles[:, :1], axis=0) + np.sum(dipoles, axis=0)


def cube_boundary_max_abs(density: np.ndarray) -> float:
    """The largest density on any face of the grid. Far from zero, the box
    cuts the density off and the cube does not hold all of it."""
    faces = [
        density[0],
        density[-1],
        density[:, 0],
        density[:, -1],
        density[:, :, 0],
        density[:, :, -1],
    ]
    return float(max(np.max(np.abs(face)) for face in faces))


def cube_quality_metrics(
    atoms: Atoms,
    density: np.ndarray,
    coords: np.ndarray,
    multipoles: np.ndarray | None = None,
) -> dict[str, Any]:
    """How faithful a sampled density is.

    The integrated charge and dipole, the range and norm of the density, and
    its largest value on the faces; with the coefficients, what they carry and
    how far the grid is from it.
    """
    density = np.asarray(density)
    coords = np.asarray(coords)
    volume = voxel_volume(atoms, density)
    charge = float(np.sum(density) * volume)
    dipole = np.sum(density[..., None] * coords, axis=(0, 1, 2)) * volume
    metrics: dict[str, Any] = {
        "voxel_volume": volume,
        "integrated_charge": charge,
        "integrated_dipole": dipole,
        "density_min": float(np.min(density)),
        "density_max": float(np.max(density)),
        "density_l2": float(np.sqrt(np.sum(density * density) * volume)),
        "boundary_max_abs": cube_boundary_max_abs(density),
    }
    if multipoles is not None:
        expected_charge = coefficient_charge(multipoles)
        expected_dipole = coefficient_dipole(atoms, multipoles)
        metrics.update(
            {
                "coefficient_charge": expected_charge,
                "coefficient_dipole": expected_dipole,
                "charge_error": charge - expected_charge,
                "dipole_error": dipole - expected_dipole,
                "dipole_error_norm": float(np.linalg.norm(dipole - expected_dipole)),
            }
        )
    return metrics


def _total_charge_and_dipole(
    multipoles: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    charge = multipoles[:, 0].sum()
    dipole = (positions * multipoles[:, :1]).sum(dim=0)
    if multipoles.shape[1] > 1:
        dipole = dipole + multipoles[:, 1:4].sum(dim=0)[[2, 0, 1]]
    return charge, dipole


class FourierDensity:
    """The density and its potential, summed in reciprocal space.

    For a crystal, and for a slab periodic along x and y, whose potential takes
    the slab's dipole correction along z.

    Args:
        sigma: The Gaussian width of each atom's density, in Angstrom.
        multipoles_max_l: The highest multipole order of the coefficients.
        kspace_cutoff_factor: The reciprocal-space cutoff as a multiple of the
            one ``sigma`` calls for. Give this or ``kspace_cutoff``.
        kspace_cutoff: The reciprocal-space cutoff, in inverse Angstrom.
        device: Where to compute.
        subtract_total_charge: Spread the net charge evenly over the atoms and
            remove it, so a charged cell has a finite potential.
    """

    def __init__(
        self,
        sigma: float = 2.0,
        multipoles_max_l: int = 1,
        kspace_cutoff_factor: float | None = None,
        kspace_cutoff: float | None = None,
        device: str = "cpu",
        subtract_total_charge: bool = False,
    ) -> None:
        if (kspace_cutoff is None) == (kspace_cutoff_factor is None):
            raise ValueError(
                "give exactly one of kspace_cutoff and kspace_cutoff_factor."
            )
        if kspace_cutoff is None:
            assert kspace_cutoff_factor is not None
            kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
                [sigma], multipoles_max_l
            )
        self.kspace_cutoff = float(kspace_cutoff)
        self.max_l = multipoles_max_l
        self.sigma = sigma
        self.device = device
        self.dtype = torch.float64
        self.subtract_total_charge = subtract_total_charge
        previous = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            self.density_basis = GTOBasis(
                max_l=multipoles_max_l,
                sigmas=[sigma],
                kspace_cutoff=self.kspace_cutoff,
                normalize="multipoles",
            ).to(device)
        finally:
            torch.set_default_dtype(previous)

    def __call__(
        self,
        atoms: Atoms,
        atomic_multipoles: np.ndarray,
        external_field: Sequence[float] | np.ndarray,
        fermi_level: float,
        coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The density, the potential, and the potential with the slab and
        applied-field terms along z, each shaped like ``coords[..., 0]``.

        Raises:
            ValueError: For a periodicity reciprocal space does not take, or
                an applied field that is not one vector.
        """
        if coords.shape[-1] != 3:
            raise ValueError(f"coords has shape {coords.shape}; it ends in 3.")
        field = np.asarray(external_field, dtype=float)
        if field.shape != (3,):
            raise ValueError(f"external_field has shape {field.shape}; it is (3,).")
        periodicity = tuple(bool(flag) for flag in atoms.pbc)
        if periodicity not in _FOURIER_PERIODICITIES:
            raise ValueError(
                f"the structure is periodic along {periodicity}, and the "
                f"reciprocal-space density takes a crystal or a slab periodic "
                f"along x and y. Use the real-space one."
            )

        def tensor(value) -> torch.Tensor:
            return torch.as_tensor(
                np.asarray(value), dtype=self.dtype, device=self.device
            )

        shape = coords.shape[:-1]
        samples = tensor(coords).reshape(-1, 3)
        positions = tensor(atoms.get_positions()).reshape(-1, 3)
        cell = tensor(atoms.get_cell().array).reshape(1, 3, 3)
        volume = torch.det(cell)
        multipoles = tensor(atomic_multipoles).reshape(positions.shape[0], -1).clone()
        if self.subtract_total_charge:
            total, _ = _total_charge_and_dipole(multipoles, positions)
            multipoles[:, 0] -= total / multipoles.shape[0]

        reciprocal = 2 * np.pi * torch.linalg.inv(cell).transpose(-1, -2)
        k_vectors, k_norm2, k_batch, k0_mask = compute_k_vectors_flat(
            cutoff=self.kspace_cutoff, cell_vectors=cell, r_cell_vectors=reciprocal
        )
        phases = k_vectors @ positions.T
        density_k = assemble_fourier_series_batch(
            source_feats=multipoles,
            cosines=torch.cos(phases),
            sines=torch.sin(phases),
            density_basis_fs=self.density_basis(k_vectors, k_norm2, k0_mask),
            volume_per_k=volume.reshape(-1)[k_batch],
        )
        potential_k = apply_coulomb_kernel_batch(
            density=density_k,
            k_factor_coulomb=compute_coulomb_factor(k_norm2=k_norm2, k0_mask=k0_mask),
        )
        sample_batch = torch.zeros(
            samples.shape[0], dtype=torch.long, device=self.device
        )

        def sampled(coefficients: torch.Tensor) -> torch.Tensor:
            return evaluate_fourier_series_at_points_flat(
                k_vectors=k_vectors,
                k_vector_batch=k_batch,
                fourier_coefficients=coefficients,
                sample_points=samples,
                sample_batch=sample_batch,
                k0_mask=k0_mask,
            )

        density = sampled(density_k)
        potential = sampled(potential_k) + fermi_level
        _, dipole = _total_charge_and_dipole(multipoles, positions)
        slab_field = FIELD_CONSTANT * dipole[2] / volume[0] * float(not periodicity[2])
        corrected = potential + (slab_field + float(field[2])) * samples[:, 2]

        def array(value: torch.Tensor) -> np.ndarray:
            return value.detach().cpu().numpy().reshape(shape)

        return array(density), array(potential), array(corrected)


class RealSpaceDensity:
    """The density, summed directly over the periodic images near each point.

    Any periodicity, monopoles and dipoles only, and no potential.

    Args:
        sigma: The Gaussian width of each atom's density, in Angstrom.
        multipoles_max_l: The highest multipole order, 0 or 1.
        device: Where to compute.
        cutoff_factor: How far a Gaussian is summed, in widths.
        chunk_size: Grid points evaluated at once, which bounds the memory.
    """

    def __init__(
        self,
        sigma: float = 2.0,
        multipoles_max_l: int = 1,
        device: str = "cpu",
        cutoff_factor: float = 6.0,
        chunk_size: int = 65536,
    ) -> None:
        if multipoles_max_l > 1:
            raise NotImplementedError(
                f"multipoles_max_l is {multipoles_max_l}; the real-space density "
                f"is written for monopoles and dipoles only."
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size is {chunk_size}; it has to be positive.")
        self.sigma = float(sigma)
        self.max_l = int(multipoles_max_l)
        self.device = device
        self.dtype = torch.float64
        self.cutoff = float(cutoff_factor) * self.sigma
        self.chunk_size = int(chunk_size)

    def _image_shifts(self, atoms: Atoms) -> torch.Tensor:
        """Every lattice translation within the cutoff, along periodic axes."""
        cell = np.asarray(atoms.cell.array, dtype=float)
        ranges = []
        for axis, periodic in enumerate(np.asarray(atoms.pbc, dtype=bool)):
            length = float(np.linalg.norm(cell[axis]))
            reach = int(np.ceil(self.cutoff / length)) if periodic and length else 0
            ranges.append(range(-reach, reach + 1))
        shifts = [
            i * cell[0] + j * cell[1] + k * cell[2]
            for i in ranges[0]
            for j in ranges[1]
            for k in ranges[2]
        ]
        return torch.tensor(np.asarray(shifts), dtype=self.dtype, device=self.device)

    def __call__(
        self, atoms: Atoms, atomic_multipoles: np.ndarray, coords: np.ndarray
    ) -> tuple[np.ndarray, None, None]:
        """The density shaped like ``coords[..., 0]``, and no potential."""
        if coords.shape[-1] != 3:
            raise ValueError(f"coords has shape {coords.shape}; it ends in 3.")
        shape = coords.shape[:-1]
        samples = torch.as_tensor(coords, dtype=self.dtype, device=self.device)
        samples = samples.reshape(-1, 3)
        positions = torch.as_tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device
        ).reshape(-1, 3)
        multipoles = torch.as_tensor(
            np.asarray(atomic_multipoles), dtype=self.dtype, device=self.device
        ).reshape(positions.shape[0], -1)
        if multipoles.shape[1] > 4:
            raise NotImplementedError(
                f"the coefficients have {multipoles.shape[1]} components per atom; "
                f"the real-space density takes a charge and a dipole, four."
            )
        charges = multipoles[:, 0]
        dipoles = (
            multipoles[:, 1:4][:, [2, 0, 1]]
            if multipoles.shape[1] > 1
            else torch.zeros_like(positions)
        )
        shifts = self._image_shifts(atoms)
        images = (positions[None] + shifts[:, None]).reshape(-1, 3)
        image_charges = charges.repeat(shifts.shape[0])
        image_dipoles = dipoles.repeat(shifts.shape[0], 1)
        variance = self.sigma**2
        norm = 1.0 / ((2.0 * np.pi) ** 1.5 * self.sigma**3)
        reach = self.cutoff**2

        density = torch.empty(samples.shape[0], dtype=self.dtype, device=self.device)
        for start in range(0, samples.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, samples.shape[0])
            offset = samples[start:stop, None] - images[None]
            distance2 = (offset * offset).sum(dim=-1)
            gaussian = torch.where(
                distance2 <= reach,
                norm * torch.exp(-0.5 * distance2 / variance),
                torch.zeros_like(distance2),
            )
            values = gaussian @ image_charges
            if multipoles.shape[1] > 1:
                # The density of a Gaussian dipole is minus the dipole dotted
                # with the gradient of the Gaussian, which is this.
                along = (offset * image_dipoles[None]).sum(dim=-1)
                values = values + (gaussian * along / variance).sum(dim=-1)
            density[start:stop] = values
        return density.cpu().numpy().reshape(shape), None, None
