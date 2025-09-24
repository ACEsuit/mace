import numpy as np
import torch
import warnings

from graph_longrange.kspace import FourierReconstructionBlock, compute_k_vectors
from graph_longrange.gto_electrostatics import (
    GTOFourierSeriesCoeficientsBlock,
    GTOChargeDensityFourierSeriesBlock,
    gto_basis_kspace_cutoff,
    KspaceCoulombOperatorBlock,
)
from graph_longrange.slabs import (
    get_nonperiodic_charge_dipole,
    slab_dipole_correction_total_field,
)

import mace.data
from mace.tools import torch_geometric, utils

from contextlib import contextmanager


@contextmanager
def use_dtype(dtype=torch.float64):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)


class PotentialInterpolator:
    def __init__(
        self,
        sigma=2.0,
        multipoles_max_l=1,
        kspace_cutoff_factor=None,
        kspace_cutoff=None,
        device="cpu",
        subtract_total_charge=False,
        dtype=None,
    ):
        warnings.warn(
            "currently only works for slabs with z-axis as the non-periodic axis."
        )
        cutoff_given = kspace_cutoff is not None
        factor_given = kspace_cutoff_factor is not None
        assert (
            cutoff_given != factor_given
        ), "Either provide a cutoff factor or a cutoff"
        if factor_given:
            kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
                [sigma], multipoles_max_l
            )
        self.kspace_cutoff = kspace_cutoff
        self.max_l = multipoles_max_l
        self.sigma = sigma
        self.device = device
        # Match dtype to torch default unless explicitly requested
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        self.subtract_total_charge = subtract_total_charge

        with use_dtype(self.dtype):
            self.density_gto_fs_block = GTOFourierSeriesCoeficientsBlock(
                sigmas=[sigma],
                max_l=multipoles_max_l,
                kspace_cutoff=self.kspace_cutoff,
                normalize="multipoles",
            ).to(self.device)
            self.density_block = GTOChargeDensityFourierSeriesBlock().to(self.device)
            self.coulomb = KspaceCoulombOperatorBlock().to(self.device)
            self.realspace_eval = FourierReconstructionBlock().to(self.device)

    def __call__(self, atoms, atomic_multipoles, external_field, fermi_level, coords):
        assert coords.shape[-1] == 3
        assert len(external_field.shape) == 1
        assert type(fermi_level) == float

        # reshape and convert to torch
        output_shape = coords.shape[:-1]
        xx = torch.tensor(coords, dtype=self.dtype, device=self.device)
        xx.reshape(-1, 3)

        keyspec = mace.data.KeySpecification()
        configs = [
            mace.data.config_from_atoms(
                atoms, key_specification=keyspec, head_name="none"
            )
        ]
        z_table = utils.AtomicNumberTable([int(z) for z in range(83)])
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                mace.data.AtomicData.from_config(
                    config,
                    z_table=z_table,
                    cutoff=5.0,
                    atomic_multipoles_max_l=self.max_l,
                )
                for config in configs
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        batchdict = next(iter(data_loader)).to(self.device)
        batchdict = batchdict.to_dict()
        node_positions = batchdict["positions"].to(self.dtype)
        cell = batchdict["cell"].reshape(-1, 3, 3).to(self.dtype)
        rcell = batchdict["rcell"].reshape(-1, 3, 3).to(self.dtype)
        volumes = batchdict["volume"].to(self.dtype)
        batch = batchdict["batch"]
        pbc = batchdict["pbc"].reshape(-1, 3)
        multipoles = torch.tensor(
            atomic_multipoles, dtype=self.dtype, device=self.device
        )

        if self.subtract_total_charge:
            total_charge, total_dipole = get_nonperiodic_charge_dipole(
                multipoles, node_positions, batch
            )
            multipoles[:, 0] -= total_charge / multipoles.shape[0]

        k_vectors, k_vectors_normed_squared, k_vectors_mask = compute_k_vectors(
            self.kspace_cutoff, cell, rcell
        )
        basis_fs = self.density_gto_fs_block(
            k_vectors, k_vectors_normed_squared, k_vectors_mask
        )
        density = self.density_block(
            multipoles.unsqueeze(-2),
            node_positions,
            k_vectors,
            basis_fs,
            volumes,
            batch,
        )
        potential = self.coulomb(density, k_vectors_normed_squared, k_vectors_mask)

        total_charge, total_dipole = get_nonperiodic_charge_dipole(
            multipoles, node_positions, batch
        )
        # raise NotImplementedError("Need to implement corrections bit properly")
        correction_field = slab_dipole_correction_total_field(
            total_dipole,
            volumes,
        )

        point_mask = torch.ones_like(xx[..., 0]).unsqueeze(0)
        samples_density = self.realspace_eval(
            k_vectors, density, xx.unsqueeze(0), point_mask
        )
        samples_potential = (
            self.realspace_eval(k_vectors, potential, xx.unsqueeze(0), point_mask)
            + fermi_level
        )
        samples_potential_corrected = (
            samples_potential
            + correction_field[0, 2] * xx[:, 2]
            + external_field[2] * xx[:, 2]
            + fermi_level
        )

        samples_density = samples_density.cpu().detach().numpy().reshape(output_shape)
        samples_potential = (
            samples_potential.cpu().detach().numpy().reshape(output_shape)
        )
        samples_potential_corrected = (
            samples_potential_corrected.cpu().detach().numpy().reshape(output_shape)
        )

        return samples_density, samples_potential, samples_potential_corrected


class PotentialInterpolatorPostProcessor:
    def __init__(self, calc, sigma, **kwargs):
        self.calc = calc
        if "kspace_cutoff" in kwargs:
            kspace_cutoff = kwargs.pop("kspace_cutoff")
        else:
            kspace_cutoff = calc.model.coulomb_energy.kspace_cutoff
        self.base_interpolator = PotentialInterpolator(
            sigma=sigma,
            multipoles_max_l=calc.model.coulomb_energy.max_l,
            kspace_cutoff=kspace_cutoff,
            **kwargs
        )

    def __call__(self, atoms, coords):
        atomic_multipoles = self.calc.results["density_coefficients"]
        external_field = self.calc.results["external_field"]
        fermi_level = self.calc.results["fermi_level"]
        return self.base_interpolator(
            atoms, atomic_multipoles, external_field, fermi_level, coords
        )
