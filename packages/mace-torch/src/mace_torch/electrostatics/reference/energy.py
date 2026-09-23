###########################################################################################
# Functions for computing electrostatic energy from from atomic multipoles
# and gaussian type orbitals
# Authors: Will Baldwin
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from collections.abc import Callable
from typing import Literal

import torch
from scipy.constants import pi

from .features import (
    apply_coulomb_kernel_batch,
    assemble_fourier_series_batch,
    compute_coulomb_factor,
)
from .gto_utils import GTOBasis, GTOSelfInteractionBlock
from .realspace_electrostatics import RealSpaceFiniteDiffereneEnergy
from .slabs import MonopoleDipoleCorrectionBlock, slab_dipole_correction_energy

PBCHandling = Literal[
    "realspace",
    "pbc",
    "slab",
    "molecule_in_box",
    "mixed_periodic",
    "auto",
]


def energy_product_batch(
    density: torch.Tensor,
    potential: torch.Tensor,
    volume: torch.Tensor,
    k_vector_batch: torch.Tensor,
) -> torch.Tensor:
    """Compute k-space electrostatic energy for flattened batch."""
    per_k = 2.0 * torch.sum(density * potential, dim=-1)
    num_graphs = int(volume.shape[0])
    if num_graphs == 1:
        energy_k = per_k.sum().view(1)
        return 0.5 * volume.reshape(-1) * energy_k / (2 * pi) ** 6
    energy_k = torch.zeros(num_graphs, dtype=per_k.dtype, device=per_k.device)
    energy_k.index_add_(0, k_vector_batch, per_k)
    return 0.5 * volume.reshape(-1) * energy_k / (2 * pi) ** 6


class GTOElectrostaticEnergy(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        kspace_cutoff: float,
        include_self_interaction: bool = False,
        pbc_handling: PBCHandling = "mixed_periodic",
    ):
        super().__init__()
        self.density_max_l = density_max_l
        self.density_smearing_width = density_smearing_width
        self.kspace_cutoff = kspace_cutoff
        self.include_self_interaction = include_self_interaction
        self.pbc_handling = pbc_handling

        self.density_basis = GTOBasis(
            max_l=density_max_l,
            sigmas=[density_smearing_width],
            kspace_cutoff=kspace_cutoff,
            normalize="multipoles",
        )
        self.self_interaction_terms = GTOSelfInteractionBlock(
            l_source=density_max_l,
            sigma_source=density_smearing_width,
            l_receive=density_max_l,
            sigmas_receive=[density_smearing_width],
            normalize_source="multipoles",
            normalize_receive="multipoles",
        )
        self.realspace_energy = RealSpaceFiniteDiffereneEnergy(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            include_self_interaction=include_self_interaction,
        )
        self.monopole_dipole_correction = MonopoleDipoleCorrectionBlock(density_max_l)
        self._forward_impl = self._select_forward_impl()

    def _select_forward_impl(self) -> Callable:
        if self.pbc_handling == "realspace":
            return self._forward_realspace
        if self.pbc_handling == "pbc":
            return self._forward_pbc
        if self.pbc_handling == "slab":
            return self._forward_slab
        if self.pbc_handling == "molecule_in_box":
            return self._forward_molecule_in_box
        if self.pbc_handling == "mixed_periodic":
            return self._forward_mixed_periodic
        if self.pbc_handling == "auto":
            return self._forward_auto
        raise ValueError(f"Unsupported pbc_handling: {self.pbc_handling}")

    def set_pbc_handling(self, pbc_handling: PBCHandling) -> None:
        self.pbc_handling = pbc_handling
        self._forward_impl = self._select_forward_impl()

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=source_feats,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )

    def _realspace_energy(
        self,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.realspace_energy(
            source_feats=source_feats,
            positions=node_positions,
            batch=batch,
        )

    def _subtract_self_interaction_if_needed(
        self,
        energy: torch.Tensor,
        source_feats: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if self.include_self_interaction:
            return energy
        self_fields = self.self_interaction_terms(source_feats)
        node_energies = torch.einsum("nb,nb->n", source_feats, self_fields)
        if num_graphs == 1:
            return energy - node_energies.sum().view(1) * 0.5
        self_energy = torch.zeros(
            num_graphs,
            dtype=node_energies.dtype,
            device=node_energies.device,
        )
        self_energy.index_add_(0, batch, node_energies)
        return energy - self_energy * 0.5

    def _compute_kspace_energy_common(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
    ) -> torch.Tensor:
        inner_products = torch.matmul(
            k_vectors, node_positions.t()
        )  # [n_k_total, n_nodes]
        mask = k_vector_batch[:, None] == batch[None, :]
        mask_f = mask.to(dtype=inner_products.dtype)
        cosines = torch.cos(inner_products) * mask_f
        sines = torch.sin(inner_products) * mask_f

        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        volume_per_k = volume.reshape(-1)[k_vector_batch]
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=cosines,
            sines=sines,
            density_basis_fs=density_basis_fs,
            volume_per_k=volume_per_k,
        )

        k_factor_coulomb = compute_coulomb_factor(k_norm2, k0_mask)
        potential = apply_coulomb_kernel_batch(
            density=density,
            k_factor_coulomb=k_factor_coulomb,
        )
        energy = energy_product_batch(
            density=density,
            potential=potential,
            volume=volume,
            k_vector_batch=k_vector_batch,
        )

        return self._subtract_self_interaction_if_needed(
            energy=energy,
            source_feats=source_feats,
            batch=batch,
            num_graphs=int(volume.shape[0]),
        )

    def _forward_realspace(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        return self._realspace_energy(
            source_feats=source_feats,
            node_positions=node_positions,
            batch=batch,
        )

    def _forward_pbc(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        return self._compute_kspace_energy_common(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=source_feats,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
        )

    def _forward_slab(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        energy = self._compute_kspace_energy_common(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=source_feats,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
        )
        slab_correction = slab_dipole_correction_energy(
            source_feats,
            node_positions,
            volume,
            batch,
        )
        return energy + slab_correction

    def _forward_molecule_in_box(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        energy = self._compute_kspace_energy_common(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=source_feats,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
        )
        molecule_correction = self.monopole_dipole_correction(
            source_feats,
            node_positions,
            volume,
            batch,
        )
        return energy + molecule_correction

    def _forward_mixed_periodic(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        energy = self._compute_kspace_energy_common(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=source_feats,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
        )
        molecule_correction = self.monopole_dipole_correction(
            source_feats,
            node_positions,
            volume,
            batch,
        )
        slab_correction = slab_dipole_correction_energy(
            source_feats,
            node_positions,
            volume,
            batch,
        )
        slab = torch.tensor([0, 0, 1], dtype=torch.bool, device=pbc.device)
        is_molecule = torch.all(torch.logical_not(pbc), dim=1)
        is_slab = torch.all(torch.logical_xor(slab, pbc), dim=1)

        correction_energy = torch.zeros_like(molecule_correction)
        correction_energy = torch.where(
            is_molecule, molecule_correction, correction_energy
        )
        correction_energy = torch.where(is_slab, slab_correction, correction_energy)
        return energy + correction_energy

    def _forward_auto(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        if torch.any(pbc):
            return self._forward_mixed_periodic(
                k_vectors=k_vectors,
                k_norm2=k_norm2,
                k_vector_batch=k_vector_batch,
                k0_mask=k0_mask,
                source_feats=source_feats,
                node_positions=node_positions,
                batch=batch,
                volume=volume,
                pbc=pbc,
            )
        return self._forward_realspace(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=source_feats,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )
