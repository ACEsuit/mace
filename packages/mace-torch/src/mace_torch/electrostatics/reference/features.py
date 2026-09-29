###########################################################################################
# Functions for computing global electrostatic features from atomic multipoles
# and gaussian type orbitals
# Authors: Will Baldwin
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from collections.abc import Callable
from typing import Literal

import torch
from scipy.constants import pi

from .gto_utils import (
    GTOBasis,
    GTOInternalFieldtoFeaturesBlock,
    GTOSelfInteractionBlock,
)
from .realspace_electrostatics import RealSpaceFiniteDifferenceElectrostaticFeatures
from .slabs import (
    CorrectivePotentialBlock,
    _get_total_dipole_z,
    slab_dipole_correction_node_fields,
)
from .utils import FIELD_CONSTANT

FeaturePBCHandling = Literal[
    "realspace",
    "pbc",
    "slab",
    "molecule_in_box",
    "mixed_periodic",
    "auto",
]


def assemble_fourier_series_batch(
    source_feats: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    density_basis_fs: torch.Tensor,
    volume_per_k: torch.Tensor,
) -> torch.Tensor:
    """Assemble rho(k) for flattened batch.

    Args:
        source_feats: [n_nodes, m_dim]
        cosines: [n_k_total, n_nodes]
        sines: [n_k_total, n_nodes]
        density_basis_fs: [n_k_total, n_sigma(=1), m_dim, 2]
        volume_per_k: [n_k_total]
    Returns:
        density: [n_k_total, 2]
    """
    n_sigma = density_basis_fs.size(1)
    m_dim = density_basis_fs.size(2)
    sm_dim = n_sigma * m_dim

    coeff_cos = torch.matmul(cosines, source_feats)
    coeff_sin = torch.matmul(sines, source_feats)

    density_basis_r = density_basis_fs[..., 0].reshape(density_basis_fs.size(0), sm_dim)
    density_basis_i = density_basis_fs[..., 1].reshape(density_basis_fs.size(0), sm_dim)

    rho_real = (density_basis_r * coeff_cos).sum(dim=-1) + (
        density_basis_i * coeff_sin
    ).sum(dim=-1)
    rho_imag = (density_basis_i * coeff_cos).sum(dim=-1) - (
        density_basis_r * coeff_sin
    ).sum(dim=-1)

    density = torch.stack([rho_real, rho_imag], dim=-1)
    density = (2 * pi) ** 3 * density / volume_per_k.unsqueeze(-1)
    return density


def compute_coulomb_factor(
    k_norm2: torch.Tensor,
    k0_mask: torch.Tensor,
) -> torch.Tensor:
    """Build the masked Coulomb factor 1/|k|^2 with the k=0 mode set to zero."""
    k0_mask_bool = k0_mask > 0.0
    safe_k_norm2 = torch.where(k0_mask_bool, torch.ones_like(k_norm2), k_norm2)
    return (~k0_mask_bool).to(k_norm2.dtype) / safe_k_norm2


def apply_coulomb_kernel_batch(
    density: torch.Tensor,
    k_factor_coulomb: torch.Tensor,
) -> torch.Tensor:
    """Apply Coulomb kernel in k-space for flattened batch."""
    factor = k_factor_coulomb.reshape(-1, *([1] * (density.dim() - 1)))
    potential = density * factor
    return potential * FIELD_CONSTANT


def project_to_features_batch(
    potential: torch.Tensor,
    feature_basis_fs: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
) -> torch.Tensor:
    """Project potential to local GTO features for flattened batch."""
    n_k = feature_basis_fs.size(0)
    n_sigma = feature_basis_fs.size(1)
    m_dim = feature_basis_fs.size(2)
    sm_dim = n_sigma * m_dim

    proj_basis_r = feature_basis_fs[..., 0].reshape(n_k, sm_dim)
    proj_basis_i = feature_basis_fs[..., 1].reshape(n_k, sm_dim)

    A = potential[:, 0].unsqueeze(-1) * proj_basis_r + (
        potential[:, 1].unsqueeze(-1) * proj_basis_i
    )
    B = potential[:, 0].unsqueeze(-1) * proj_basis_i - (
        potential[:, 1].unsqueeze(-1) * proj_basis_r
    )

    A = A * k_factor_proj.unsqueeze(-1)
    B = B * k_factor_proj.unsqueeze(-1)

    proj_cos = torch.matmul(A.t(), cosines)
    proj_sin = torch.matmul(B.t(), sines)
    proj_total = 2.0 * (proj_cos + proj_sin)
    projections = proj_total.t().reshape(cosines.size(1), n_sigma, m_dim)
    return projections / (2 * pi) ** 3


def reconstruct_esps_batch(
    potential: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k0_mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct electrostatic potential at node positions."""
    summand = (
        2.0 * potential[:, 0].unsqueeze(-1) * cosines
        - 2.0 * potential[:, 1].unsqueeze(-1) * sines
    )
    k0_factor = torch.ones_like(k0_mask)
    k0_factor[k0_mask > 0.0] = 0.5
    summand = summand * k0_factor.unsqueeze(-1)
    return torch.sum(summand, dim=0) / (2 * pi) ** 3


class NonPeriodicFeatureCorrections(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        projection_max_l: int,
        projection_smearing_widths: list[float],
        integral_normalization: str = "receiver",
        quadrupole_feature_corrections: bool = False,
    ):
        super().__init__()
        self.self_field = CorrectivePotentialBlock(
            density_max_l=density_max_l,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
        )
        self.displaced_interactions = GTOInternalFieldtoFeaturesBlock(
            l_receive=projection_max_l,
            sigmas_receive=projection_smearing_widths,
            normalize_receive=integral_normalization,
        )

    def slab(
        self,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
    ) -> torch.Tensor:
        node_fields = slab_dipole_correction_node_fields(
            source_feats=source_feats,
            node_positions=node_positions,
            volumes=volumes,
            batch=batch,
        )
        return self.displaced_interactions(
            batch=batch,
            positions=node_positions,
            node_fields=node_fields,
        )

    def molecule_in_box(
        self,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
    ) -> torch.Tensor:
        node_fields = self.self_field(
            charge_coefficients=source_feats,
            positions=node_positions,
            volumes=volumes,
            batch=batch,
        )
        return self.displaced_interactions(
            batch=batch,
            positions=node_positions,
            node_fields=node_fields,
        )

    def mixed_periodic(
        self,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
        correction_node_masks: dict,
    ) -> torch.Tensor:
        node_fields_molecule = self.self_field(
            charge_coefficients=source_feats,
            positions=node_positions,
            volumes=volumes,
            batch=batch,
        )
        node_fields_slab = slab_dipole_correction_node_fields(
            source_feats=source_feats,
            node_positions=node_positions,
            volumes=volumes,
            batch=batch,
        )
        is_molecule = correction_node_masks["is_molecule_node"]
        is_slab = correction_node_masks["is_slab_node"]
        node_fields = torch.zeros_like(node_fields_molecule)
        node_fields[is_molecule] = node_fields_molecule[is_molecule]
        node_fields[is_slab] = node_fields_slab[is_slab]

        return self.displaced_interactions(
            batch=batch,
            positions=node_positions,
            node_fields=node_fields,
        )

    def forward(
        self,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
        *,
        pbc_handling: Literal["slab", "molecule_in_box", "mixed_periodic"],
        correction_node_masks: dict | None = None,
        pbc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pbc_handling == "slab":
            return self.slab(
                source_feats=source_feats,
                node_positions=node_positions,
                batch=batch,
                volumes=volumes,
            )

        if pbc_handling == "molecule_in_box":
            return self.molecule_in_box(
                source_feats=source_feats,
                node_positions=node_positions,
                batch=batch,
                volumes=volumes,
            )

        if correction_node_masks is None:
            if pbc is None:
                raise ValueError("mixed_periodic corrections require masks or pbc.")
            pbc_bool = pbc.to(dtype=torch.bool)
            is_molecule_graph = (~pbc_bool).all(dim=1)
            is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])
            correction_node_masks = {
                "is_molecule_node": torch.index_select(is_molecule_graph, 0, batch),
                "is_slab_node": torch.index_select(is_slab_graph, 0, batch),
            }

        return self.mixed_periodic(
            source_feats=source_feats,
            node_positions=node_positions,
            batch=batch,
            volumes=volumes,
            correction_node_masks=correction_node_masks,
        )


class GTOElectrostaticFeatures(torch.nn.Module):
    output_permutation: torch.Tensor

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        feature_max_l: int,
        feature_smearing_widths: list[float],
        include_self_interaction: bool,
        kspace_cutoff: float,
        quadrupole_feature_corrections: bool = False,
        integral_normalization: str = "receiver",
        pbc_handling: FeaturePBCHandling = "mixed_periodic",
    ):
        super().__init__()
        self.density_basis = GTOBasis(
            max_l=density_max_l,
            sigmas=[density_smearing_width],
            kspace_cutoff=kspace_cutoff,
            normalize="multipoles",
        )
        self.feature_basis = GTOBasis(
            max_l=feature_max_l,
            sigmas=feature_smearing_widths,
            kspace_cutoff=kspace_cutoff,
            normalize=integral_normalization,
        )
        self.kspace_cutoff = kspace_cutoff
        self.include_self_interaction = include_self_interaction
        self.pbc_handling = pbc_handling

        self.self_interaction_terms = GTOSelfInteractionBlock(
            l_source=density_max_l,
            sigma_source=density_smearing_width,
            l_receive=feature_max_l,
            sigmas_receive=feature_smearing_widths,
            normalize_source="multipoles",
            normalize_receive=integral_normalization,
        )
        self.realspace_features = RealSpaceFiniteDifferenceElectrostaticFeatures(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            projection_max_l=feature_max_l,
            projection_smearing_widths=feature_smearing_widths,
            include_self_interaction=include_self_interaction,
            integral_normalization=integral_normalization,
        )
        self.non_periodic_correction_terms = NonPeriodicFeatureCorrections(
            density_max_l=density_max_l,
            projection_max_l=feature_max_l,
            projection_smearing_widths=feature_smearing_widths,
            integral_normalization=integral_normalization,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
        )
        self.register_buffer(
            "output_permutation",
            self._build_output_permutation(
                max_l=feature_max_l,
                n_radial=len(feature_smearing_widths),
            ),
        )

        self._precompute_geometry_impl = self._select_precompute_geometry_impl()
        self._forward_dynamic_impl = self._select_forward_dynamic_impl()

    @staticmethod
    def _build_output_permutation(max_l: int, n_radial: int) -> torch.Tensor:
        indices = []
        block = (max_l + 1) ** 2
        for ell in range(max_l + 1):
            for c in range(n_radial):
                offset = c * block
                indices += range(ell**2 + offset, (ell + 1) ** 2 + offset)
        return torch.tensor(indices, dtype=torch.long)

    def _permute_output_channels(self, features_flat: torch.Tensor) -> torch.Tensor:
        return torch.index_select(features_flat, dim=-1, index=self.output_permutation)

    def _select_precompute_geometry_impl(self) -> Callable:
        if self.pbc_handling == "realspace":
            return self._precompute_geometry_realspace
        if self.pbc_handling == "pbc":
            return self._precompute_geometry_pbc
        if self.pbc_handling == "slab":
            return self._precompute_geometry_slab
        if self.pbc_handling == "molecule_in_box":
            return self._precompute_geometry_molecule_in_box
        if self.pbc_handling == "mixed_periodic":
            return self._precompute_geometry_mixed_periodic
        if self.pbc_handling == "auto":
            return self._precompute_geometry_auto
        raise ValueError(f"Unsupported pbc_handling: {self.pbc_handling}")

    def _select_forward_dynamic_impl(self) -> Callable:
        if self.pbc_handling == "realspace":
            return self._forward_dynamic_realspace
        if self.pbc_handling == "pbc":
            return self._forward_dynamic_pbc
        if self.pbc_handling == "slab":
            return self._forward_dynamic_slab
        if self.pbc_handling == "molecule_in_box":
            return self._forward_dynamic_molecule_in_box
        if self.pbc_handling == "mixed_periodic":
            return self._forward_dynamic_mixed_periodic
        if self.pbc_handling == "auto":
            return self._forward_dynamic_auto
        raise ValueError(f"Unsupported pbc_handling: {self.pbc_handling}")

    def set_pbc_handling(self, pbc_handling: FeaturePBCHandling) -> None:
        self.pbc_handling = pbc_handling
        self._precompute_geometry_impl = self._select_precompute_geometry_impl()
        self._forward_dynamic_impl = self._select_forward_dynamic_impl()

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
        cache = self._precompute_geometry_impl(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )
        return self._forward_dynamic_impl(
            source_feats=source_feats,
            cache=cache,
        )

    def precompute_geometry(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        cache = self._precompute_geometry_impl(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )
        return cache

    def forward_dynamic(
        self,
        cache: dict,
        source_feats: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_dynamic_impl(
            source_feats=source_feats,
            cache=cache,
        )

    def _precompute_geometry_realspace(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        return {
            "mode": "realspace",
            "node_positions": node_positions,
            "batch": batch,
        }

    def _precompute_geometry_periodic_common(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        inner_products = torch.matmul(
            k_vectors, node_positions.t()
        )  # [n_k_total, n_nodes]
        mask = k_vector_batch[:, None] == batch[None, :]
        mask_f = mask.to(dtype=inner_products.dtype)
        cosines = torch.cos(inner_products) * mask_f
        sines = torch.sin(inner_products) * mask_f

        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        feature_basis_fs = self.feature_basis(k_vectors, k_norm2, k0_mask)

        volume_per_k = volume.reshape(-1)[k_vector_batch]
        k0_mask_bool = k0_mask > 0.0
        k_factor_coulomb = compute_coulomb_factor(k_norm2, k0_mask)
        k_factor_proj = torch.ones_like(k_norm2)
        k_factor_proj[k0_mask_bool] = 0.5

        return {
            "mode": "pbc",
            "k_norm2": k_norm2,
            "k_vector_batch": k_vector_batch,
            "k0_mask": k0_mask,
            "volume_per_k": volume_per_k,
            "k_factor_coulomb": k_factor_coulomb,
            "k_factor_proj": k_factor_proj,
            "volumes": volume.reshape(-1),
            "batch": batch,
            "node_positions": node_positions,
            "pbc": pbc,
            "cosines": cosines,
            "sines": sines,
            "density_basis_fs": density_basis_fs,
            "feature_basis_fs": feature_basis_fs,
        }

    def _precompute_geometry_pbc(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        return self._precompute_geometry_periodic_common(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )

    def _precompute_geometry_slab(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        return self._precompute_geometry_periodic_common(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )

    def _precompute_geometry_molecule_in_box(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        return self._precompute_geometry_periodic_common(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )

    def _precompute_geometry_mixed_periodic(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        cache = self._precompute_geometry_periodic_common(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )
        pbc_bool = pbc.to(dtype=torch.bool).reshape(-1, 3)
        pbc_bool = pbc_bool.expand(volume.reshape(-1).shape[0], -1)
        is_molecule_graph = (~pbc_bool).all(dim=1)
        is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])
        cache["correction_node_masks"] = {
            "is_molecule_node": torch.index_select(is_molecule_graph, 0, batch),
            "is_slab_node": torch.index_select(is_slab_graph, 0, batch),
        }
        return cache

    def _precompute_geometry_auto(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        if torch.any(pbc):
            return self._precompute_geometry_mixed_periodic(
                k_vectors=k_vectors,
                k_norm2=k_norm2,
                k_vector_batch=k_vector_batch,
                k0_mask=k0_mask,
                node_positions=node_positions,
                batch=batch,
                volume=volume,
                pbc=pbc,
            )
        return self._precompute_geometry_realspace(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )

    @staticmethod
    def _require_cache_mode(cache: dict, expected_mode: str) -> None:
        cache_mode = cache.get("mode")
        if cache_mode != expected_mode:
            raise ValueError(f"Expected a {expected_mode!r} cache, got {cache_mode!r}.")

    def compute_esps(
        self, cache: dict, source_feats: torch.Tensor, pbc: torch.Tensor
    ) -> torch.Tensor:
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=cache["cosines"],
            sines=cache["sines"],
            density_basis_fs=cache["density_basis_fs"],
            volume_per_k=cache["volume_per_k"],
        )
        potential = apply_coulomb_kernel_batch(
            density=density,
            k_factor_coulomb=cache["k_factor_coulomb"],
        )
        esps = reconstruct_esps_batch(
            potential=potential,
            cosines=cache["cosines"],
            sines=cache["sines"],
            k0_mask=cache["k0_mask"],
        )
        pbc_bool = cache["pbc"].to(dtype=torch.bool)
        is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])
        if is_slab_graph.any():
            total_dipole_z = _get_total_dipole_z(
                source_feats, cache["node_positions"], cache["batch"]
            )
            field_z = FIELD_CONSTANT * total_dipole_z / cache["volumes"]
            spread_field_z = torch.index_select(field_z, 0, cache["batch"])
            delta_esps = spread_field_z * cache["node_positions"][:, 2]
            slab_node_mask = torch.index_select(is_slab_graph, 0, cache["batch"])
            esps = esps + delta_esps * slab_node_mask.to(dtype=delta_esps.dtype)
        return esps

    def _forward_dynamic_periodic_common(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        self._require_cache_mode(cache, "pbc")
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=cache["cosines"],
            sines=cache["sines"],
            density_basis_fs=cache["density_basis_fs"],
            volume_per_k=cache["volume_per_k"],
        )
        potential = apply_coulomb_kernel_batch(
            density=density,
            k_factor_coulomb=cache["k_factor_coulomb"],
        )
        features_si = project_to_features_batch(
            potential=potential,
            feature_basis_fs=cache["feature_basis_fs"],
            cosines=cache["cosines"],
            sines=cache["sines"],
            k_factor_proj=cache["k_factor_proj"],
        )
        features_flat = features_si.reshape(features_si.size(0), -1)
        features_flat = self._permute_output_channels(features_flat)
        if not self.include_self_interaction:
            si_terms = self.self_interaction_terms(source_feats)
            features_flat = features_flat - si_terms
        return features_flat

    def _forward_dynamic_realspace(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        self._require_cache_mode(cache, "realspace")
        features, _, _ = self.realspace_features(
            source_feats=source_feats,
            node_positions=cache["node_positions"],
            batch=cache["batch"],
        )
        return features

    def _forward_dynamic_pbc(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        return self._forward_dynamic_periodic_common(
            source_feats=source_feats, cache=cache
        )

    def _forward_dynamic_slab(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        features_flat = self._forward_dynamic_periodic_common(
            source_feats=source_feats, cache=cache
        )
        correction_terms = self.non_periodic_correction_terms.slab(
            source_feats=source_feats,
            node_positions=cache["node_positions"],
            batch=cache["batch"],
            volumes=cache["volumes"],
        )
        return features_flat + correction_terms

    def _forward_dynamic_molecule_in_box(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        features_flat = self._forward_dynamic_periodic_common(
            source_feats=source_feats, cache=cache
        )
        correction_terms = self.non_periodic_correction_terms.molecule_in_box(
            source_feats=source_feats,
            node_positions=cache["node_positions"],
            batch=cache["batch"],
            volumes=cache["volumes"],
        )
        return features_flat + correction_terms

    def _forward_dynamic_mixed_periodic(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        features_flat = self._forward_dynamic_periodic_common(
            source_feats=source_feats, cache=cache
        )
        correction_node_masks = cache.get("correction_node_masks")
        if correction_node_masks is None:
            raise ValueError(
                "mixed_periodic features require a cache with correction_node_masks."
            )
        correction_terms = self.non_periodic_correction_terms.mixed_periodic(
            source_feats=source_feats,
            node_positions=cache["node_positions"],
            batch=cache["batch"],
            volumes=cache["volumes"],
            correction_node_masks=correction_node_masks,
        )
        return features_flat + correction_terms

    def _forward_dynamic_auto(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        cache_mode = cache.get("mode")
        if cache_mode == "realspace":
            return self._forward_dynamic_realspace(
                source_feats=source_feats,
                cache=cache,
            )
        if cache_mode == "pbc":
            return self._forward_dynamic_mixed_periodic(
                source_feats=source_feats,
                cache=cache,
            )
        raise ValueError(f"Unsupported cache mode for auto features: {cache_mode!r}.")
