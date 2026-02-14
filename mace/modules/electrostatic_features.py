###########################################################################################
# Long Range Features
# Authors: Will Baldwin
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


from typing import Callable, Optional, Tuple, Union, List
import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from graph_longrange.utils import to_dense_batch
from graph_longrange.kspace import FourierReconstructionBlock

from graph_longrange.gto_electrostatics import (
    GTOFourierSeriesCoeficientsBlock,
    GTOChargeDensityFourierSeriesBlock,
    GTOLocalOrbitalProjectionBlock,
    KspaceCoulombOperatorBlock,
    GTOSelfInteractionBlock,
    MonopoleDipoleFieldBlock,
    DisplacedGTOExternalFieldBlock,
    CorrectivePotentialBlock,
    GTOInternalFieldtoFeaturesBlock,
    RealSpaceFiniteDifferenceElectrostaticFeatures,
    GTOInternalFieldtoFeaturesBlock,
)
from graph_longrange.slabs import (
    slab_dipole_correction_node_fields,
    get_nonperiodic_charge_dipole,
    slab_dipole_correction_total_field,
)
from .wrapper_ops import CuEquivarianceConfig, TransposeIrrepsLayoutWrapper


@compile_mode("script")
class PBCAgnosticElectrostaticFeatureBlock(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        projection_max_l: int,
        projection_smearing_widths: List[float],
        kspace_cutoff: float,
        field_irreps: o3.Irreps,
        include_self_interaction=False,
        integral_normalization="receiver",
        quadrupole_feature_corrections=False,
        **kwargs,
    ):
        super().__init__()

        self.pbc_features = PureElectrostaticsLRFeatureBlock(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            projection_max_l=projection_max_l,
            projection_smearing_widths=projection_smearing_widths,
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=include_self_interaction,
            integral_normalization=integral_normalization,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
        )
        if density_max_l > 0 or projection_max_l > 0 or include_self_interaction:
            import warnings

            warnings.warn(
                "realspace features currently doesn't support max_l > 0 or self interaction features"
            )

        self.realspace_features = RealSpaceFiniteDifferenceElectrostaticFeatures(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            projection_max_l=projection_max_l,
            projection_smearing_widths=projection_smearing_widths,
            integral_normalization=integral_normalization,
        )
        self.cueq_config = kwargs.get("cueq_config", None)
        layout_target = (
            self.cueq_config.layout_str
            if (self.cueq_config is not None and hasattr(self.cueq_config, "layout_str"))
            else "mul_ir"
        )
        self.transpose_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=field_irreps,
            source="mul_ir",
            target=layout_target,
            cueq_config=self.cueq_config,
        )

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_vectors_normed_squared: torch.Tensor,
        k_vectors_mask: torch.Tensor,
        source_feats: torch.Tensor,  # [n_nodes, 1, (max_l_s+1)**2]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
        pbc: torch.Tensor,
        use_pbc_evaluator: bool,
        return_electrostatic_potentials: bool = False,
    ) -> torch.Tensor:
        if pbc[0, 0] or use_pbc_evaluator:
            field_feats, _, esps = self.pbc_features(
                k_vectors,
                k_vectors_normed_squared,
                k_vectors_mask,
                source_feats,
                node_positions,
                batch,
                volumes,
                pbc,
                return_electrostatic_potentials,
            )
            if self.transpose_mul_ir is not None:
                field_feats = self.transpose_mul_ir(field_feats)
            return field_feats, esps
        else:
            field_feats, _, esps = self.realspace_features(
                source_feats, node_positions, batch
            )
            if self.transpose_mul_ir is not None:
                field_feats = self.transpose_mul_ir(field_feats)
            return field_feats, esps


@compile_mode("script")
class PureElectrostaticsLRFeatureBlock(torch.nn.Module):
    """this takes a set of source weights for GTO basis of charge density, and computes the field due to the density.
    Returns the projections of this field onto a second local basis of GTOs.
    Also optionally returns the realspace electrostatic potential."""

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        projection_max_l: int,
        projection_smearing_widths: List[float],
        kspace_cutoff: float,
        include_self_interaction=False,
        integral_normalization="receiver",
        quadrupole_feature_corrections=False,
    ):
        super().__init__()
        self.include_self_interaction = include_self_interaction

        # density
        self.density_gto_fs_block = GTOFourierSeriesCoeficientsBlock(
            sigmas=[density_smearing_width],
            max_l=density_max_l,
            kspace_cutoff=kspace_cutoff,
            normalize="multipoles",
        )
        self.density_block = GTOChargeDensityFourierSeriesBlock()

        # convolve
        self.field_conv_operator = KspaceCoulombOperatorBlock()

        # project
        self.project_gto_fs_block = GTOFourierSeriesCoeficientsBlock(
            sigmas=projection_smearing_widths,
            max_l=projection_max_l,
            kspace_cutoff=kspace_cutoff,
            normalize=integral_normalization,
        )
        self.projector = GTOLocalOrbitalProjectionBlock()

        num_radial_channels = len(projection_smearing_widths)
        indices = []
        for l in range(projection_max_l + 1):
            for c in range(num_radial_channels):
                offset = c * (projection_max_l + 1) ** 2
                indices += range(l**2 + offset, (l + 1) ** 2 + offset)

        self.register_buffer("indices", torch.tensor(indices, dtype=torch.int64))

        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            projection_max_l,
            projection_smearing_widths,
            "multipoles",
            integral_normalization,
        )

        self.non_periodic_correction_terms = NPCCorrectsFeatureBlock(
            density_max_l,
            projection_max_l,
            projection_smearing_widths,
            integral_normalization,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
        )

        self.realspace_evaluator = FourierReconstructionBlock()

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_vectors_normed_squared: torch.Tensor,
        k_vectors_mask: torch.Tensor,
        source_feats: torch.Tensor,  # [n_nodes, 1, (max_l_s+1)**2]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
        pbc: torch.Tensor,
        return_electrostatic_potentials: bool = False,
    ) -> torch.Tensor:

        basis_fs = self.density_gto_fs_block(
            k_vectors, k_vectors_normed_squared, k_vectors_mask
        )  # [n_graph, max_n_k, 1, (max_l_s+1)**2, 2]
        # print(f"basis_fs.shape : {basis_fs.shape}")

        density = self.density_block(
            source_feats, node_positions, k_vectors, basis_fs, volumes, batch
        )  # [n_graph, max_n_k, 2]
        # print(f"density.shape : {density.shape}")

        potential = self.field_conv_operator(
            density, k_vectors_normed_squared, k_vectors_mask
        )  # [n_graph, max_n_k, 2]
        # print(f"potential.shape : {potential.shape}")

        basis_fs = self.project_gto_fs_block(
            k_vectors, k_vectors_normed_squared, k_vectors_mask
        )  # [n_graph, max_n_k, n_receive_radial, (max_l_r+1)**2, 2]
        # print(f"basis_fs_2.shape : {basis_fs.shape}")

        projections = self.projector(
            k_vectors, node_positions, potential, batch, k_vectors_mask, basis_fs
        )  # [n_nodes, n_sigma, (max_l+1)**2]
        # print(f"projections.shape : {projections.shape}")

        reshaped = projections.flatten(start_dim=-2)[:, self.indices]
        self_interaction_terms = self.self_interaction(source_feats.squeeze(-2))
        if not self.include_self_interaction:
            reshaped -= self_interaction_terms

        # need to do pbc checks
        is_pbc = torch.index_select(torch.all(pbc, dim=1), -1, batch)
        correction_terms = self.non_periodic_correction_terms(
            source_feats=source_feats,
            node_positions=node_positions,
            batch=batch,
            volumes=volumes,
            pbc=pbc,
        )
        reshaped += correction_terms

        if return_electrostatic_potentials:
            total_charge, total_dipole = get_nonperiodic_charge_dipole(
                source_feats.squeeze(-2), node_positions, batch
            )

            correction_field = slab_dipole_correction_total_field(
                total_dipole,
                volumes,
            )

            # evaluate the real space potential at atomic sites
            eval_nodes, nodes_mask = to_dense_batch(node_positions, batch)
            esps = self.realspace_evaluator(
                k_vectors, potential, eval_nodes, nodes_mask.unsqueeze(-2)
            )
            esps_corrected = (
                esps + correction_field[:, 2].unsqueeze(-1) * eval_nodes[..., 2]
            )[nodes_mask, ...]

            return reshaped, self_interaction_terms, esps_corrected

        return reshaped, self_interaction_terms, None


@compile_mode("script")
class NPCCorrectsFeatureBlock(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        projection_max_l: int,
        projection_smearing_widths: List[float],
        integral_normalization="receiver",
        quadrupole_feature_corrections=False,
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

    def forward(
        self,
        source_feats: torch.Tensor,  # [n_nodes, 1, (max_l_s+1)**2]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        node_fields_molecule = self.self_field(
            charge_coefficients=source_feats.squeeze(-2),
            positions=node_positions,
            volumes=volumes,
            batch=batch,
        )  # [V, Ex, Ey, Ez]
        node_fields_slab = slab_dipole_correction_node_fields(
            source_feats=source_feats.squeeze(-2),
            node_positions=node_positions,
            volumes=volumes,
            batch=batch,
        )
        slab = torch.tensor([0, 0, 1], device=pbc.device)
        is_molecule = torch.all(torch.logical_not(pbc), dim=1)
        is_slab = torch.all(torch.logical_xor(slab, pbc), dim=1)
        is_molecule = torch.index_select(is_molecule, 0, batch)
        is_slab = torch.index_select(is_slab, 0, batch)

        node_fields = torch.zeros_like(node_fields_molecule)
        node_fields[is_molecule] = node_fields_molecule[is_molecule]
        node_fields[is_slab] = node_fields_slab[is_slab]

        projections = self.displaced_interactions(
            batch=batch, positions=node_positions, node_fields=node_fields
        )
        return projections
