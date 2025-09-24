from typing import Dict, List, Optional

import torch
from e3nn.util.jit import compile_mode

from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock
from mace.modules.models import ScaleShiftMACE
from mace.modules.utils import get_atomic_virials_stresses, get_outputs, prepare_graph
from mace.modules.wrapper_ops import CuEquivarianceConfig, OEQConfig
from mace.tools.scatter import scatter_mean, scatter_sum


def _copy_mace_readout(
    mace_readout: torch.nn.Module, cueq_config: Optional[CuEquivarianceConfig] = None
) -> torch.nn.Module:
    """
    Helper function to copy a MACE readout block.
    """
    if isinstance(mace_readout, LinearReadoutBlock):
        return LinearReadoutBlock(
            irreps_in=mace_readout.linear.irreps_in,  # type:ignore
            irrep_out=mace_readout.linear.irreps_out,  # type:ignore
            cueq_config=cueq_config,
        )
    if isinstance(mace_readout, NonLinearReadoutBlock):  # type:ignore
        return NonLinearReadoutBlock(
            irreps_in=mace_readout.linear_1.irreps_in,  # type:ignore
            MLP_irreps=mace_readout.hidden_irreps,
            gate=mace_readout.non_linearity._modules["acts"][  # pylint: disable=W0212
                0
            ].f,
            irrep_out=mace_readout.linear_2.irreps_out,  # type:ignore
            num_heads=mace_readout.num_heads,
            cueq_config=cueq_config,
        )
    raise TypeError("Unsupported readout type.")


def _get_readout_input_dim(block: torch.nn.Module) -> int:
    if isinstance(block, LinearReadoutBlock):
        return block.linear.irreps_in.dim  # type:ignore
    if isinstance(block, NonLinearReadoutBlock):  # type:ignore
        return block.linear_1.irreps_in.dim  # type:ignore
    raise TypeError("Unsupported readout type for input dimension retrieval.")


@compile_mode("script")
class MACELES(ScaleShiftMACE):
    def __init__(self, les_arguments: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from les import Les
        except ImportError as exc:
            raise ImportError(
                "Cannot import 'les'. Please install the 'les' library from https://github.com/ChengUCB/les."
            ) from exc
        if les_arguments is None:
            les_arguments = {"use_atomwise": False}
        self.compute_bec = les_arguments.get("compute_bec", False)
        self.bec_output_index = les_arguments.get("bec_output_index", None)
        self.les = Les(les_arguments=les_arguments)
        self.les_readouts = torch.nn.ModuleList()
        self.readout_input_dims = [
            _get_readout_input_dim(readout) for readout in self.readouts  # type:ignore
        ]
        cueq_config = kwargs.get("cueq_config", None)
        for readout in self.readouts:  # type:ignore
            self.les_readouts.append(
                _copy_mace_readout(readout, cueq_config=cueq_config)
            )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        compute_bec: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Setting LES cell input to zero when boundary conditions are not periodic
        cell_les = cell.clone()
        pbc_tensor = data["pbc"].to(device=data["cell"].device)
        no_pbc_mask_cfg = ~pbc_tensor.any(dim=-1)
        no_pbc_mask_rows = no_pbc_mask_cfg.repeat_interleave(3)
        cell_les[no_pbc_mask_rows] = torch.zeros(
            (no_pbc_mask_rows.sum(), 3), dtype=cell_les.dtype, device=cell_les.device
        )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(
            vectors.dtype
        )  # [n_graphs, num_heads]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        # Embeddings of additional features
        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(
                data["batch"],
                embedding_features,
            )
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []
        node_qs_list: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_list.append(node_feats)

        for i, (readout, les_readout) in enumerate(
            zip(self.readouts, self.les_readouts)
        ):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es = readout(node_feats_list[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]
            node_qs = les_readout(node_feats_list[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]  # type:ignore
            node_qs_list.append(node_qs)
            node_es_list.append(node_es)

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        total_energy = e0 + inter_e
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()

        les_q = torch.sum(torch.stack(node_qs_list, dim=1), dim=1)
        les_result = self.les(
            latent_charges=les_q,
            positions=positions,
            cell=cell_les.view(-1, 3, 3),
            batch=data["batch"],
            compute_energy=True,
            compute_bec=(compute_bec or self.compute_bec),
            bec_output_index=self.bec_output_index,
        )
        les_energy_opt = les_result["E_lr"]
        if les_energy_opt is None:
            les_energy = torch.zeros_like(total_energy)
        else:
            les_energy = les_energy_opt
        total_energy += les_energy

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=inter_e + les_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
            "les_energy": les_energy,
            "latent_charges": les_q,
            "BEC": les_result["BEC"],
        }


# ------------------------------
# Field-aware Fukui MACE variant
# ------------------------------
from typing import Any, Callable, Dict, List, Optional, Type
import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
import logging

from mace.tools.scatter import scatter_sum
from mace.modules.utils import (
    get_outputs,
)

from mace.modules import (
    InteractionBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    NonLinearBiasReadoutBlock,
)

from graph_longrange.kspace import compute_k_vectors
from graph_longrange.gto_electrostatics import (
    gto_basis_kspace_cutoff,
    GTOExternalFieldBlock,
    DisplacedGTOExternalFieldBlock,
    KSpaceDirectElectrostaticEnergyBlock,
    PBCAgnosticDirectElectrostaticEnergyBlock,
)
from .electrostatic_features import (
    PBCAgnosticElectrostaticFeatureBlock,
)
from .field_blocks import (
    EnvironmentDependentSpinSourceBlock,
    field_update_blocks,
    field_readout_blocks,
    MultiLayerFeatureMixer,
)
from .utils import compute_total_charge_dipole_permuted


def _permute_to_e3nn_convention(x: torch.Tensor) -> torch.Tensor:
    return x[..., torch.LongTensor([1, 2, 0]).to(x.device)]


@compile_mode("script")
class FieldFukuiMACE(ScaleShiftMACE):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: torch.Tensor,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        kspace_cutoff_factor: float = 1.5,
        atomic_multipoles_max_l: int = 0,
        atomic_multipoles_smearing_width: float = 1.0,
        field_feature_max_l: int = 0,
        field_feature_widths: List[float] = (1.0,),
        num_recursion_steps: int = 1,
        field_si: bool = False,
        edge_irreps: Optional[o3.Irreps] = None,
        include_electrostatic_self_interaction: bool = False,
        add_local_electron_energy: bool = False,
        use_reduced_cg: bool = True,
        apply_cutoff: bool = True,
        field_dependence_type: str = "local_linear",
        final_field_readout_type: str = "OneBodyMLPFieldReadout",
        readout_cls: Optional[Type[NonLinearReadoutBlock]] = NonLinearReadoutBlock,
        quadrupole_feature_corrections: bool = False,
        return_electrostatic_potentials: bool = False,
        heads: Optional[List[str]] = None,
        field_feature_norms: Optional[List[float]] = None,
        field_norm_factor: Optional[float] = 0.02,
        fixedpoint_update_config: Optional[Dict[str, Any]] = None,
        field_readout_config: Optional[Dict[str, Any]] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
    ):
        # Initialize the MACE backbone first (interactions, products, readouts, embeddings)
        super().__init__(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            num_interactions=num_interactions,
            num_elements=num_elements,
            hidden_irreps=hidden_irreps,
            MLP_irreps=MLP_irreps,
            atomic_energies=atomic_energies,
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=atomic_numbers,
            correlation=correlation,
            gate=gate,
            radial_MLP=radial_MLP,
            radial_type=radial_type,
            heads=heads if heads is not None else ["Default"],
            atomic_inter_scale=[1.0] * (len(heads) if heads is not None else 1),
            atomic_inter_shift=[0.0] * (len(heads) if heads is not None else 1),
            edge_irreps=edge_irreps,
            use_reduced_cg=use_reduced_cg,
            use_agnostic_product=True,
            apply_cutoff=apply_cutoff,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            readout_cls=readout_cls,
            keep_last_layer_irreps=True,
        )
        self.num_recursion_steps = int(num_recursion_steps)
        self.atomic_multipoles_max_l = int(atomic_multipoles_max_l)

        # k-space cutoff heuristic
        kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
            [atomic_multipoles_smearing_width] + list(field_feature_widths),
            max(atomic_multipoles_max_l, field_feature_max_l),
        )
        self.register_buffer(
            "kspace_cutoff",
            torch.tensor(kspace_cutoff, dtype=torch.get_default_dtype()),
        )

        # Normalization for field features
        if field_feature_norms is not None:
            assert len(field_feature_norms) == len(field_feature_widths) * (
                field_feature_max_l + 1
            ), f"{len(field_feature_widths) * (field_feature_max_l+1)}, {len(field_feature_norms)}"
        else:
            field_feature_norms = (
                [1.0] * len(field_feature_widths) * (field_feature_max_l + 1)
            )
        expanded: List[float] = []
        for l in range(field_feature_max_l + 1):
            for j in range(len(field_feature_widths)):
                expanded += [field_feature_norms[l * len(field_feature_widths) + j]] * (
                    2 * l + 1
                )
        self.register_buffer(
            "field_feature_norms",
            torch.tensor(expanded, dtype=torch.get_default_dtype()),
        )

        self.lr_source_maps = torch.nn.ModuleList(
            EnvironmentDependentSpinSourceBlock(
                irreps_in=hidden_irreps, max_l=atomic_multipoles_max_l
            )
            for _ in range(num_recursion_steps + 1)
        )

        # Field-dependent components
        self.charges_irreps = 2 * o3.Irreps.spherical_harmonics(atomic_multipoles_max_l)
        lr_sh_irreps = o3.Irreps.spherical_harmonics(field_feature_max_l)
        self.potential_irreps = (lr_sh_irreps * len(field_feature_widths)).sort()[
            0
        ].simplify() * 2

        self.electric_potential_descriptor = PBCAgnosticElectrostaticFeatureBlock(
            density_max_l=atomic_multipoles_max_l,
            density_smearing_width=atomic_multipoles_smearing_width,
            projection_max_l=field_feature_max_l,
            projection_smearing_widths=field_feature_widths,
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=field_si,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
        )

        self.fukui_source_map = NonLinearBiasReadoutBlock(
            hidden_irreps,
            MLP_irreps.simplify(),
            gate,
            o3.Irreps("2x0e"),
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )

        if fixedpoint_update_config is None:
            fixedpoint_update_config = {}
        lr_source_cls = fixedpoint_update_config.pop("type")
        if isinstance(lr_source_cls, str):
            lr_source_cls = field_update_blocks[lr_source_cls]
        # Map optional class names to implementations
        pe_cls = fixedpoint_update_config.get("potential_embedding_cls", None)
        if isinstance(pe_cls, str):
            # currently only AgnosticChargeBiasedLinearPotentialEmbedding is required
            from .field_blocks import (
                AgnosticChargeBiasedLinearPotentialEmbedding as _PE,
            )

            fixedpoint_update_config["potential_embedding_cls"] = _PE
        nl_cls = fixedpoint_update_config.get("nonlinearity_cls", None)
        if isinstance(nl_cls, str):
            from .field_blocks import MLPNonLinearity as _NL

            fixedpoint_update_config["nonlinearity_cls"] = _NL
        # Reconstruct irreps needed for field update maps
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        max_ell_field_update = 2
        field_update_sh_irreps = o3.Irreps.spherical_harmonics(max_ell_field_update)
        self.from_ell_max_field_update = (max_ell_field_update + 1) ** 2
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        field_interaction_irreps = (
            (field_update_sh_irreps * num_features).sort()[0].simplify()
        )
        self.field_dependent_charges_maps = torch.nn.ModuleList()
        for _ in range(num_recursion_steps):
            self.field_dependent_charges_maps.append(
                lr_source_cls(
                    node_attrs_irreps=node_attr_irreps,
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=field_update_sh_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    target_irreps=field_interaction_irreps,
                    hidden_irreps=hidden_irreps,
                    avg_num_neighbors=avg_num_neighbors,
                    potential_irreps=self.potential_irreps,
                    charges_irreps=self.charges_irreps,
                    num_elements=num_elements,
                    field_norm_factor=float(field_norm_factor or 1.0),
                    **fixedpoint_update_config,
                )
            )

        # Post-SCF readout
        self.add_local_electron_energy = add_local_electron_energy
        if field_readout_config is None:
            field_readout_config = {}
        field_readout_cls = field_readout_config.pop("type")
        if isinstance(field_readout_cls, str):
            field_readout_cls = field_readout_blocks[field_readout_cls]
        self.local_electron_energy = field_readout_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=hidden_irreps,
            edge_attrs_irreps=field_update_sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=field_interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            potential_irreps=self.potential_irreps,
            charges_irreps=self.charges_irreps,
            **field_readout_config,
        )

        self.external_field_contribution = DisplacedGTOExternalFieldBlock(
            field_feature_max_l, list(field_feature_widths), "receiver"
        )
        self.coulomb_energy = PBCAgnosticDirectElectrostaticEnergyBlock(
            density_max_l=atomic_multipoles_max_l,
            density_smearing_width=atomic_multipoles_smearing_width,
            kspace_cutoff=float(kspace_cutoff),
            include_self_interaction=include_electrostatic_self_interaction,
        )
        self.return_electrostatic_potentials = return_electrostatic_potentials
        self.layer_feature_mixer = MultiLayerFeatureMixer(
            node_feats_irreps=hidden_irreps, num_interactions=num_interactions
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        use_pbc_evaluator: bool = False,
        fermi_level: Optional[torch.Tensor] = None,
        external_field: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )

        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        if fermi_level is None:
            fermi_level = data["fermi_level"]
        if external_field is None:
            external_field = data["external_field"]
        external_potential = torch.hstack(
            (torch.zeros_like(fermi_level).unsqueeze(-1), external_field)
        )

        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(vectors.dtype)

        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(_permute_to_e3nn_convention(vectors))
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        node_es_list: List[torch.Tensor] = []
        node_feats_list: List[torch.Tensor] = []
        spin_charge_density = torch.zeros(
            (data["batch"].size(-1), self.charges_irreps.dim),
            device=data["batch"].device,
            dtype=vectors.dtype,
        )

        for i, (interaction, product, lr_src) in enumerate(
            zip(self.interactions, self.products, self.lr_source_maps)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_list.append(node_feats)

            feat_idx = -1 if len(self.readouts) == 1 else min(i, len(self.readouts) - 1)
            node_es = self.readouts[feat_idx](node_feats, node_heads)[
                num_atoms_arange, node_heads
            ]
            node_es_list.append(node_es)

            spin_charge_sources = lr_src(node_feats).squeeze(-2)
            spin_charge_density = spin_charge_density + spin_charge_sources

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        # Build k-grid
        k_vectors, kv_norms_squared, kv_mask = compute_k_vectors(
            self.kspace_cutoff, cell.view(-1, 3, 3), data["rcell"].view(-1, 3, 3)
        )

        # SCF fixed point
        features_mixed = self.layer_feature_mixer(torch.stack(node_feats_list, dim=0))
        spin_charge_density = spin_charge_density.view(
            spin_charge_density.shape[0], 2, -1
        )
        fukui_sources = self.fukui_source_map(node_feats)
        fukui_norm = scatter_sum(
            src=fukui_sources.double(), index=data["batch"], dim=0, dim_size=num_graphs
        )[data["batch"]].to(vectors.dtype)
        fukui_sources = fukui_sources / fukui_norm
        Q_p_S = (data["total_charge"] + (data["total_spin"] - 1))[data["batch"]]
        Q_m_S = (data["total_charge"] - (data["total_spin"] - 1))[data["batch"]]
        pred_total_charges_0 = scatter_sum(
            src=spin_charge_density[:, :, 0].double(),
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )[data["batch"]].to(vectors.dtype)
        spin_charge_density = spin_charge_density.clone()
        spin_charge_density[:, 0, 0] = spin_charge_density[:, 0, 0] + fukui_sources[
            :, 0
        ] * ((Q_p_S / 2) - pred_total_charges_0[:, 0])
        spin_charge_density[:, 1, 0] = spin_charge_density[:, 1, 0] + fukui_sources[
            :, 1
        ] * ((Q_m_S / 2) - pred_total_charges_0[:, 1])

        potential_features = torch.zeros(
            (data["batch"].size(-1), self.potential_irreps.dim),
            device=data["batch"].device,
            dtype=vectors.dtype,
        )
        field_independent_spin_charge_density = spin_charge_density.clone()
        esps: Optional[torch.Tensor] = None

        for i in range(self.num_recursion_steps):
            field_feats_alpha, _, esps = self.electric_potential_descriptor(
                k_vectors=k_vectors,
                k_vectors_normed_squared=kv_norms_squared,
                k_vectors_mask=kv_mask,
                source_feats=spin_charge_density[:, 0, :].clone().unsqueeze(-2),
                node_positions=positions,
                batch=data["batch"],
                volumes=data["volume"],
                pbc=data["pbc"].view(-1, 3),
                use_pbc_evaluator=use_pbc_evaluator,
                return_electrostatic_potentials=self.return_electrostatic_potentials,
            )
            field_feats_beta, _, esps = self.electric_potential_descriptor(
                k_vectors=k_vectors,
                k_vectors_normed_squared=kv_norms_squared,
                k_vectors_mask=kv_mask,
                source_feats=spin_charge_density[:, 1, :].clone().unsqueeze(-2),
                node_positions=positions,
                batch=data["batch"],
                volumes=data["volume"],
                pbc=data["pbc"].view(-1, 3),
                use_pbc_evaluator=use_pbc_evaluator,
                return_electrostatic_potentials=self.return_electrostatic_potentials,
            )

            # Add external field contribution and subtract barycenter for gauge invariance
            barycenter = scatter_mean(
                src=positions.double(),
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            ).to(positions.dtype)
            half_external_field = 0.5 * self.external_field_contribution(
                data["batch"],
                positions - barycenter[data["batch"], :],
                external_potential,
            )
            field_feats_alpha = (
                field_feats_alpha + half_external_field
            ) / self.field_feature_norms
            field_feats_beta = (
                field_feats_beta + half_external_field
            ) / self.field_feature_norms

            potential_features = torch.cat(
                (field_feats_alpha, field_feats_beta), dim=-1
            )
            charge_sources_out = self.field_dependent_charges_maps[i](
                node_attrs=data["node_attrs"],
                node_feats=features_mixed,
                edge_attrs=edge_attrs[:, : self.from_ell_max_field_update],
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                potential_features=potential_features,
                local_charges=spin_charge_density.view(
                    spin_charge_density.shape[0], -1
                ),
            )

            current_fukui_sources = charge_sources_out[:, -2:]
            charge_sources = charge_sources_out[:, :-2]
            spin_charge_density_sources = charge_sources.view(
                spin_charge_density.shape[0], 2, -1
            )
            spin_charge_density = spin_charge_density + spin_charge_density_sources

            fukui_norm2 = scatter_sum(
                src=current_fukui_sources.double(),
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )[data["batch"]].to(vectors.dtype)
            current_fukui_sources = current_fukui_sources / fukui_norm2
            pred_total_charges = scatter_sum(
                src=spin_charge_density[:, :, 0].double(),
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )[data["batch"]].to(vectors.dtype)
            spin_charge_density = spin_charge_density.clone()
            spin_charge_density[:, 0, 0] = spin_charge_density[
                :, 0, 0
            ] + current_fukui_sources[:, 0] * ((Q_p_S / 2) - pred_total_charges[:, 0])
            spin_charge_density[:, 1, 0] = spin_charge_density[
                :, 1, 0
            ] + current_fukui_sources[:, 1] * ((Q_m_S / 2) - pred_total_charges[:, 1])

        total_energy = e0 + inter_e
        local_q_e = self.local_electron_energy(
            node_attrs=data["node_attrs"],
            node_feats=node_feats,
            edge_attrs=edge_attrs[:, : self.from_ell_max_field_update],
            edge_feats=edge_feats,
            edge_index=data["edge_index"],
            field_feats=potential_features,
            charges_0=field_independent_spin_charge_density.view(
                field_independent_spin_charge_density.shape[0], -1
            ),
            charges_induced=spin_charge_density.view(spin_charge_density.shape[0], -1),
        )
        le_total = scatter_sum(
            src=local_q_e, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        if getattr(self, "add_local_electron_energy", False):
            total_energy = total_energy + le_total
        else:
            le_total = torch.zeros_like(le_total)

        charge_density = spin_charge_density.sum(dim=1)
        spin_density = spin_charge_density[:, 0, :] - spin_charge_density[:, 1, :]
        total_charge, total_dipole = compute_total_charge_dipole_permuted(
            charge_density, positions, data["batch"], num_graphs
        )
        electro_energy = self.coulomb_energy(
            charge_density,
            positions,
            data["batch"],
            cell.view(-1, 3, 3),
            data["rcell"].view(-1, 3, 3),
            data["volume"],
            data["pbc"].view(-1, 3),
            num_graphs,
            use_pbc_evaluator=use_pbc_evaluator,
        )
        total_energy = (
            total_energy
            + electro_energy
            + torch.sum(external_potential[:, 1:] * total_dipole, dim=-1)
        )

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=total_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces or compute_atomic_stresses,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            from .utils import get_atomic_virials_stresses as _gav

            atomic_virials, atomic_stresses = _gav(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )

        return {
            "energy": total_energy,
            "node_energy": node_e0.clone().double() + node_inter_es.clone().double(),
            "interaction_energy": inter_e,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
            "density_coefficients": charge_density,
            "spin_density": spin_density,
            "charges_history": torch.stack(
                [spin_charge_density.clone().detach()], dim=-1
            ),
            "fermi_level": external_potential[:, 0],
            "external_field": external_potential[:, 1:],
            "charges": charge_density[:, 0],
            "spins": spin_density[:, 0],
            "dipole": total_dipole,
            "total_charge": total_charge,
            "electrostatic_energy": electro_energy,
            "electron_energy": le_total,
            "electrostatic_potentials": esps,
            "spin_charge_density": spin_charge_density,
        }
