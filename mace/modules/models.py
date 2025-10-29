
###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# Modified: MIL pooling integration via external module .mil_pooling (ConjunctivePooling)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.modules.embeddings import GenericJointEmbedding
from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_mean, scatter_sum
from mace.tools.torch_tools import get_change_of_basis, spherical_to_cartesian
# IMPORTANT: Use external MIL pooling implementation living in mace.modules.mil_pooling
from mace.modules.mil_pooling import ConjunctivePooling

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipolePolarReadoutBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipolePolarReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .utils import (
    compute_dielectric_gradients,
    compute_fixed_charge_dipole,
    compute_fixed_charge_dipole_polar,
    get_atomic_virials_stresses,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
    prepare_graph,
)


@compile_mode("script")
class MACE(torch.nn.Module):
    """MACE with optional MIL pooling residual on the final layer."""
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
            atomic_energies: np.ndarray,
            avg_num_neighbors: float,
            atomic_numbers: List[int],
            correlation: Union[int, List[int]],
            gate: Optional[Callable],
            pair_repulsion: bool = False,
            apply_cutoff: bool = True,
            use_reduced_cg: bool = True,
            use_so3: bool = False,
            use_agnostic_product: bool = False,
            use_last_readout_only: bool = False,
            use_embedding_readout: bool = False,
            distance_transform: str = "None",
            edge_irreps: Optional[o3.Irreps] = None,
            radial_MLP: Optional[List[int]] = None,
            radial_type: Optional[str] = "bessel",
            heads: Optional[List[str]] = None,
            cueq_config: Optional[Dict[str, Any]] = None,
            embedding_specs: Optional[Dict[str, Any]] = None,
            oeq_config: Optional[Dict[str, Any]] = None,
            lammps_mliap: Optional[bool] = False,
            readout_cls: Optional[Type[NonLinearReadoutBlock]] = NonLinearReadoutBlock,
            # MIL knobs
            use_mil_pooling: bool = True,
            mil_d_attn: int = 8,
            mil_dropout: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.register_buffer("num_interactions", torch.tensor(num_interactions, dtype=torch.int64))
        if heads is None:
            heads = ["Default"]
        self.heads = heads
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        self.lammps_mliap = lammps_mliap
        self.apply_cutoff = apply_cutoff
        self.edge_irreps = edge_irreps
        self.use_reduced_cg = use_reduced_cg
        self.use_agnostic_product = use_agnostic_product
        self.use_so3 = use_so3
        self.use_last_readout_only = use_last_readout_only
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        embedding_size = node_feats_irreps.count(o3.Irrep(0, 1))
        if embedding_specs is not None:
            self.embedding_specs = embedding_specs
            self.joint_embedding = GenericJointEmbedding(
                base_dim=embedding_size,
                embedding_specs=embedding_specs,
                out_dim=embedding_size,
            )
            if use_embedding_readout:
                self.embedding_readout = LinearReadoutBlock(
                    node_feats_irreps,
                    o3.Irreps(f"{len(heads)}x0e"),
                    cueq_config,
                    oeq_config,
                )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(p=num_polynomial_cutoff)
            self.pair_repulsion = True
        # spherical harmonics
        if not use_so3:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p=1)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        def generate_irreps(l):
            return o3.Irreps("+".join([f"1x{i}e+1x{i}o" for i in range(l + 1)]))
        sh_irreps_inter = sh_irreps if hidden_irreps.count(o3.Irrep(0, -1)) == 0 else generate_irreps(max_ell)
        interaction_irreps = (sh_irreps_inter * num_features).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization="component")
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )
        self.interactions = torch.nn.ModuleList([inter])
        use_sc_first = "Residual" in str(interaction_cls_first)
        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            use_reduced_cg=use_reduced_cg,
            use_agnostic_product=use_agnostic_product,
        )
        self.products = torch.nn.ModuleList([prod])
        self.readouts = torch.nn.ModuleList()
        if not use_last_readout_only:
            self.readouts.append(
                LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"), cueq_config, oeq_config)
            )

        final_node_irreps: Union[o3.Irreps, str] = hidden_irreps  # 初始为第一层 product 的输出
        for i in range(num_interactions - 1):
            hidden_irreps_out = str(hidden_irreps[0]) if i == num_interactions - 2 else hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                edge_irreps=edge_irreps,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product,
            )
            self.products.append(prod)
            final_node_irreps = hidden_irreps_out
            if i == num_interactions - 2:
                self.readouts.append(
                    readout_cls(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                        cueq_config,
                        oeq_config,
                    )
                )
            elif not use_last_readout_only:
                self.readouts.append(
                    LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"), cueq_config, oeq_config)
                )

        self.use_mil_pooling = use_mil_pooling
        self.mil_d_attn = mil_d_attn
        self.mil_dropout = mil_dropout
        if self.use_mil_pooling:
            if isinstance(final_node_irreps, str):
                self._final_node_irreps = o3.Irreps(final_node_irreps)
            else:
                self._final_node_irreps = final_node_irreps
            feat_dim = self._final_node_irreps.dim
            self.mil_pre_norm = torch.nn.LayerNorm(feat_dim, eps=1e-6)
            self.mil_readout = ConjunctivePooling(
                irreps_in=self._final_node_irreps.simplify(),
                out_dim=len(self.heads),
                d_attn=self.mil_d_attn,
                dropout=self.mil_dropout,
            )
            self.mil_gamma_raw = torch.nn.Parameter(torch.zeros(len(self.heads)))
            self.register_buffer("mil_gamma_cap", torch.tensor(0.3))

    @staticmethod
    def _channel_norm(x: torch.Tensor, eps: float = 1e-5, center: bool = True) -> torch.Tensor:
        # x: [N, C]
        if center:
            x = x - x.mean(dim=0, keepdim=True)
        var = x.pow(2).mean(dim=0, keepdim=True)
        return x * (var + eps).rsqrt()

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
    ) -> Dict[str, Optional[torch.Tensor]]:
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange.to(torch.int64)
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads.to(torch.int64)
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        node_e0 = self.atomic_energies_fn(data["node_attrs"])[num_atoms_arange, node_heads]
        e0 = scatter_sum(src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs).to(vectors.dtype)

        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers)
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers)
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
            pair_energy = scatter_sum(src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs)
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {name: data[name] for name, _ in self.embedding_specs.items()}
            node_feats += self.joint_embedding(data["batch"], embedding_features)
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = self.embedding_readout(node_feats, node_heads).squeeze(-1)
                embedding_energy = scatter_sum(src=embedding_node_energy, index=data["batch"], dim=0, dim_size=num_graphs)
                e0 += embedding_energy

        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_concat: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(zip(self.interactions, self.products)):
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
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice)
            node_feats_concat.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es = readout(node_feats_concat[feat_idx], node_heads)[num_atoms_arange, node_heads]
            energy = scatter_sum(node_es, data["batch"], dim=0, dim_size=num_graphs)
            energies.append(energy)
            node_energies_list.append(node_es)

        # === MIL pooling residual (graph-level) ===
        if getattr(self, "use_mil_pooling", False):
            last_feats = self.mil_pre_norm(node_feats_concat[-1])  # 你这路的变量是 node_feats_concat
            graph_logits: List[torch.Tensor] = []
            for g in range(num_graphs):
                mask = (data["batch"] == g)
                feats_g = last_feats[mask]
                if feats_g.numel() == 0:
                    graph_logits.append(
                        torch.zeros((len(self.heads),), device=last_feats.device, dtype=last_feats.dtype))
                else:
                    out_g = self.mil_readout(feats_g)  # [H]
                    out_g = out_g - out_g.mean()
                    graph_logits.append(out_g)
            mil_energy = torch.stack(graph_logits, dim=0)  # [G, H]
            mil_gamma = self.mil_gamma_cap * torch.tanh(self.mil_gamma_raw)  # [H]
            energies.append(mil_energy * mil_gamma)
            node_energies_list.append(torch.zeros_like(node_e0))

        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)
        node_energy = torch.sum(torch.stack(node_energies_list, dim=-1), dim=-1)
        node_feats_out = torch.cat(node_feats_concat, dim=-1)

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
            "contributions": contributions,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
        }


@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(self, atomic_inter_scale: float, atomic_inter_shift: float, **kwargs):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(scale=atomic_inter_scale, shift=atomic_inter_shift)
        self.use_mil_pooling = getattr(self, "use_mil_pooling", False)  # 已由父类消耗 kwargs 设置
        self.mil_d_attn = getattr(self, "mil_d_attn", 8)  # 已由父类消耗 kwargs 设置
        self.mil_dropout = getattr(self, "mil_dropout", 0.1)  # 已由父类消耗 kwargs 设置
        self.mil_pre_norm = getattr(self, "mil_pre_norm", True)
        self.mil_use_layernorm = getattr(self, "mil_use_layernorm", True)
        self.mil_eps = getattr(self, "mil_eps", 1e-6)
        self.mil_center = getattr(self, "mil_center", False)
        self.mil_gamma = getattr(self, "mil_gamma", 0.1)
        self.mil_gamma_cap = getattr(self, "mil_gamma_cap", None)
        self.mil_no_force_grad = getattr(self, "mil_no_force_grad", False)
        if self.use_mil_pooling:
            feat_dim = None
            try:
                final_irreps = self.products[-1].irreps_out
                if isinstance(final_irreps, str):
                    final_irreps = o3.Irreps(final_irreps)
                feat_dim = final_irreps.dim
            except Exception:
                feat_dim = kwargs.get("mil_feat_dim", None)
            if self.mil_use_layernorm and (feat_dim is not None) and (feat_dim > 0):
                self.mil_norm = torch.nn.LayerNorm(
                    normalized_shape=feat_dim,
                    eps=self.mil_eps,
                    elementwise_affine=True,
                )
            else:
                self.mil_norm = torch.nn.Identity()

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
    ) -> Dict[str, Optional[torch.Tensor]]:
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )

        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange.to(torch.int64)
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads.to(torch.int64)
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        node_e0 = self.atomic_energies_fn(data["node_attrs"])[num_atoms_arange, node_heads]
        e0 = scatter_sum(src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs).to(vectors.dtype)

        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers)

        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers)
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {name: data[name] for name, _ in self.embedding_specs.items()}
            node_feats += self.joint_embedding(data["batch"], embedding_features)
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = self.embedding_readout(node_feats, node_heads).squeeze(-1)
                embedding_energy = scatter_sum(src=embedding_node_energy, index=data["batch"], dim=0, dim_size=num_graphs)
                e0 += embedding_energy

        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(zip(self.interactions, self.products)):
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
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice)
            node_feats_list.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es_list.append(readout(node_feats_list[feat_idx], node_heads)[num_atoms_arange, node_heads])
        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)
        # === MIL graph-level residual (ScaleShiftMACE) ===
        if getattr(self, "use_mil_pooling", False):
            last_feats = node_feats_list[-1]  # [N, C]
            feats_for_mil = last_feats
            if getattr(self, "mil_pre_norm", False):
                if getattr(self, "mil_use_layernorm", False) and hasattr(self, "mil_norm"):
                    feats_for_mil = self.mil_norm(feats_for_mil)  # LayerNorm
                else:
                    eps = getattr(self, "mil_eps", 1e-6)
                    center = getattr(self, "mil_center", True)
                    feats_for_mil = self._channel_norm(feats_for_mil, eps=eps, center=center)
            graph_logits = []
            for g in range(num_graphs):
                mask = (data["batch"] == g)
                feats_g = feats_for_mil[mask]
                if feats_g.numel() == 0:
                    graph_logits.append(torch.zeros((len(self.heads),),device=feats_for_mil.device,dtype=feats_for_mil.dtype))
                else:
                    out_g = self.mil_readout(feats_g)  # [H]
                    if hasattr(self, "mil_gamma") and self.mil_gamma is not None:
                        gamma = self.mil_gamma
                        if hasattr(self, "mil_gamma_cap") and self.mil_gamma_cap is not None:
                            gamma = torch.clamp(torch.as_tensor(gamma, device=feats_for_mil.device,dtype=feats_for_mil.dtype),max=self.mil_gamma_cap).item()
                        out_g = out_g * gamma
                    if getattr(self, "mil_center", False):
                        out_g = out_g - out_g.mean(keepdim=True)
                    graph_logits.append(out_g)
            mil_e = torch.stack(graph_logits, dim=0)  # [G, H]
            if getattr(self, "mil_no_force_grad", False):
                mil_e_to_add = mil_e.detach()
            else:
                mil_e_to_add = mil_e
            inter_e = inter_e + mil_e_to_add
        total_energy = e0 + inter_e
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()

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
        }


@compile_mode("script")
class AtomicDipolesMACE(torch.nn.Module):
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
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[None],
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        distance_transform: str = "None",
        radial_type: Optional[str] = "bessel",
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        edge_irreps: Optional[o3.Irreps] = None,
    ):
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer("num_interactions", torch.tensor(num_interactions, dtype=torch.int64))
        assert atomic_energies is None

        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(irreps_in=node_attr_irreps, irreps_out=node_feats_irreps)
        self.radial_embedding = RadialEmbeddingBlock(r_max=r_max, num_bessel=num_bessel, num_polynomial_cutoff=num_polynomial_cutoff, radial_type=radial_type)
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization="component")
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        use_sc_first = "Residual" in str(interaction_cls_first)
        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert len(hidden_irreps) > 1, "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(hidden_irreps[1])
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(NonLinearDipoleReadoutBlock(hidden_irreps_out, MLP_irreps, gate, dipole_only=True))
            else:
                self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert compute_force is False and compute_virials is False and compute_stress is False and compute_displacement is False
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1

        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(positions=data["positions"], edge_index=data["edge_index"], shifts=data["shifts"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers)

        dipoles = []
        for interaction, product, readout in zip(self.interactions, self.products, self.readouts):
            node_feats, sc = interaction(node_attrs=data["node_attrs"], node_feats=node_feats, edge_attrs=edge_attrs, edge_feats=edge_feats, edge_index=data["edge_index"], cutoff=cutoff)
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"])
            node_dipoles = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            dipoles.append(node_dipoles)

        contributions_dipoles = torch.stack(dipoles, dim=-1)  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(src=atomic_dipoles, index=data["batch"], dim=0, dim_size=num_graphs)  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(charges=data["charges"], positions=data["positions"], batch=data["batch"], num_graphs=num_graphs)
        total_dipole = total_dipole + baseline
        return {"dipole": total_dipole, "atomic_dipoles": atomic_dipoles}


@compile_mode("script")
class AtomicDielectricMACE(torch.nn.Module):
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
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[None],
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        distance_transform: str = "None",
        radial_type: Optional[str] = "bessel",
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        edge_irreps: Optional[o3.Irreps] = None,
        dipole_only: Optional[bool] = True,
        use_polarizability: Optional[bool] = True,
        means_stds: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer("num_interactions", torch.tensor(num_interactions, dtype=torch.int64))

        self.register_buffer("dipole_mean", torch.zeros(3))
        self.register_buffer("dipole_std", torch.ones(3))
        self.register_buffer("polarizability_mean", torch.zeros(3, 3))
        self.use_polarizability = use_polarizability
        self.register_buffer("polarizability_std", torch.ones(3, 3))
        self.register_buffer("change_of_basis", get_change_of_basis())
        if means_stds is not None:
            if "dipole_mean" in means_stds: self.dipole_mean.data.copy_(means_stds["dipole_mean"])
            if "dipole_std" in means_stds: self.dipole_std.data.copy_(means_stds["dipole_std"])
            if "polarizability_mean" in means_stds: self.polarizability_mean.data.copy_(means_stds["polarizability_mean"])
            if "polarizability_std" in means_stds: self.polarizability_std.data.copy_(means_stds["polarizability_std"])
        assert atomic_energies is None

        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(irreps_in=node_attr_irreps, irreps_out=node_feats_irreps)
        self.radial_embedding = RadialEmbeddingBlock(r_max=r_max, num_bessel=num_bessel, num_polynomial_cutoff=num_polynomial_cutoff, radial_type=radial_type)
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization="component")
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        use_sc_first = "Residual" in str(interaction_cls_first)
        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipolePolarReadoutBlock(hidden_irreps, use_polarizability=True))

        for i in range(num_interactions - 1):
            hidden_irreps_out = hidden_irreps if i == num_interactions - 2 else hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(NonLinearDipolePolarReadoutBlock(hidden_irreps_out, MLP_irreps, gate, use_polarizability=True))
            else:
                self.readouts.append(LinearDipolePolarReadoutBlock(hidden_irreps, use_polarizability=True))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_dielectric_derivatives: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert not (compute_force or compute_virials or compute_stress or compute_displacement)
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms = data["ptr"][1:] - data["ptr"][:-1]

        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(positions=data["positions"], edge_index=data["edge_index"], shifts=data["shifts"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers)

        charges = []
        dipoles = []
        polarizabilities = []
        for interaction, product, readout in zip(self.interactions, self.products, self.readouts):
            node_feats, sc = interaction(node_attrs=data["node_attrs"], node_feats=node_feats, edge_attrs=edge_attrs, edge_feats=edge_feats, edge_index=data["edge_index"], cutoff=cutoff)
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"])
            node_out = readout(node_feats).squeeze(-1)
            charges.append(node_out[:, 0])
            node_dipoles = node_out[:, 2:5]
            node_polarizability = torch.cat((node_out[:, 1].unsqueeze(-1), node_out[:, 5:]), dim=-1)
            polarizabilities.append(node_polarizability)
            dipoles.append(node_dipoles)

        contributions_dipoles = torch.stack(dipoles, dim=-1)
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)
        atomic_charges = torch.stack(charges, dim=-1).sum(-1)
        total_charge_excess = scatter_mean(src=atomic_charges, index=data["batch"], dim_size=num_graphs) - (data["total_charge"] / num_atoms)
        atomic_charges = atomic_charges - total_charge_excess[data["batch"]]
        total_dipole = scatter_sum(src=atomic_dipoles, index=data["batch"], dim=0, dim_size=num_graphs)
        baseline = compute_fixed_charge_dipole_polar(charges=atomic_charges, positions=data["positions"], batch=data["batch"], num_graphs=num_graphs)
        total_dipole = total_dipole + baseline

        contributions_polarizabilities = torch.stack(polarizabilities, dim=-1)
        atomic_polarizabilities = torch.sum(contributions_polarizabilities, dim=-1)
        total_polarizability_spherical = scatter_sum(src=atomic_polarizabilities, index=data["batch"], dim=0, dim_size=num_graphs)
        total_polarizability = spherical_to_cartesian(total_polarizability_spherical, self.change_of_basis)

        if compute_dielectric_derivatives:
            dmu_dr = compute_dielectric_gradients(dielectric=total_dipole, positions=data["positions"])
            dalpha_dr = compute_dielectric_gradients(dielectric=total_polarizability.flatten(-2), positions=data["positions"])
        else:
            dmu_dr = None
            dalpha_dr = None

        return {
            "charges": atomic_charges,
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
            "polarizability": total_polarizability,
            "polarizability_sh": total_polarizability_spherical,
            "dmu_dr": dmu_dr,
            "dalpha_dr": dalpha_dr,
        }


@compile_mode("script")
class EnergyDipolesMACE(torch.nn.Module):
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
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[np.ndarray],
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        edge_irreps: Optional[o3.Irreps] = None,
    ):
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer("num_interactions", torch.tensor(num_interactions, dtype=torch.int64))

        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(irreps_in=node_attr_irreps, irreps_out=node_feats_irreps)
        self.radial_embedding = RadialEmbeddingBlock(r_max=r_max, num_bessel=num_bessel, num_polynomial_cutoff=num_polynomial_cutoff)
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization="component")
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        use_sc_first = "Residual" in str(interaction_cls_first)
        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert len(hidden_irreps) > 1, "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(hidden_irreps[:2])
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(NonLinearDipoleReadoutBlock(hidden_irreps_out, MLP_irreps, gate, dipole_only=False))
            else:
                self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        displacement = torch.zeros((num_graphs, 3, 3), dtype=data["positions"].dtype, device=data["positions"].device)
        if compute_virials or compute_stress or compute_displacement:
            data["positions"], data["shifts"], displacement = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        node_e0 = self.atomic_energies_fn(data["node_attrs"])[num_atoms_arange, data["head"][data["batch"]]]
        e0 = scatter_sum(src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs)

        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(positions=data["positions"], edge_index=data["edge_index"], shifts=data["shifts"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers)

        energies = [e0]
        node_energies_list = [node_e0]
        dipoles = []
        for interaction, product, readout in zip(self.interactions, self.products, self.readouts):
            node_feats, sc = interaction(node_attrs=data["node_attrs"], node_feats=node_feats, edge_attrs=edge_attrs, edge_feats=edge_feats, edge_index=data["edge_index"], cutoff=cutoff)
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"])
            node_out = readout(node_feats).squeeze(-1)
            node_energies = node_out[:, 0]
            energy = scatter_sum(src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs)
            energies.append(energy)
            node_dipoles = node_out[:, 1:]
            dipoles.append(node_dipoles)

        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)
        node_energy = torch.sum(torch.stack(node_energies_list, dim=-1), dim=-1)
        contributions_dipoles = torch.stack(dipoles, dim=-1)
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)
        total_dipole = scatter_sum(src=atomic_dipoles, index=data["batch"].unsqueeze(-1), dim=0, dim_size=num_graphs)
        baseline = compute_fixed_charge_dipole(charges=data["charges"], positions=data["positions"], batch=data["batch"], num_graphs=num_graphs)
        total_dipole = total_dipole + baseline

        forces, virials, stress, _, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
