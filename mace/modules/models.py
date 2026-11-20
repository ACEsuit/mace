###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
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
    safe_double,
)


@compile_mode("script")
class MACE(torch.nn.Module):
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
        use_edge_irreps_first: bool = False,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        embedding_specs: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        lammps_mliap: Optional[bool] = False,
        readout_cls: Optional[Type[NonLinearReadoutBlock]] = NonLinearReadoutBlock,
        keep_last_layer_irreps: bool = False,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
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
        self.use_edge_irreps_first = use_edge_irreps_first

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

        if not use_so3:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p=1)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))

        # interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        def generate_irreps(l):
            str_irrep = "+".join([f"1x{i}e+1x{i}o" for i in range(l + 1)])
            return o3.Irreps(str_irrep)

        sh_irreps_inter = sh_irreps
        if hidden_irreps.count(o3.Irrep(0, -1)) > 0:
            sh_irreps_inter = generate_irreps(max_ell)
        interaction_irreps = (sh_irreps_inter * num_features).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * num_features).sort()[0].simplify()

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)
        if num_interactions == 1 and not keep_last_layer_irreps:
            hidden_irreps_out = str(hidden_irreps[0])
        else:
            hidden_irreps_out = hidden_irreps
        edge_irreps_first = None
        if use_edge_irreps_first and edge_irreps is not None:
            edge_irreps_first = o3.Irreps(f"{edge_irreps.count(o3.Irrep(0, 1))}x0e")
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps_out,
            edge_irreps=edge_irreps_first,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps_out,
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
                LinearReadoutBlock(
                    hidden_irreps_out,
                    o3.Irreps(f"{len(heads)}x0e"),
                    cueq_config,
                    oeq_config,
                )
            )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2 and not keep_last_layer_irreps:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
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
                    LinearReadoutBlock(
                        hidden_irreps,
                        o3.Irreps(f"{len(heads)}x0e"),
                        cueq_config,
                        oeq_config,
                    )
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
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
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
        pbc = ctx.pbc
        node_heads = ctx.node_heads.to(torch.int64)
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(
            vectors.dtype
        )  # [n_graphs, n_heads]
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
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

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
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_concat: List[torch.Tensor] = []

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
            node_feats_concat.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es = readout(node_feats_concat[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]
            energy = scatter_sum(node_es, data["batch"], dim=0, dim_size=num_graphs)
            energies.append(energy)
            node_energies_list.append(node_es)

        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)
        node_energy = torch.sum(torch.stack(node_energies_list, dim=-1), dim=-1)
        node_feats_out = torch.cat(node_feats_concat, dim=-1)

        forces, virials, stress, hessian, edge_forces, _ = get_outputs(
            energy=total_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            pbc=pbc,
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
                pbc=pbc,
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
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
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
        energy_accumulation_dtype: Optional[str] = "float64",
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Widen only the reported graph-energy reductions. Keep the original
        # interaction-energy path for force and stress differentiation. None
        # keeps the reductions in the working dtype. MPS has no float64, so
        # there the reductions stay in the working dtype either way.
        use_float64_accumulation = False
        if energy_accumulation_dtype is not None:
            if energy_accumulation_dtype != "float64":
                raise ValueError("energy_accumulation_dtype must be None or 'float64'")
            use_float64_accumulation = data["node_attrs"].device.type != "mps"
        # Setup
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
        pbc = ctx.pbc
        node_heads = ctx.node_heads.to(torch.int64)
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(
            vectors.dtype
        )  # [n_graphs, num_heads]
        e0_accumulated = e0
        if use_float64_accumulation:
            e0_accumulated = scatter_sum(
                src=node_e0.to(torch.float64),
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )

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
                embedding_node_energy = torch.atleast_1d(
                    self.embedding_readout(node_feats, node_heads)[
                        num_atoms_arange, node_heads
                    ].squeeze(-1)
                )
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy
                if use_float64_accumulation:
                    e0_accumulated = e0_accumulated + scatter_sum(
                        src=embedding_node_energy.to(torch.float64),
                        index=data["batch"],
                        dim=0,
                        dim_size=num_graphs,
                    )

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []

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

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es_list.append(
                readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange, node_heads
                ]
            )

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        total_energy = e0 + inter_e
        if use_float64_accumulation:
            total_energy = (
                e0_accumulated
                + scatter_sum(
                    src=node_inter_es.to(torch.float64),
                    index=data["batch"],
                    dim=-1,
                    dim_size=num_graphs,
                )
            ).to(total_energy.dtype)
        node_energy = safe_double(node_e0.clone()) + safe_double(node_inter_es.clone())

        forces, virials, stress, hessian, edge_forces, _ = get_outputs(
            energy=inter_e,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            pbc=pbc,
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
                pbc=pbc,
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
        atomic_energies: Optional[
            None
        ] = None,  # Just here to make it compatible with energy models, MUST be None
        apply_cutoff: bool = True,  # pylint: disable=unused-argument
        use_agnostic_product: bool = False,  # pylint: disable=unused-argument
        use_reduced_cg: bool = True,  # pylint: disable=unused-argument
        use_so3: bool = False,  # pylint: disable=unused-argument
        distance_transform: str = "None",  # pylint: disable=unused-argument
        radial_type: Optional[str] = "bessel",
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        oeq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        edge_irreps: Optional[o3.Irreps] = None,  # pylint: disable=unused-argument
        use_edge_irreps_first: bool = False,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        assert atomic_energies is None

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readouts
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

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

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
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[1]
                )  # Select only l=1 vectors for last layer
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
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=True
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,  # pylint: disable=W0613
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_edge_forces: bool = False,  # pylint: disable=W0613
        compute_atomic_stresses: bool = False,  # pylint: disable=W0613
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        assert compute_displacement is False
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_dipoles = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            dipoles.append(node_dipoles)

        # Compute the dipoles
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        output = {
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output


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
        atomic_energies: Optional[
            None
        ] = None,  # Just here to make it compatible with energy models, MUST be None
        apply_cutoff: bool = True,  # pylint: disable=unused-argument
        use_agnostic_product: bool = False,  # pylint: disable=unused-argument
        use_reduced_cg: bool = True,  # pylint: disable=unused-argument
        use_so3: bool = False,  # pylint: disable=unused-argument
        distance_transform: str = "None",  # pylint: disable=unused-argument
        radial_type: Optional[str] = "bessel",
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        oeq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        edge_irreps: Optional[o3.Irreps] = None,  # pylint: disable=unused-argument
        use_edge_irreps_first: bool = False,  # pylint: disable=unused-argument
        dipole_only: Optional[bool] = True,  # pylint: disable=unused-argument
        use_polarizability: Optional[bool] = True,  # pylint: disable=unused-argument
        means_stds: Optional[Dict[str, torch.Tensor]] = None,  # pylint: disable=W0613
        use_last_readout_only: bool = False,  # pylint: disable=unused-argument
        use_embedding_readout: bool = False,  # pylint: disable=unused-argument
        readout_cls: Optional[Callable] = None,  # pylint: disable=unused-argument
        pair_repulsion: bool = False,  # pylint: disable=unused-argument
        heads: Optional[List[str]] = None,  # pylint: disable=unused-argument
        only_dipole: bool = False,  # pylint: disable=unused-argument
        atomic_energies_fn: Optional[Callable] = None,
        embedding_specs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        _ = atomic_energies_fn, embedding_specs
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )

        # Predefine buffers to be TorchScript-safe
        self.register_buffer("dipole_mean", torch.zeros(3))
        self.register_buffer("dipole_std", torch.ones(3))
        self.register_buffer(
            "polarizability_mean", torch.zeros(3, 3)
        )  # 3x3 matrix flattened
        self.use_polarizability = use_polarizability
        self.register_buffer("polarizability_std", torch.ones(3, 3))
        self.register_buffer("change_of_basis", get_change_of_basis())
        # self.register_buffer("mean_polarizability_sh", torch.zeros(6))
        # self.register_buffer("std_polarizability_sh", torch.ones(6))
        if means_stds is not None:
            if "dipole_mean" in means_stds:
                self.dipole_mean.data.copy_(means_stds["dipole_mean"])
            if "dipole_std" in means_stds:
                self.dipole_std.data.copy_(means_stds["dipole_std"])
            if "polarizability_mean" in means_stds:
                self.polarizability_mean.data.copy_(means_stds["polarizability_mean"])
            if "polarizability_std" in means_stds:
                self.polarizability_std.data.copy_(means_stds["polarizability_std"])
            # if "mean_polarizability_sh" in means_stds:
            #    self.mean_polarizability_sh.data.copy_(means_stds["mean_polarizability_sh"])
            # if "std_polarizability_sh" in means_stds:
            #    self.std_polarizability_sh.data.copy_(means_stds["std_polarizability_sh"])'''
        assert atomic_energies is None
        # self.use_polarizability = use_polarizability
        # self.use_dipole = use_dipole

        if heads is None:
            heads = ["Default"]
        self.heads = heads

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readouts
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(
            LinearDipolePolarReadoutBlock(
                hidden_irreps,
                use_polarizability=True,
                cueq_config=cueq_config,
            )
        )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                # does it always do polar and dipole together?
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                # hidden_irreps_out = str(
                #     hidden_irreps[1]
                # )  # Select only l=1 vectors for last layer
                hidden_irreps_out = (
                    hidden_irreps  # this is different in the AtomicDipoleMACE
                )
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
                cueq_config=cueq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipolePolarReadoutBlock(
                        hidden_irreps_out,
                        MLP_irreps,
                        gate,
                        use_polarizability=True,
                        cueq_config=cueq_config,
                    )
                )
                # print("Nonlinear irrpes: ", hidden_irreps_out, MLP_irreps)
                # exit()
            else:
                self.readouts.append(
                    LinearDipolePolarReadoutBlock(
                        hidden_irreps,
                        # use_charge=True,
                        use_polarizability=True,
                        cueq_config=cueq_config,
                    )
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,  # pylint: disable=W0613
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_dielectric_derivatives: bool = False,  # no training on derivatives
        compute_edge_forces: bool = False,  # pylint: disable=W0613
        compute_atomic_stresses: bool = False,  # pylint: disable=W0613
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        assert compute_displacement is False
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms = data["ptr"][1:] - data["ptr"][:-1]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        charges = []
        dipoles = []
        polarizabilities = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )

            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

            node_out = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            charges.append(node_out[:, 0])

            if self.use_polarizability:
                node_dipoles = node_out[:, 2:5]
                node_polarizability = torch.cat(
                    (node_out[:, 1].unsqueeze(-1), node_out[:, 5:]), dim=-1
                )
                polarizabilities.append(node_polarizability)
                dipoles.append(node_dipoles)
            else:
                node_dipoles = node_out[:, 1:4]
                dipoles.append(node_dipoles)
                # raise ValueError(
                #    "Polarizability is not used in this model, but it is required for the AtomicDielectricMACE."
                # )
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        atomic_charges = torch.stack(charges, dim=-1).sum(-1)  # [n_nodes,]
        # The idea is to normalize the charges so that they sum to the net charge in the system before predicting the dipole.
        total_charge_excess = scatter_mean(
            src=atomic_charges, index=data["batch"], dim_size=num_graphs
        ) - (data["total_charge"] / num_atoms)
        atomic_charges = atomic_charges - total_charge_excess[data["batch"]]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole_polar(
            charges=atomic_charges,  # or data["charges"], ?????
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        if self.use_polarizability:
            # Compute the polarizabilities
            contributions_polarizabilities = torch.stack(
                polarizabilities, dim=-1
            )  # [n_nodes,6,n_contributions]
            atomic_polarizabilities = torch.sum(
                contributions_polarizabilities, dim=-1
            )  # [n_nodes,6]
            total_polarizability_spherical = scatter_sum(
                src=atomic_polarizabilities,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )  # [n_graphs,6]
            total_polarizability = spherical_to_cartesian(
                total_polarizability_spherical, self.change_of_basis
            )

            if compute_dielectric_derivatives:
                dmu_dr = compute_dielectric_gradients(
                    dielectric=total_dipole,
                    positions=data["positions"],
                )
                dalpha_dr = compute_dielectric_gradients(
                    dielectric=total_polarizability.flatten(-2),
                    positions=data["positions"],
                )
            else:
                dmu_dr = None
                dalpha_dr = None
        else:
            if compute_dielectric_derivatives:
                dmu_dr = compute_dielectric_gradients(
                    dielectric=total_dipole,
                    positions=data["positions"],
                )
            else:
                dmu_dr = None
            total_polarizability = None
            total_polarizability_spherical = None
            dalpha_dr = None

        output = {
            "charges": atomic_charges,
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
            "polarizability": total_polarizability,
            "polarizability_sh": total_polarizability_spherical,
            "dmu_dr": dmu_dr,
            "dalpha_dr": dalpha_dr,
        }
        return output


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
        apply_cutoff: bool = True,  # pylint: disable=unused-argument
        use_reduced_cg: bool = True,  # pylint: disable=unused-argument
        use_so3: bool = False,  # pylint: disable=unused-argument
        distance_transform: str = "None",  # pylint: disable=unused-argument
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        oeq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        edge_irreps: Optional[o3.Irreps] = None,  # pylint: disable=unused-argument
        use_edge_irreps_first: bool = False,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readouts
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

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

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
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[:2]
                )  # Select scalars and l=1 vectors for last layer
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
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=False
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_edge_forces: bool = False,  # pylint: disable=W0613
        compute_atomic_stresses: bool = False,  # pylint: disable=W0613
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, data["head"][data["batch"]]
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_out = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            node_energies = node_out[:, 0]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)
            node_dipoles = node_out[:, 1:]
            dipoles.append(node_dipoles)

        # Compute the energies and dipoles
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"].unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        forces, virials, stress, _, _, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            pbc=data["pbc"] if "pbc" in data else None,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        output = {
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
        return output

###########################################################################################
# Electrostatics / PolarModel (v1)
#
# Registry-based electrostatics backends and a plain torch.nn.Module PolarModel
# that returns polar quantities in an ``extras`` dict.  The legacy PolarMACE
# implementation in mace.modules.extensions is intentionally left untouched.
###########################################################################################

# Model registry ---------------------------------------------------------------------------

_model_registry: Dict[str, Type[torch.nn.Module]] = {}
_MODEL_DENYLIST = {"PolarMACE"}


def register_model(
    name: str, cls: Optional[Type[torch.nn.Module]] = None
) -> Union[Type[torch.nn.Module], Callable[[Type[torch.nn.Module]], Type[torch.nn.Module]]]:
    """Register a model class under ``name``.

    Usable both as ``register_model(name, cls)`` and as a decorator
    ``@register_model(name)``.  Names in ``_MODEL_DENYLIST`` are reserved for
    legacy models with a different interface.
    """

    def _register(model_cls: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
        if name in _MODEL_DENYLIST:
            raise ValueError(f"Model name {name!r} is reserved (denylist)")
        existing = _model_registry.get(name)
        if existing is not None and existing is not model_cls:
            raise ValueError(f"Model {name!r} is already registered")
        _model_registry[name] = model_cls
        return model_cls

    if cls is None:
        return _register
    return _register(cls)


def get_model_class(name: str) -> Type[torch.nn.Module]:
    if name not in _model_registry:
        raise KeyError(
            f"Model {name!r} is not registered. Available: {list(_model_registry)}"
        )
    return _model_registry[name]


def list_models() -> List[str]:
    return list(_model_registry)


# Electrostatics backend registry -----------------------------------------------------------


class ElectrostaticsBackend(torch.nn.Module):
    """Abstract long-range electrostatics solver resolved at model build time."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = dict(config or {})
        self.pbc_handling = self.config.get("pbc_handling", "auto")

    def compute(self, positions: torch.Tensor, charges: torch.Tensor, **kwargs):
        raise NotImplementedError

    def set_pbc_handling(self, mode: str) -> None:
        self.pbc_handling = mode
        self.config["pbc_handling"] = mode

    @property
    def backend_name(self) -> str:
        return self.__class__.__name__


class ReferenceElectrostaticsBackend(ElectrostaticsBackend):
    """Reference backend.

    Uses graph_longrange when it is available; for non-periodic systems (or when
    graph_longrange is not installed) it degrades to a well-defined zero/no-op
    fallback so the model can still be constructed and exercised minimally.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._evaluator = None

    def _get_evaluator(self, cell, pbc):
        if self._evaluator is None:
            try:
                from graph_longrange import LongRangeElectrostatics, PBCHandler

                handler = PBCHandler(pbc_mode=self.pbc_handling)
                self._evaluator = LongRangeElectrostatics(handler, cutoff=10.0)
            except (ImportError, ModuleNotFoundError):
                self._evaluator = False
        return self._evaluator

    def compute(self, positions: torch.Tensor, charges: torch.Tensor, **kwargs):
        cell = kwargs.get("cell")
        pbc = kwargs.get("pbc")
        periodic = pbc is not None and bool(torch.as_tensor(pbc).any())
        evaluator = self._get_evaluator(cell, pbc) if periodic and cell is not None else None
        if evaluator not in (None, False) and cell is not None:
            energy, forces = evaluator(positions, charges, cell)
            return {"electrostatic_energy": energy, "electrostatic_forces": forces}
        # Non-periodic or unavailable solver: zero contribution with correct shapes.
        return {
            "electrostatic_energy": torch.zeros_like(charges),
            "electrostatic_forces": torch.zeros_like(positions),
        }


_electrostatics_backend_registry: Dict[str, Type[ElectrostaticsBackend]] = {
    "reference": ReferenceElectrostaticsBackend,
}


def register_electrostatics_backend(name: str, cls: Type[ElectrostaticsBackend]) -> None:
    """Register a backend class. Idempotent when re-registering the same class."""
    existing = _electrostatics_backend_registry.get(name)
    if existing is not None:
        if existing is cls:
            return
        raise ValueError(f"Electrostatics backend {name!r} is already registered")
    _electrostatics_backend_registry[name] = cls


def get_electrostatics_backend(name: str) -> Type[ElectrostaticsBackend]:
    if name not in _electrostatics_backend_registry:
        raise KeyError(
            f"Electrostatics backend {name!r} is not registered. "
            f"Available: {list(_electrostatics_backend_registry)}"
        )
    return _electrostatics_backend_registry[name]


def build_electrostatics_backend(
    name: str, config: Optional[Dict[str, Any]] = None
) -> ElectrostaticsBackend:
    return get_electrostatics_backend(name)(config)


# PolarModel ---------------------------------------------------------------------------------


@register_model("PolarModel")
class PolarModel(MACE):
    """v1 polarizable MACE with registry-dispatched electrostatics.

    Polar quantities are returned under the ``extras`` key of the forward
    output, and the long-range solver identity is recorded in ``config``.
    """

    POLAR_FLAGS = (
        "kspace_cutoff_factor",
        "atomic_multipoles_max_l",
        "atomic_multipoles_smearing_width",
        "field_feature_max_l",
        "field_feature_widths",
        "field_feature_norms",
        "num_recursion_steps",
        "field_si",
        "include_electrostatic_self_interaction",
        "add_local_electron_energy",
        "quadrupole_feature_corrections",
        "return_electrostatic_potentials",
        "field_norm_factor",
        "fixedpoint_update_config",
        "field_readout_config",
    )

    _BACKBONE_FORWARD_KWARGS = frozenset(
        {
            "training",
            "compute_force",
            "compute_virials",
            "compute_stress",
            "compute_displacement",
            "compute_hessian",
            "compute_edge_forces",
            "compute_atomic_stresses",
            "lammps_mliap",
        }
    )

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
        # Standard MACE options (accepted explicitly so CLI kwargs work).
        pair_repulsion: bool = False,
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        use_agnostic_product: bool = True,
        use_last_readout_only: bool = False,
        use_embedding_readout: bool = False,
        distance_transform: str = "None",
        edge_irreps: Optional[o3.Irreps] = None,
        use_edge_irreps_first: bool = False,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        embedding_specs: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        lammps_mliap: Optional[bool] = False,
        readout_cls: Optional[Type[NonLinearReadoutBlock]] = NonLinearReadoutBlock,
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
        # Polar options (all 15 flags).
        kspace_cutoff_factor: float = 1.0,
        atomic_multipoles_max_l: int = 1,
        atomic_multipoles_smearing_width: float = 0.0,
        field_feature_max_l: int = 2,
        field_feature_widths: Optional[List[float]] = None,
        num_recursion_steps: int = 3,
        field_si: float = 1.0,
        include_electrostatic_self_interaction: bool = True,
        add_local_electron_energy: bool = False,
        quadrupole_feature_corrections: bool = True,
        return_electrostatic_potentials: bool = False,
        field_feature_norms: Optional[List[float]] = None,
        field_norm_factor: float = 0.02,
        fixedpoint_update_config: Optional[Dict[str, Any]] = None,
        field_readout_config: Optional[Dict[str, Any]] = None,
        # Electrostatics backend.
        electrostatics_backend: str = "reference",
        pbc_handling: str = "auto",
    ):
        # Force backbone options for the v1 polar architecture.
        keep_last_layer_irreps = True
        use_agnostic_product = True

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
            pair_repulsion=pair_repulsion,
            apply_cutoff=apply_cutoff,
            use_reduced_cg=use_reduced_cg,
            use_so3=use_so3,
            use_agnostic_product=use_agnostic_product,
            use_last_readout_only=use_last_readout_only,
            use_embedding_readout=use_embedding_readout,
            distance_transform=distance_transform,
            edge_irreps=edge_irreps,
            use_edge_irreps_first=use_edge_irreps_first,
            radial_MLP=radial_MLP,
            radial_type=radial_type,
            heads=heads,
            cueq_config=cueq_config,
            embedding_specs=embedding_specs,
            oeq_config=oeq_config,
            lammps_mliap=lammps_mliap,
            readout_cls=readout_cls,
            keep_last_layer_irreps=keep_last_layer_irreps,
        )

        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

        # Recorded forced flags (used by tests / serialization).
        self.keep_last_layer_irreps = True
        self.use_agnostic_product = True
        self._permute_to_e3nn_convention = True

        # Polar config flags.
        self.kspace_cutoff_factor = kspace_cutoff_factor
        self.atomic_multipoles_max_l = atomic_multipoles_max_l
        self.atomic_multipoles_smearing_width = atomic_multipoles_smearing_width
        self.field_feature_max_l = field_feature_max_l
        self.field_feature_widths = field_feature_widths
        self.field_feature_norms = field_feature_norms
        self.num_recursion_steps = num_recursion_steps
        self.field_si = field_si
        self.include_electrostatic_self_interaction = include_electrostatic_self_interaction
        self.add_local_electron_energy = add_local_electron_energy
        self.quadrupole_feature_corrections = quadrupole_feature_corrections
        self.return_electrostatic_potentials = return_electrostatic_potentials
        self.field_norm_factor = field_norm_factor
        self.fixedpoint_update_config = fixedpoint_update_config
        self.field_readout_config = field_readout_config
        self.electrostatics_backend_name = electrostatics_backend
        self.pbc_handling = pbc_handling

        # Polar readout over the final hidden features.
        self.polar_hidden_irreps = hidden_irreps
        self.polar_hidden_dim = hidden_irreps.dim
        self.polar_readout = LinearDipolePolarReadoutBlock(
            irreps_in=hidden_irreps,
            use_polarizability=True,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )

        # Resolve and record the electrostatics backend at build time.
        backend_cfg = {name: getattr(self, name, None) for name in self.POLAR_FLAGS}
        backend_cfg["pbc_handling"] = pbc_handling
        self.electrostatics_backend = build_electrostatics_backend(
            electrostatics_backend, config=backend_cfg
        )

        self.config = {
            "model_type": "PolarModel",
            "electrostatics_backend": electrostatics_backend,
            "resolved": {
                "electrostatics_backend": type(self.electrostatics_backend).__name__,
                "electrostatics_backend_name": electrostatics_backend,
            },
            "pbc_handling": pbc_handling,
            "r_max": float(r_max),
            "num_bessel": int(num_bessel),
            "num_polynomial_cutoff": int(num_polynomial_cutoff),
            "max_ell": int(max_ell),
            "num_interactions": int(num_interactions),
            "num_elements": int(num_elements),
            "hidden_irreps": str(hidden_irreps),
            "MLP_irreps": str(MLP_irreps),
            "correlation": correlation,
            "radial_type": radial_type,
            "keep_last_layer_irreps": True,
            "use_agnostic_product": True,
        }
        for flag in self.POLAR_FLAGS:
            self.config[flag] = getattr(self, flag, None)

    def set_electrostatic_pbcs(self, pbc_handling: str) -> None:
        self.pbc_handling = pbc_handling
        self.config["pbc_handling"] = pbc_handling
        self.electrostatics_backend.set_pbc_handling(pbc_handling)

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        # Only pass arguments the proven MACE backbone understands.
        backbone_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in self._BACKBONE_FORWARD_KWARGS
        }
        out = super().forward(data, **backbone_kwargs)

        positions = data["positions"]
        batch = data["batch"]
        n_atoms = positions.shape[0]
        num_graphs = out["energy"].shape[0]

        # Last-layer hidden features -> polar readout (2x0e + 1x1o + 1x2e).
        node_feats = out["node_feats"]
        node_feats = node_feats[..., -self.polar_hidden_dim :]
        polar_out = self.polar_readout(node_feats)

        charges = polar_out[:, 0]

        # Fixed number of unrolled recursion steps (no SCF fixed point).
        electrostatic_energy = charges.new_zeros(num_graphs)
        for _ in range(self.num_recursion_steps):
            result = self.electrostatics_backend.compute(
                positions=positions,
                charges=charges,
                cell=getattr(data, "cell", None),
                pbc=getattr(data, "pbc", None),
                batch=batch,
            )
            contribution = result["electrostatic_energy"]
            if contribution.shape == charges.shape:  # per-atom -> per-graph
                contribution = scatter_sum(contribution, batch, dim=0, dim_size=num_graphs)
            electrostatic_energy = electrostatic_energy + contribution

        # Calculator tensor_shapes compatibility.
        density_dim = (self.atomic_multipoles_max_l + 1) ** 2
        density_out = polar_out.new_zeros(n_atoms, density_dim)
        copy_dim = min(density_dim, polar_out.shape[-1])
        density_out[:, :copy_dim] = polar_out[:, :copy_dim]
        spin_density = torch.zeros_like(density_out)

        interaction_energy = scatter_sum(
            out["node_energy"], batch, dim=0, dim_size=num_graphs
        )
        dipole = scatter_sum(polar_out[:, 2:5], batch, dim=0, dim_size=num_graphs)

        extras = {
            "interaction_energy": interaction_energy,
            "electrostatic_energy": electrostatic_energy,
            "electron_energy": charges.new_zeros(num_graphs),
            "density_coefficients": density_out,
            "spin_density": spin_density,
            "charges": charges,
            "spins": spin_density[:, 0],
            "dipole": dipole,
            "polarizability": None,
            "total_charge": scatter_sum(charges, batch, dim=0, dim_size=num_graphs),
            "spin_charge_density": torch.zeros(
                n_atoms, 2, density_dim, device=polar_out.device, dtype=polar_out.dtype
            ),
            "fukui_functions": torch.zeros(
                n_atoms, 2, device=polar_out.device, dtype=polar_out.dtype
            ),
            "fermi_level": charges.new_zeros(num_graphs),
            "external_field": positions.new_zeros(num_graphs, 3),
        }
        if self.return_electrostatic_potentials:
            extras["electrostatic_potentials"] = electrostatic_energy

        out["extras"] = extras
        return out
