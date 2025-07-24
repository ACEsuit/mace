from mace.modules.models import ScaleShiftMACE
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.modules.embeddings import GenericJointEmbedding
from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_sum

from mace.modules.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from mace.modules.utils import (
    compute_fixed_charge_dipole,
    get_atomic_virials_stresses,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
    prepare_graph,
)
from mace.modules.wrapper_ops import CuEquivarianceConfig


def _copy_mace_readout(
    mace_readout: torch.nn.Module, cueq_config: Optional[CuEquivarianceConfig] = None
) -> torch.nn.Module:
    """
    Helper function to copy a MACE readout block.
    """
    if isinstance(mace_readout, LinearReadoutBlock):
        return LinearReadoutBlock(
            irreps_in=mace_readout.linear.irreps_in, # type:ignore
            irrep_out=mace_readout.linear.irreps_out, # type:ignore
            cueq_config=cueq_config,
        )
    elif isinstance(mace_readout, NonLinearReadoutBlock):  # type:ignore
        return NonLinearReadoutBlock(
            irreps_in=mace_readout.linear_1.irreps_in,  # type:ignore
            MLP_irreps=mace_readout.hidden_irreps,
            gate=mace_readout.non_linearity._modules["acts"][0].f,  # type:ignore
            irrep_out=mace_readout.linear_2.irreps_out,  # type:ignore
            num_heads=mace_readout.num_heads,
            cueq_config=cueq_config,
        )
    else:
        raise TypeError("Unsupported readout type.")


def _get_readout_input_dim(block: torch.nn.Module) -> int:
    if isinstance(block, LinearReadoutBlock):
        return block.linear.irreps_in.dim  # type:ignore
    elif isinstance(block, NonLinearReadoutBlock):  # type:ignore
        return block.linear_1.irreps_in.dim  # type:ignore
    else:
        raise TypeError("Unsupported readout type for input dimension retrieval.")

@compile_mode("script")
class MACELES(ScaleShiftMACE):
    def __init__(self, les_arguments: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from les import Les
        except:
            raise ImportError(
                "Cannot import 'les'. Please install the 'les' library from https://github.com/ChengUCB/les."
            )
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
        self, data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        compute_bec: bool = False,
        lammps_mliap: bool = False,
        **kwargs,
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
        node_qs_list = []
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

        for i, (readout, les_readout) in enumerate(
            zip(self.readouts, self.les_readouts)
        ):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es = readout(node_feats_concat[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]
            node_qs = les_readout(node_feats_concat[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]  # type:ignore
            node_qs_list.append(node_qs)
            energy = scatter_sum(node_es, data["batch"], dim=0, dim_size=num_graphs)
            energies.append(energy)
            node_energies_list.append(node_es)

        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)
        node_energy = torch.sum(torch.stack(node_energies_list, dim=-1), dim=-1)
        node_feats_out = torch.cat(node_feats_concat, dim=-1)

        les_q = torch.sum(torch.stack(node_qs_list, dim=1), dim=1)
        les_result = self.les(
            latent_charges=les_q,
            positions=data['positions'],
            cell=data['cell'].view(-1, 3, 3),
            batch=data["batch"],
            compute_energy=True,
            compute_bec=(compute_bec or self.compute_bec),
            bec_output_index=self.bec_output_index,
            )
        les_energy_opt = les_result['E_lr']   
        if les_energy_opt is None:
            les_energy = torch.zeros_like(total_energy)
        else:
            les_energy = les_energy_opt
        total_energy += les_energy

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
            "les_energy": les_energy,
            "latent_charges": les_q,
            "BEC": les_result['BEC'],
        }
