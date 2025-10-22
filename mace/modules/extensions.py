from typing import Dict, List, Optional, Type, Callable, Union, Any, Tuple
from abc import abstractmethod

import torch
from e3nn.util.jit import compile_mode
from e3nn import nn, o3
import numpy as np

from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock
from mace.modules.models import ScaleShiftMACE
from mace.modules.utils import get_atomic_virials_stresses, get_outputs, prepare_graph
from mace.modules.wrapper_ops import (
    CuEquivarianceConfig,
    FullyConnectedTensorProduct,
    Linear,
    TensorProduct,
)
from mace.tools.scatter import scatter_sum

from .irreps_tools import reshape_irreps, tp_out_irreps_with_instructions
from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    EquivariantProductBasisWithSelfMagmomBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)

from .utils import (
    get_symmetric_displacement,
    get_edge_vectors_and_lengths
)

from .radial import ZBLBasis

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

# === magnetic mace integration ===
class SHModule(torch.nn.Module):
    """Example of how to use SphericalHarmonics from within a
    `torch.nn.Module`"""

    def __init__(self, l_max):
        super().__init__()
        try:
            import sphericart.torch
        except ImportError as exc:
            raise ImportError(
                "Cannot import 'sphericart.torch'. Please install the 'sphericart.torch' library from https://github.com/lab-cosmo/sphericart."
            ) from exc
        self.SH = sphericart.torch.SolidHarmonics(l_max)

    def forward(self, xyz):
        sh = self.SH(torch.index_select(
                xyz, 1, torch.tensor([2, 0, 1], dtype=torch.long,device=xyz.device)
            ))
        return sh
    
class ChebychevBasisWithConst(torch.nn.Module):
    """
    Fully differentiable Chebyshev basis (T₀ to Tₙ₋₁) using recurrence.
    Returns full set including constant term T₀(x) = 1.
    """

    def __init__(self, r_max: float, num_basis=8):
        super().__init__()
        self.num_basis = num_basis
        self.r_max = r_max
        self.register_buffer("n", torch.arange(num_basis))  # compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [N, 1], assumed in [-1, 1]

        Returns:
            Tensor of shape [N, num_basis]: T₀(x), T₁(x), ..., Tₙ₋₁(x)
        """
        T0 = torch.ones_like(x)       # T₀(x)
        if self.num_basis == 1:
            return T0

        T1 = x                        # T₁(x)
        B = [T0, T1]

        for _ in range(2, self.num_basis):
            T2 = 2 * x * T1 - T0
            B.append(T2)
            T0, T1 = T1, T2

        out = torch.cat(B[:self.num_basis], dim=-1)  # shape: [N, num_basis]
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis})"

@compile_mode("script")
class ChebychevBasis2(torch.nn.Module):
    """
    Fully differentiable Chebyshev basis (T_n) using recurrence.
    Matches behavior of torch.special.chebyshev_polynomial_t(x, n) in shape and interface.
    """

    def __init__(self, r_max: float, num_basis=8):
        super().__init__()
        self.num_basis = num_basis
        self.r_max = r_max
        self.register_buffer("n", torch.arange(1, num_basis + 1))  # for compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [N, 1], assumed in [-1, 1]

        Returns:
            Tensor of shape [N, num_basis]: T₁(x), T₂(x), ..., Tₙ(x)
        """
        B = [x]  # T₁(x)
        T0 = torch.ones_like(x)  # T₀(x)
        T1 = x
        for _ in range(2, self.num_basis + 1):
            T2 = 2 * x * T1 - T0
            B.append(T2)
            T0, T1 = T1, T2

        out = torch.cat(B[:self.num_basis], dim=-1)  # shape: [N, num_basis]
        return out

@compile_mode("script")
class MagneticInteractionBlock(InteractionBlock):
    def __init__(
        self,
        magmom_node_inv_feats_irreps: Optional[o3.Irreps] = None,
        magmom_node_attrs_irreps: Optional[o3.Irreps] = None,
        **kwargs,
    ) -> None:
        self.magmom_node_inv_feats_irreps = magmom_node_inv_feats_irreps
        self.magmom_node_attrs_irreps = magmom_node_attrs_irreps
        super().__init__(**kwargs)

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        magmom_node_inv_feats: Optional[torch.Tensor] = None,
        magmom_node_attrs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

@compile_mode("script")
class MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock(MagneticInteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None

        print("into MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock")
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        print("===done init linear===")
        # TensorProduct for real space
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        print("===done init conv_tp===")
        # TensorProduct in magnetic moment space
        magmom_irreps_mid, magmom_instructions = tp_out_irreps_with_instructions(
            #self.conv_tp.irreps_out,
            irreps_mid,
            self.magmom_node_attrs_irreps,
            self.target_irreps,
        )
        self.magmom_conv_tp = TensorProduct(
            self.conv_tp.irreps_out,
            self.magmom_node_attrs_irreps,
            magmom_irreps_mid,
            instructions=magmom_instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        print("===done init magmom conv_tp===")
        # Convolution weights 
        input_dim = self.edge_feats_irreps.num_irreps
        magmom_input_dim = self.magmom_node_inv_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim + magmom_input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # transforming from radial l channels to magnetic l channels
        self.conv_tp_weights_magmom = nn.FullyConnectedNet(
            [input_dim + magmom_input_dim, ] + [self.magmom_conv_tp.weight_numel, ]
        )

        # Linear
        self.irreps_out = self.target_irreps

        self.magmom_linear = Linear(
            self.magmom_conv_tp.irreps_out,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        self.magmom_skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor, # (n_edges, n_basis)
        edge_index: torch.Tensor,
        magmom_node_inv_feats: torch.Tensor,
        magmom_node_attrs: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)
        
        # boardcast node feats to number of nodes
        magmom_inv_feats_j = magmom_node_inv_feats[sender]
        
        edge_feats_with_magmom = torch.cat([edge_feats, magmom_inv_feats_j], dim=-1)        
        
        # combined learnable radial
        tp_weights = self.conv_tp_weights(edge_feats_with_magmom)

        # density normalization
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        
        tp_weights_magmom = self.conv_tp_weights_magmom(edge_feats_with_magmom)
        
        magmom_mji = self.magmom_conv_tp(
            mji, magmom_node_attrs[sender], tp_weights_magmom
        )  # [n_edges, irreps]
        
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        
        magmom_message = scatter_sum(
            src=magmom_mji, index=receiver, dim = 0, dim_size=num_nodes,
        )

        magmom_message = self.magmom_linear(magmom_message) / (density + 1)
        magmom_message = self.magmom_skip_tp(magmom_message, node_attrs)
        return (
            self.reshape(magmom_message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]

@compile_mode("script")
class MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock(MagneticInteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # TensorProduct for real space
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # TensorProduct in magnetic moment space
        magmom_irreps_mid, magmom_instructions = tp_out_irreps_with_instructions(
            #self.conv_tp.irreps_out,
            irreps_mid,
            self.magmom_node_attrs_irreps,
            self.target_irreps,
        )
        self.magmom_conv_tp = TensorProduct(
            self.conv_tp.irreps_out,
            self.magmom_node_attrs_irreps,
            magmom_irreps_mid,
            instructions=magmom_instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        
        # Convolution weights 
        input_dim = self.edge_feats_irreps.num_irreps
        magmom_input_dim = self.magmom_node_inv_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim + magmom_input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # transforming from radial l channels to magnetic l channels
        self.conv_tp_weights_magmom = nn.FullyConnectedNet(
            [input_dim + magmom_input_dim, ] + [self.magmom_conv_tp.weight_numel, ]
        )

        # Linear
        self.irreps_out = self.target_irreps

        self.magmom_linear = Linear(
            self.magmom_conv_tp.irreps_out,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # self.magmom_skip_tp = FullyConnectedTensorProduct(
        #     self.irreps_out,
        #     self.node_attrs_irreps,
        #     self.irreps_out,
        #     cueq_config=self.cueq_config,
        # )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)
        

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor, # (n_edges, n_basis)
        edge_index: torch.Tensor,
        magmom_node_inv_feats: torch.Tensor,
        magmom_node_attrs: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        
        num_nodes = node_feats.shape[0]

        # residue connection
        sc = self.skip_tp(node_feats, node_attrs)

        #
        node_feats = self.linear_up(node_feats)
        
        # boardcast node feats to number of nodes
        magmom_inv_feats_j = magmom_node_inv_feats[sender]

        print("======")
        print(type(edge_feats))
        print(edge_feats)
        print(edge_feats.shape)
        print(type(magmom_inv_feats_j))
        print(magmom_inv_feats_j.shape)
        
        print("======")
        edge_feats_with_magmom = torch.cat([edge_feats, magmom_inv_feats_j], dim=-1)        
        
        # combined learnable radial
        tp_weights = self.conv_tp_weights(edge_feats_with_magmom)

        # density normalization
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        
        tp_weights_magmom = self.conv_tp_weights_magmom(edge_feats_with_magmom)
        
        magmom_mji = self.magmom_conv_tp(
            mji, magmom_node_attrs[sender], tp_weights_magmom
        )  # [n_edges, irreps]
        
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        
        magmom_message = scatter_sum(
            src=magmom_mji, index=receiver, dim=0, dim_size=num_nodes,
        )

        magmom_message = self.magmom_linear(magmom_message) / (density + 1)

        return (
            self.reshape(magmom_message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]

@compile_mode("script")
class MagneticMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        m_max: List[int],
        num_mag_radial_basis: int, 
        max_m_ell: int,
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
    ):
        super().__init__()        
        try:
            import sphericart.torch
        except ImportError as exc:
            raise ImportError(
                "Cannot import 'sphericart.torch'. Please install the 'sphericart.torch' library from https://github.com/lab-cosmo/sphericart."
            ) from exc

        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "m_max", torch.tensor(m_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if heads is None:
            heads = ["default"]
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
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(p=num_polynomial_cutoff)
            self.pair_repulsion = True

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        # --- magnetic stuffs ---
        # m_max is not used here but Chebychev is still on (-1, 1)
        # this needs to have a specicies dependent transform
        self.mag_radial_embedding = ChebychevBasis2(
            r_max = 0.0,
            num_basis=num_mag_radial_basis,
        )

        magmom_sh_irreps = o3.Irreps.spherical_harmonics(max_m_ell)

        # simplify this later
        self.mag_solid_harmoics = SHModule(o3.SphericalHarmonics(
            magmom_sh_irreps, normalize=True, normalization="component"
        )._lmax)
    
        # --- interaction and product basis modules ---
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
            magmom_node_inv_feats_irreps=o3.Irreps(f"{self.mag_radial_embedding.num_basis}x0e"),
            magmom_node_attrs_irreps=magmom_sh_irreps
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        
        # define class for many body interaction
        prod = EquivariantProductBasisWithSelfMagmomBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            # assume only a single correlation
            correlation=correlation[0],
            use_sc=use_sc_first,
            num_elements=len(self.atomic_numbers),
            cueq_config=cueq_config,
            magmom_node_inv_feats_irreps=o3.Irreps(f"{self.mag_radial_embedding.num_basis}x0e"),
            magmom_node_attrs_irreps=o3.Irreps.spherical_harmonics(self.mag_solid_harmoics.SH.l_max())
        )

        self.products = torch.nn.ModuleList([prod])
        self.readouts = torch.nn.ModuleList()
        self.readouts.append(
            LinearReadoutBlock(
                hidden_irreps, o3.Irreps(f"{len(heads)}x0e"), cueq_config
            )
        )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
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
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
                magmom_node_inv_feats_irreps=o3.Irreps(f"{self.mag_radial_embedding.num_basis}x0e"),
                magmom_node_attrs_irreps=magmom_sh_irreps
            )

            self.interactions.append(inter)
            prod = EquivariantProductBasisWithSelfMagmomBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                # assume only a single correlation
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc = True,
                cueq_config=prod.cueq_config,
                magmom_node_inv_feats_irreps=o3.Irreps(f"{self.mag_radial_embedding.num_basis}x0e"),
                magmom_node_attrs_irreps=o3.Irreps.spherical_harmonics(self.mag_solid_harmoics.SH.l_max())
            )

            self.products.append(prod)
            
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                        cueq_config,
                    )
                )
            else:
                self.readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps, o3.Irreps(f"{len(heads)}x0e"), cueq_config
                    )
                )

    @abstractmethod
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        raise NotImplementedError


@compile_mode("script")
class MagneticGinzburgScaleShiftMACE(MagneticMACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        num_mag_radial_basis_one_body = kwargs.pop("num_mag_radial_basis_one_body")
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=0.0
        )

        # coefficient for the chebyshev polynomails
        self.onebody_magmombasis_coeffs = torch.nn.Parameter(torch.randn(len(self.atomic_numbers), num_mag_radial_basis_one_body, len(self.heads)))

        self.one_body_cheb_basis_with_const = ChebychevBasisWithConst(
            r_max = 1.0,
            num_basis = num_mag_radial_basis_one_body,
        )

        # correction to shift E0s for each species
        self.register_buffer(
            "one_body_magmom_const_correction", torch.zeros(len(self.atomic_numbers), len(self.heads))
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
        compute_magforces: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        data["magmom"].requires_grad_(True)
        
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )
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
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )  # [n_graphs, num_heads]

        # node embedding on species
        node_feats = self.node_embedding(data["node_attrs"])

        # prepare the Rnl and Ylm
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, _ = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        # --- magnetic stuffs ---
        
        magmom_lenghts = torch.norm(data["magmom"], dim=-1, keepdim=True)
        element_dependent_scaling = self.m_max[torch.argmax(data["node_attrs"], dim=1)].unsqueeze(-1)
        element_dependent_scaling.requires_grad_(True)
        element_dependent_scaling.retain_grad()
        
        magmom_lenghts_trans = 1 - 2 * (magmom_lenghts / element_dependent_scaling) ** 2
        magmom_node_attrs = self.mag_solid_harmoics(data["magmom"])

        #
        magmom_node_feats = self.mag_radial_embedding(magmom_lenghts_trans) # (n_atoms, n_basis)

        # one body contribution radials, this is with constant shift so that it can be fitted
        magmom_one_body_radials = self.one_body_cheb_basis_with_const(
            magmom_lenghts_trans
        )
        
        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list = []
        
        for (idx, (interaction, product, readout)) in enumerate(zip(
            self.interactions, self.products, self.readouts
        )):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                magmom_node_inv_feats=magmom_node_feats,
                magmom_node_attrs=magmom_node_attrs
            )
          
            node_feats = product(
                node_feats=node_feats, sc=sc,node_attrs=data["node_attrs"], 
                magmom_node_inv_feats=magmom_node_feats,
                magmom_node_attrs=magmom_node_attrs,
            )
            
            node_feats_list.append(node_feats)
            if idx == (len(self.readouts) - 1):    
                # linear (natom, num_basis) -> (natom, 1)
                # remove certain constant to make it matches with E0, 
                # self.one_body_magmom_const_correction is computed outside after pre-training
                # Select the correct coefficient row for each atom via einsum
                selected_coeffs = torch.einsum(
                    "ns,sbh->nbh", data["node_attrs"], self.onebody_magmombasis_coeffs
                )
                #
                one_body_correction = torch.einsum(
                    'ns,sh->nh', data["node_attrs"], self.one_body_magmom_const_correction
                )
                # Compute dot product over nbasis → (n_nodes, num_heads)
                onebody_magmom_contri = (magmom_one_body_radials.unsqueeze(-1) * selected_coeffs).sum(dim=1)

                # apply correction so that the zero matches E0 exactly
                onebody_magmom_contri -= one_body_correction

                # Gather energy per atom + one-body magmom contribution for each head
                node_es_list.append(
                    readout(node_feats, node_heads)[num_atoms_arange, node_heads] +
                    onebody_magmom_contri[num_atoms_arange, node_heads]
                )
            else:
                node_es_list.append(
                    readout(node_feats, node_heads)[num_atoms_arange, node_heads]
                )  # {[n_nodes, ], }

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1) 

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es, node_heads)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]
        
        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es
        forces, virials, stress, hessian, _, magforces = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            magmoms=data["magmom"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_magforces=compute_magforces,
        )
        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "magforces": magforces,
            # "edge_froces": edge_froces,
            "virials": virials,
            "stress": stress,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }
        return output
    

# this does not differentiate through SCF but just a convenient wrapper for equilibrating magmom 
# for a given position. Still later we can do something like:
# given position also predict the magnetic moment based on the previous magnetic moment
# to accelerate SCF cycles that have to be done
class MagneticSCFMACE(torch.nn.Module):
    def __init__(self, model, n_scf_step=10, scf_tol=1e-5, scf_logging=False, scf_step_size=1.0, use_scf = True):
        super().__init__()
        self.magmom_mace = model # original magnetic mace
        self.n_scf_step = n_scf_step
        self.scf_tol = scf_tol
        self.cache_magmom = None
        self.scf_logging = scf_logging
        self.scf_step_size = scf_step_size
        self.use_scf = use_scf
 
        self.cache_magmom = None

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        
        device = next(self.magmom_mace.parameters()).device

        # === Initialize magmom ===
        if "magmom" in data:
            magmom = data["magmom"].detach().to(device).clone()
        elif self.cache_magmom is not None:
            magmom = self.cache_magmom.clone()
        else:
            raise ValueError("No initial magnetic moment provided and no cache available.")

        magmom = magmom.to(device)
        magmom.requires_grad_(True)
        energy_history = []

        # === Define optimizer ===
        if self.use_scf:
            optimizer = torch.optim.LBFGS([magmom], max_iter=self.n_scf_step, tolerance_grad=self.scf_tol, \
                                          line_search_fn="strong_wolfe", lr=self.scf_step_size)

            def closure():
                optimizer.zero_grad()

                # Update magnetic moments in config
                data["magmom"] = magmom

                # Evaluate model
                output = self.magmom_mace(
                    data,
                    training=training,
                    compute_force=compute_force,
                    compute_virials=compute_virials,
                    compute_stress=compute_stress,
                    compute_displacement=compute_displacement,
                )

                energy = output["energy"][0]
                energy_history.append(energy.item())

                # Set gradient manually from mag_forces
                magmom.grad = -output["magforces"].detach()
                if self.scf_logging:
                    print(f"[SCF LBFGS] Energy = {energy.item():.6f} | Mag force norm = {magmom.grad.norm().item():.6f}")
                return energy

            optimizer.step(closure)

            # Cache final magnetic moments
            self.cache_magmom = magmom.detach()

            # Final output (evaluate one last time with final magmom)
            data["dft_magmom"] = magmom

        final_output = self.magmom_mace(
            data,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
        )

        # Add SCF info to output
        final_output["scf_energy_history"] = torch.tensor(energy_history, dtype=torch.float32)
        final_output["scf_steps"] = len(energy_history)
        final_output["equilibrated_magmom"] = magmom.detach()

        return final_output
