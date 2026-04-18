from typing import Dict, List, Optional

import torch
from e3nn.util.jit import compile_mode
from e3nn import nn, o3
from e3nn.io import CartesianTensor
from e3nn.o3._reduce import ReducedTensorProducts

from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock, LinearDipoleReadoutBlock, LinearDipolePolarReadoutBlock, NonLinearDipolePolarReadoutBlock, LinearLesReadoutBlock, NonLinearLesReadoutBlock
from mace.modules.models import ScaleShiftMACE
from mace.modules.utils import get_atomic_virials_stresses, get_outputs, prepare_graph
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.scatter import scatter_sum
from mace.tools.torch_tools import get_change_of_basis, spherical_to_cartesian

def _copy_mace_readout(
    mace_readout: torch.nn.Module, 
    change_irrep_out: Optional[str] = None, # o3.Irreps("1x1o")
    cueq_config: Optional[CuEquivarianceConfig] = None
) -> torch.nn.Module:
    """
    Helper function to copy a MACE readout block.
    """
    if isinstance(mace_readout, LinearReadoutBlock):
        return LinearReadoutBlock(
            irreps_in=mace_readout.linear.irreps_in,  # type:ignore
            irrep_out=o3.Irreps(change_irrep_out) if change_irrep_out is not None else mace_readout.linear.irreps_out,  # type:ignore
            cueq_config=cueq_config,
        )
    if isinstance(mace_readout, NonLinearReadoutBlock):  # type:ignore
        return NonLinearReadoutBlock(
            irreps_in=mace_readout.linear_1.irreps_in,  # type:ignore
            MLP_irreps=mace_readout.hidden_irreps,
            gate=mace_readout.non_linearity._modules["acts"][  # pylint: disable=W0212
                0
            ].f,
            irrep_out=o3.Irreps(change_irrep_out) if change_irrep_out is not None else mace_readout.linear_2.irreps_out,  # type:ignore
            num_heads=mace_readout.num_heads,
            cueq_config=cueq_config,
        )
    raise TypeError("Unsupported readout type.")

def _copy_mace_readout_tp(
    mace_readout: torch.nn.Module,
    make_w_pos: bool = True,
    cueq_config: Optional[CuEquivarianceConfig] = None,
    use_nonlinear_readout: bool = False,
) -> torch.nn.Module:
    """
    Helper function to copy a MACE readout block.
    """
    print("use_nonlinear_readout for alpha?", use_nonlinear_readout)
    if isinstance(mace_readout, LinearReadoutBlock):
        if use_nonlinear_readout:
            return NonLinearLesReadoutBlock(
                irreps_in=mace_readout.linear.irreps_in,  # type:ignore
                cueq_config=cueq_config,
            )
        else:
            return LinearLesReadoutBlock(
                irreps_in=mace_readout.linear.irreps_in,  # type:ignore
                make_w_pos = make_w_pos,
                cueq_config=cueq_config,
            )
    if isinstance(mace_readout, NonLinearReadoutBlock):  # type:ignore
        if use_nonlinear_readout:
            return NonLinearLesReadoutBlock(
                irreps_in=mace_readout.linear_1.irreps_in,  # type:ignore
                cueq_config=cueq_config,
            )
        else:
            return LinearLesReadoutBlock(
                irreps_in=mace_readout.linear_1.irreps_in,  # type:ignore
                make_w_pos = make_w_pos,
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
        self.use_dipoles = les_arguments.get("use_dipole", False)
        self.use_quads = les_arguments.get("use_quad", False)
        self.use_induced_charges = les_arguments.get("use_induced_charge", False)
        self.use_induced_dipoles = les_arguments.get("use_induced_dipole", False)
        self.use_anisotropic_polarizability = les_arguments.get("use_anisotropic_polarizability", False)
        self.alpha_irreps = les_arguments.get("alpha_irreps", '0e+1o+2e')
        self.alpha_1o_nonlinear_readout = les_arguments.get("alpha_1o_nonlinear_readout", False)
        self.alpha_1o_linear_w_pos = les_arguments.get("alpha_1o_linear_w_pos", True)
        self.make_alpha_positive = les_arguments.get("make_alpha_positive", False)
        self.make_kappa_positive = les_arguments.get("make_kappa_positive", False)

        print("use_dipoles", self.use_dipoles)
        print("use_induced_charges", self.use_induced_charges)
        print("use_induced_dipoles", self.use_induced_dipoles)
        self.les = Les(les_arguments=les_arguments)

        self.les_readouts = torch.nn.ModuleList()
        self.les_u_readouts = torch.nn.ModuleList()
        self.les_quad_2e_readouts = torch.nn.ModuleList()
        self.les_quad_1o_readouts = torch.nn.ModuleList()
        self.les_alpha_readouts = torch.nn.ModuleList()
        self.les_alpha_1o_readouts = torch.nn.ModuleList()
        self.les_alpha_2e_readouts = torch.nn.ModuleList()
        self.les_kappa_readouts = torch.nn.ModuleList()
        self.les_output_scale = les_arguments.get("output_scale", 0.1)
        self.les_kappa_scale = les_arguments.get("kappa_scale", 0.01)
        self.les_alpha_scale = les_arguments.get("alpha_scale", 0.01)
        
        self.readout_input_dims = [
            _get_readout_input_dim(readout) for readout in self.readouts  # type:ignore
        ]
        cueq_config = kwargs.get("cueq_config", None)
        for i, readout in enumerate(self.readouts):  # type:ignore
            self.les_readouts.append(
                _copy_mace_readout(readout, cueq_config=cueq_config)
            )
            if self.use_dipoles:
                self.les_u_readouts.append(
                    #_copy_mace_readout(readout,  change_irrep_out="1x1o", cueq_config=cueq_config)
                    _copy_mace_readout(self.readouts[0],  change_irrep_out="1x1o", cueq_config=cueq_config)
                )
            if self.use_quads:
                mace_irreps = str(self.readouts[0].linear.irreps_in)
                if "2e" in mace_irreps:
                    print("Using l=2 readout to predict quadrupoles.")
                    change_of_basis_quads = ReducedTensorProducts('ij=ji', i="1o", filter_ir_out=['2e']).change_of_basis
                    self.les_quad_2e_readouts.append(
                        _copy_mace_readout(self.readouts[0], change_irrep_out="1x2e", cueq_config=cueq_config)
                    )
                    self.register_buffer("change_of_basis_quads", change_of_basis_quads)
                if "1o" in mace_irreps:
                    print("Using l=1 readout to predict quadrupoles.")
                    self.les_quad_1o_readouts.append(
                        _copy_mace_readout_tp(self.readouts[0], 
                        use_nonlinear_readout=self.alpha_1o_nonlinear_readout,
                        make_w_pos=False,
                        cueq_config=cueq_config
                        )
                    )
            if self.use_induced_charges:
                self.les_kappa_readouts.append(
                    _copy_mace_readout(readout, cueq_config=cueq_config)
                )
            if self.use_induced_dipoles:
                if self.use_anisotropic_polarizability:
                    mace_irreps = str(self.readouts[0].linear.irreps_in)
                    if "2e" in mace_irreps and "2e" in self.alpha_irreps:
                        print("Using l=2 readout to predict anisotropic polarizability.")
                        change_of_basis = CartesianTensor("ij=ji").reduced_tensor_products().change_of_basis
                        self.les_alpha_2e_readouts.append(
                            _copy_mace_readout(self.readouts[0], change_irrep_out="1x0e + 1x2e", cueq_config=cueq_config)
                        )
                        self.register_buffer("change_of_basis", change_of_basis)
                    if "1o" in mace_irreps and "1o" in self.alpha_irreps:
                        #Obtain 2e from l=1 outer products
                        print("Using l=1 readout to predict anisotropic polarizability.")
                        self.les_alpha_1o_readouts.append(
                            _copy_mace_readout_tp(self.readouts[0], 
                            use_nonlinear_readout=self.alpha_1o_nonlinear_readout,
                            make_w_pos=self.alpha_1o_linear_w_pos,
                            cueq_config=cueq_config
                            )
                        )
                    if not ("1o" in mace_irreps or "2e" in mace_irreps):
                        raise ValueError("Unsupported irreps for anisotropic polarizability. Expected '1o' or '2e' in the readout irreps.")
                if not self.use_anisotropic_polarizability or "0e" in self.alpha_irreps:
                    self.les_alpha_readouts.append(
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


        # for backward compatibility
        if not hasattr(self, "les_output_scale"):
            self.les_output_scale = 1.
        if not hasattr(self, "les_kappa_scale"):
            self.les_kappa_scale = 1.
        if not hasattr(self, "les_alpha_scale"):
            self.les_alpha_scale = 1.
        if not hasattr(self, "use_anisotropic_polarizability"):
            self.use_anisotropic_polarizability = False

        # Setting LES cell input to zero when boundary conditions are not periodic
        cell_les = cell.clone()
        pbc_tensor = data["pbc"].to(device=data["cell"].device)
        no_pbc_mask_cfg = ~pbc_tensor.any(dim=-1)
        no_pbc_mask_rows = no_pbc_mask_cfg.repeat_interleave(3)
        cell_les[no_pbc_mask_rows] = torch.zeros(
            (no_pbc_mask_rows.sum(), 3),
            dtype=cell_les.dtype,
            device=cell_les.device
        )
        if displacement is not None:
            symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
            cell_les_view = cell_les.view(-1, 3, 3)
            cell_les_view = cell_les_view + torch.matmul(cell_les_view, symmetric_displacement)
            cell_les = cell_les_view.view_as(cell_les)

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
        node_us_list: List[torch.Tensor] = []
        node_quads_list: List[torch.Tensor] = []
        node_kappas_list: List[torch.Tensor] = []
        node_alphas_list: List[torch.Tensor] = []

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
            if hasattr(self, "use_dipoles") and self.use_dipoles:
                les_u_readout = self.les_u_readouts[i]
                node_us = les_u_readout(node_feats_list[feat_idx])[
                    num_atoms_arange
                ]  # type:ignore
                node_us_list.append(node_us)
                #print('dipoles', i, node_us[:3])
            if hasattr(self, "use_induced_charges") and self.use_induced_charges:
                les_kappa_readout = self.les_kappa_readouts[i]
                node_kappas = les_kappa_readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange, node_heads
                    ]  # type:ignore
                node_kappas_list.append(node_kappas)
            if hasattr(self, "use_quads"):
                if hasattr(self, "les_quad_2e_readouts") and len(self.les_quad_2e_readouts) > i:
                    les_quad_readout = self.les_quad_2e_readouts[i]
                    node_quads = les_quad_readout(node_feats_list[feat_idx])[
                        num_atoms_arange
                    ]  # type:ignore
                    node_quads = spherical_to_cartesian(
                        node_quads, self.change_of_basis_quads
                    )
                    node_quads_list.append(node_quads)
                if hasattr(self, "les_quad_1o_readouts") and len(self.les_quad_1o_readouts) > i:
                    les_quad_readout = self.les_quad_1o_readouts[i]
                    node_quads = les_quad_readout(node_feats_list[feat_idx])[
                        num_atoms_arange
                    ]  # type:ignore
                    node_quads_list.append(node_quads)
            if hasattr(self, "use_induced_dipoles") and self.use_induced_dipoles:
                if hasattr(self, "les_alpha_1o_readouts") and len(self.les_alpha_1o_readouts) > i:
                    les_alpha_readout = self.les_alpha_1o_readouts[i]
                    node_alphas = les_alpha_readout(node_feats_list[feat_idx])[
                        num_atoms_arange
                        ]  # type:ignore
                    node_alphas_list.append(node_alphas)
                if hasattr(self, "les_alpha_2e_readouts") and len(self.les_alpha_2e_readouts) > i:
                    les_alpha_readout = self.les_alpha_2e_readouts[i]
                    node_alphas = les_alpha_readout(node_feats_list[feat_idx])[
                        num_atoms_arange
                        ]  # type:ignore
                    node_alphas = spherical_to_cartesian(
                        node_alphas, self.change_of_basis
                    )
                    node_alphas_list.append(node_alphas)
                if len(self.les_alpha_readouts) > i:
                    les_alpha_readout = self.les_alpha_readouts[i]
                    node_alphas = les_alpha_readout(node_feats_list[feat_idx], node_heads)[
                        num_atoms_arange, node_heads
                        ]  # type:ignore
                    if hasattr(self, "use_anisotropic_polarizability") and self.use_anisotropic_polarizability:
                        eye = torch.eye(3,device=node_alphas.device)
                        node_alphas = node_alphas.unsqueeze(-1).unsqueeze(-1) * eye.unsqueeze(0)
                    node_alphas_list.append(node_alphas)

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        total_energy = e0 + inter_e
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()

        les_q = torch.sum(torch.stack(node_qs_list, dim=1), dim=1) * self.les_output_scale
        if len(node_us_list) > 0:
            les_u = torch.sum(torch.stack(node_us_list, dim=-1), dim=-1) * self.les_output_scale
        else:
            les_u = None

        if len(node_kappas_list) > 0:
            les_kappa = torch.sum(torch.stack(node_kappas_list, dim=1), dim=1) * self.les_kappa_scale
        else:
            les_kappa = None

        if len(node_quads_list) > 0:
            les_quad = torch.sum(torch.stack(node_quads_list, dim=1), dim=1) * self.les_output_scale
            #Make quads traceless:
            traces = les_quad.diagonal(dim1=-1, dim2=-2).sum(dim=1)
            eye = torch.eye(3, device=les_quad.device)
            les_quad = les_quad - eye[None, :, :] * traces[:, None, None] / 3
        else:
            les_quad = None

        if len(node_alphas_list) > 0:
            les_alpha = torch.sum(torch.stack(node_alphas_list, dim=1), dim=1) * self.les_alpha_scale
            #print('les_alpha', les_alpha.shape, les_alpha[:3])
        else:
            les_alpha = None

        if hasattr(self, 'make_alpha_positive') and self.make_alpha_positive and les_alpha is not None:
            if les_alpha.dim() == 2:
                les_alpha = les_alpha**2
            if les_alpha.dim() == 3 and les_alpha.shape[1] == 3 and les_alpha.shape[2] == 3:
                les_alpha = torch.einsum("nij,nkj->nik", les_alpha, les_alpha)
        if hasattr(self, 'make_kappa_positive') and self.make_kappa_positive and les_kappa is not None:
            les_kappa = les_kappa**2
        
        les_positions = data["positions"] if displacement is not None else positions
        les_result = self.les(
            atomic_numbers=data["atomic_numbers"],
            latent_charges=les_q,
            latent_dipoles=les_u,
            latent_quads=les_quad,
            latent_alphas=les_alpha,
            latent_kappas=les_kappa,
            positions=les_positions,
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
            "latent_charges": les_result["latent_charges"],
            "latent_dipoles": les_result["latent_dipoles"],
            "latent_kappas": les_kappa,
            "latent_alphas": les_result["latent_alphas"],
            "latent_quads": les_quad,
            "BEC": les_result["BEC"],
        }
