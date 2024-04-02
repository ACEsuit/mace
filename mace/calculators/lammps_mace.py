from typing import Dict, List, Optional, Tuple

import torch
from torch_runstats.scatter import scatter
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum


@compile_mode("script")
class LAMMPS_MACE(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        local_or_ghost: torch.Tensor,
        compute_virials: bool = False,
        compute_force: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = data["ptr"].numel() - 1
        compute_displacement = False
        if compute_virials:
            compute_displacement = True
        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=compute_displacement,
        )
        node_energy = out["node_energy"]
        if node_energy is None:
            return {
                "total_energy_local": None,
                "node_energy": None,
                "forces": None,
                "virials": None,
            }
        positions = data["positions"]
        edge_index = data["edge_index"]
        batch = data["batch"]
        displacement = out["displacement"]
        vectors = out["vectors"]
        forces: Optional[torch.Tensor] = torch.zeros_like(positions) if compute_force else None
        virials: Optional[torch.Tensor] = torch.zeros_like(data["cell"]) if compute_virials else None
  

        if compute_virials and vectors is not None:
            vector_force: Optional[torch.Tensor]  = torch.zeros_like(vectors)
        else:
            vector_force = None

        if compute_virials and vectors is not None:
            edge_virial: Optional[torch.Tensor]   = torch.zeros_like(vectors).unsqueeze(dim=-1)
        else:
            edge_virial = None

        if edge_virial is not None:
            edge_virial = edge_virial.repeat(1, 1, 3)

        if compute_virials and vectors is not None:
            atom_virial: Optional[torch.Tensor]   = torch.zeros_like(positions).unsqueeze(dim=-1)
        else:
            atom_virial = None

        if atom_virial is not None:
            atom_virial = atom_virial.repeat(1, 1, 3)

        #accumulate energies of local atoms
        node_energy_local = node_energy * local_or_ghost
        total_energy_local = scatter_sum(
            src=node_energy_local, index=data["batch"], dim=-1, dim_size=num_graphs
        )

        # compute partial forces and (possibly) partial virials
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(total_energy_local)
        ]

        if compute_virials and vectors is not None:
            forces, vector_force = torch.autograd.grad(
                outputs=[total_energy_local],  
                inputs=[positions, vectors],  
                grad_outputs=grad_outputs,
                retain_graph=False,  
                create_graph=False, 
                allow_unused=True,
            )

            if vector_force is not None and vectors is not None:
                edge_virial = torch.einsum("zi,zj->zij", vector_force, vectors)
            else:
                edge_virial = None
 
            if edge_virial is not None and edge_index[0] is not None:
                atom_virial = scatter_sum(edge_virial, edge_index[0], dim=0, dim_size=len(positions),)
            else:
                atom_virial = None

            if edge_virial is not None and edge_index[1] is not None:
                scattered_virial = scatter_sum(edge_virial, edge_index[1], dim=0, dim_size=len(positions))
                if atom_virial is not None:
                    atom_virial = (atom_virial + scattered_virial) / 2
                else:
                    atom_virial = None
            else:
                atom_virial = None

            if atom_virial is not None and batch is not None:
                virials = scatter_sum(atom_virial, batch, dim=0,)
            else:
                virials = None

            if virials is not None :
                virials = (virials + virials.transpose(-1, -2)) / 2
            else:
                virials = None

            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
            if virials is not None:
                virials = -1 * virials
            else:
                virials = torch.zeros((1, 3, 3))
            if atom_virial is not None:
                atom_virial = -1 * atom_virial
            else:
                atom_virial = torch.zeros((1,3,3))
        else:
            forces = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)

        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
            "atom_virial": atom_virial,
        }


