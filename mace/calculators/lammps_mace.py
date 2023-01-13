from typing import Dict, List, Optional

import torch
from e3nn.util.jit import compile_mode

from mace.modules.utils import get_outputs
from mace.tools.scatter import scatter_sum


@compile_mode("script")
class LAMMPS_MACE(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        mask_ghost: torch.Tensor,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = data["ptr"].numel() - 1
        compute_displacement = False
        if compute_virials or compute_stress:
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
            return {"energy": None, "forces": None, "virials": None, "stress": None}
        displacement = out["displacement"]
        virials: Optional[torch.Tensor] = torch.zeros_like(data["cell"])
        stress: Optional[torch.Tensor] = torch.zeros_like(data["cell"])
        if mask_ghost is not None and displacement is not None:
            # displacement.requires_grad_(True)  # For some reason torchscript needs that.
            node_energy_ghost = node_energy * mask_ghost
            total_energy_ghost = scatter_sum(
                src=node_energy_ghost, index=data["batch"], dim=-1, dim_size=num_graphs
            )
            grad_outputs: List[Optional[torch.Tensor]] = [
                torch.ones_like(total_energy_ghost)
            ]
            virials = torch.autograd.grad(
                outputs=[total_energy_ghost],
                inputs=[displacement],
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )[0]

            if virials is not None:
                virials = -1 * virials
                cell = data["cell"].view(-1, 3, 3)
                volume = torch.einsum(
                    "zi,zi->z",
                    cell[:, 0, :],
                    torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                ).unsqueeze(-1)
                stress = virials / volume.view(-1, 1, 1)
            else:
                virials = torch.zeros_like(displacement)

        total_energy = scatter_sum(
            src=node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
        )

        forces, _, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=False,
            compute_force=compute_force,
            compute_virials=False,
            compute_stress=False,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
            "stress": stress,
        }
