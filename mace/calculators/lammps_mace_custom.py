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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return out