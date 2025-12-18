from typing import Dict, List, Optional

import torch
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum


@compile_mode("script")
class LAMMPS_MACE(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)

        if not hasattr(model, "heads"):
            model.heads = [None]
        self.register_buffer(
            "head",
            torch.tensor(
                self.model.heads.index(kwargs.get("head", self.model.heads[-1])),
                dtype=torch.long,
            ).unsqueeze(0),
        )

        # Best-effort detection of MACEField-like models (constant folded into TorchScript)
        self.is_macefield: bool = bool(
            hasattr(model, "field_feats") or hasattr(model, "field_linear")
        )

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        local_or_ghost: torch.Tensor,
        compute_virials: bool = False,
        # --- MACEField extensions (safe defaults for plain MACE) ---
        electric_field: Optional[torch.Tensor] = None,  # [3] or [1,3] or [n_graphs,3]
        compute_polarization: bool = False,
        compute_becs: bool = False,
        compute_polarizability: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = data["ptr"].numel() - 1
        compute_displacement = bool(compute_virials)

        data["head"] = self.head

        # --- call underlying model ---
        if self.is_macefield:
            # Always supply a field to MACEField (it affects energies/forces even if you don't output P/BEC/alpha)
            if electric_field is None:
                if "electric_field" in data:
                    ef = data["electric_field"]
                else:
                    # fallback: zero field
                    ef = torch.zeros(
                        (1, 3),
                        device=data["positions"].device,
                        dtype=data["positions"].dtype,
                    )
            else:
                ef = electric_field

            # normalise shape to [n_graphs, 3] or [1, 3]
            ef = ef.to(
                device=data["positions"].device, dtype=data["positions"].dtype
            ).view(-1, 3)

            # If any higher response requested, polarization must be on
            do_pol = bool(
                compute_polarization or compute_becs or compute_polarizability
            )

            out = self.model(
                data,
                training=False,
                compute_force=False,
                compute_virials=False,
                compute_stress=False,
                compute_displacement=compute_displacement,
                compute_polarization=do_pol,
                compute_becs=bool(compute_becs),
                compute_polarizability=bool(compute_polarizability),
                electric_field=ef,
            )
        else:
            # Plain MACE path (unchanged)
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
                # MACEField extras
                "polarization": None,
                "becs": None,
                "polarizability": None,
            }

        positions = data["positions"]
        displacement = out.get("displacement", None)

        forces: Optional[torch.Tensor] = torch.zeros_like(positions)
        virials: Optional[torch.Tensor] = torch.zeros_like(data["cell"])

        # accumulate energies of local atoms
        node_energy_local = node_energy * local_or_ghost
        total_energy_local = scatter_sum(
            src=node_energy_local, index=data["batch"], dim=-1, dim_size=num_graphs
        )

        # compute partial forces and (possibly) partial virials
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(total_energy_local)
        ]

        if compute_virials and displacement is not None:
            forces, virials = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions, displacement],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            forces = -forces if forces is not None else torch.zeros_like(positions)
            virials = (
                -virials if virials is not None else torch.zeros_like(displacement)
            )
        else:
            forces = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            forces = -forces if forces is not None else torch.zeros_like(positions)

        # MACEField extras (will be None unless you asked for them)
        pol = out.get("polarization", None) if self.is_macefield else None
        becs = out.get("becs", None) if self.is_macefield else None
        alpha = out.get("polarizability", None) if self.is_macefield else None

        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
            # MACEField extras
            "polarization": pol,
            "becs": becs,
            "polarizability": alpha,
        }
