"""Exact fixed point and implicit derivatives of an affine density response."""

import torch
from torch import Tensor, nn


class AffineFixedPoint(nn.Module):
    """Solve rho = offset + response @ rho + fermi_response @ mu.

    Density ``[atoms,multipoles]`` is in e Å**l, Fermi level ``[graphs]`` in eV.
    For fixed charge the additional rows impose the supplied total charges.
    For fixed Fermi level the caller includes that field in the offset.
    ``linalg.solve`` differentiates the equations, including second derivatives.
    No iteration history or gradient through a truncated trajectory is used.
    """

    def __init__(self, constant_charge: bool):
        super().__init__()
        self.constant_charge = constant_charge

    def forward(
        self,
        response: Tensor,
        offset: Tensor,
        fermi_response: Tensor,
        membership: Tensor,
        total_charge: Tensor,
        fermi_level: Tensor,
    ) -> tuple[Tensor, Tensor]:
        atoms, width = offset.shape
        count = atoms * width
        system = (
            torch.eye(count, dtype=response.dtype, device=response.device) - response
        )
        right = offset.reshape(-1)
        if self.constant_charge:
            graphs = total_charge.shape[0]
            selector = offset.new_zeros((width,))
            selector[0] = 1
            constraints = (
                (membership[:, :, None] * selector[None, None, :])
                .permute(1, 0, 2)
                .reshape(graphs, count)
            )
            system = torch.cat(
                (
                    torch.cat((system, -fermi_response), dim=1),
                    torch.cat((constraints, system.new_zeros((graphs, graphs))), dim=1),
                ),
                dim=0,
            )
            right = torch.cat((right, total_charge))
        solution = torch.linalg.solve(system, right)
        density = solution[:count].reshape(atoms, width)
        return density, solution[count:] if self.constant_charge else fermi_level
