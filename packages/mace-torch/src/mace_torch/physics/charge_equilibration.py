"""Constrained quadratic charge equilibration with implicit linear derivatives.

The energy and constraint follow ACEsuit/mace-scf's MACEQEq/AutogradHardness
at 784a510339aacc4412b61b6926ef1a92bca387c0 (MIT). The matrix is assembled
from the dispatched linear potential, without nested autograd in forward.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from mace_torch.electrostatics import LongRangeGeometry


def constrained_charges(
    matrix: Tensor,
    electronegativity: Tensor,
    charge: Tensor,
    constraints: Tensor | None = None,
) -> Tensor:
    """Solve ``matrix @ q + chi + B @ lambda = 0, B.T @ q = charge``.

    ``matrix [n,n]`` includes hardness (eV/e²), ``electronegativity [n]``
    is in eV/e, and ``charge`` and the returned ``q [n]`` are in e.
    ``constraints [n,g]`` assigns atoms to graphs (one graph when omitted).
    The derivative of ``linalg.solve`` is the implicit derivative of this linear
    system, including its constraint; it supports force-training double backward.
    Singular systems raise, with no regularization or detached-charge fallback.
    """
    count = electronegativity.shape[0]
    if constraints is None:
        constraints = matrix.new_ones((count, 1))
    graphs = constraints.shape[1]
    system = torch.cat(
        (
            torch.cat((matrix, constraints), dim=1),
            torch.cat((constraints.T, matrix.new_zeros((graphs, graphs))), dim=1),
        ),
        dim=0,
    )
    rhs = torch.cat((-electronegativity, charge.reshape(-1)))
    return torch.linalg.solve(system, rhs)[:count]


class ChargeEquilibration(nn.Module):
    """Reference QEq span over a build-time-dispatched potential projection.

    A fused QEq span has the same call: ``(graph, electronegativity, hardness,
    total_charge, geometry) -> charges``. Its descriptor's SCF method distinguishes
    QEq from a nonlinear fixed point and from the legacy nonperiodic alternative.
    The fallback is a dense reference solve, not a large-system iterative solver.
    """

    projection: Any

    def __init__(self, projection: nn.Module):
        super().__init__()
        self.projection = projection

    def potential_matrix(
        self, graph: Mapping[str, Any], geometry: LongRangeGeometry
    ) -> Tensor:
        """``[n_atoms,n_atoms]`` Coulomb potential matrix, in eV/e².

        Molecule-in-box potentials may differ from the energy Hessian by a
        constant in each graph block. Under fixed total charge that changes
        only the Lagrange multiplier, never the charges or their derivatives.
        Geometry preparation is shared by every unit-density evaluation.
        """
        count = graph["positions"].shape[0]
        basis = torch.eye(
            count, dtype=geometry.positions.dtype, device=geometry.positions.device
        )
        cache = self.projection.prepare(geometry)
        # The reference's real-space/slab corrections do not support vmap.
        # Keep the explicit column assembly in this fallback, with no Jacobian
        # construction inside forward. A fused span can replace this whole solve.
        return torch.stack(
            [
                self.projection(cache, column[:, None])[:, 0]
                for column in basis.unbind()
            ],
            dim=1,
        )

    def forward(
        self,
        graph: Mapping[str, Any],
        electronegativity: Tensor,
        hardness: Tensor,
        total_charge: Tensor,
        geometry: LongRangeGeometry,
    ) -> Tensor:
        """Equilibrated charges ``[n_atoms]`` in e, independently per structure.

        Atomic electronegativities/hardness have shape ``[n_atoms]`` and total
        charge ``[n_graphs]``. Tensor-valued membership gives each structure its
        own constraint without reading sizes or specializing to the partition.
        """
        matrix = self.potential_matrix(graph, geometry) + torch.diag(hardness)
        constraints = torch.nn.functional.one_hot(
            graph["batch"], num_classes=graph["ptr"].shape[0] - 1
        ).to(matrix.dtype)
        return constrained_charges(matrix, electronegativity, total_charge, constraints)
