"""The reference electrostatics solver, and resolving a solver at build time.

A solver builds the long-range energy op from a descriptor. The op reads the
graph it is handed and nothing else: the positions, the batch, the per-graph
periodicity, and the cell. **The cell is the graph's**, whichever the graph
builder put there: the physical cell when any axis is periodic, with an
all-zero row replaced so the volume is finite, and the box built around the
atoms when none is. The op derives the reciprocal cell and the volume from it,
so a strain the derivative engine applies to that cell reaches both, and the
stress it contributes is the derivative of the energy it computed.

**Nothing is decided in forward.** The solver is resolved by name, checked
against the descriptor, and asked for its op once, when the model is built. Its
periodicity handling is fixed then too: each profile maps to one evaluation of
the reference, never to one chosen from the batch it happens to see.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import NotImplementedType
from typing import Any, Protocol

import torch
from mace_core.electrostatics import (
    ElectrostaticsSolverDescriptor,
    SolverCapabilities,
    get_solver,
)
from torch import Tensor, nn

__all__ = [
    "PBC_HANDLING",
    "ElectrostaticsSolver",
    "LongRangeEnergy",
    "ReferenceSolver",
    "build_long_range",
    "build_scf_solve",
    "reciprocal_cell_and_volume",
]

#: The reference's evaluation for each profile. A bulk crystal is a full Ewald
#: sum, a slab adds the dipole correction along its normal, an open molecule is
#: summed exactly in real space, and a batch mixing them treats each structure
#: by its own periodicity, a molecule among them in the box around it with the
#: monopole and dipole corrections that divide by that box's volume.
PBC_HANDLING: dict[str, str] = {
    "full_periodic": "pbc",
    "z_slab": "slab",
    "molecular": "realspace",
    "partial": "mixed_periodic",
}

#: The options the reference understands.
SELF_INTERACTION = "include_self_interaction"


class ElectrostaticsSolver(Protocol):
    """What a solver registered under ``mace.electrostatics_backends.torch`` is."""

    name: str
    capabilities: SolverCapabilities

    def long_range_energy(
        self, descriptor: ElectrostaticsSolverDescriptor
    ) -> nn.Module:
        """The op: ``(graph, source_feats) -> energy per graph``."""
        ...

    def make_scf_solve(
        self, descriptor: ElectrostaticsSolverDescriptor
    ) -> nn.Module | NotImplementedType:
        """A fused self-consistent solve, or ``NotImplemented`` for none."""
        ...


def reciprocal_cell_and_volume(cell: Tensor) -> tuple[Tensor, Tensor]:
    """``2 pi inv(cell)^T`` and ``|det cell|``, per graph.

    A cell with no volume has no reciprocal cell, and gets zeros rather than an
    inverse of a singular matrix, as the frozen tree's graph does.
    """
    cell = cell.view(-1, 3, 3)
    volume = torch.linalg.det(cell).abs()
    inverse, _ = torch.linalg.inv_ex(cell.transpose(-1, -2))
    finite = (volume > 0).view(-1, 1, 1)
    return torch.where(finite, 2 * torch.pi * inverse, torch.zeros_like(cell)), volume


class LongRangeEnergy(nn.Module):
    """The reference long-range energy of a batch of Gaussian multipoles.

    Args:
        descriptor: The solve it computes.
        solver: The name of the solver that built it, which is part of the
            model's state for a solver that is not bit for bit the reference.
    """

    def __init__(self, descriptor: ElectrostaticsSolverDescriptor, solver: str) -> None:
        super().__init__()
        from mace_torch.electrostatics.reference.energy import GTOElectrostaticEnergy

        self.descriptor = descriptor
        self.solver = solver
        self.kspace_cutoff = float(descriptor.kspace_cutoff)
        self.energy = GTOElectrostaticEnergy(
            density_max_l=descriptor.multipole_max_l,
            density_smearing_width=descriptor.smearing_width,
            kspace_cutoff=self.kspace_cutoff,
            include_self_interaction=SELF_INTERACTION
            in descriptor.external_field_flags,
            pbc_handling=PBC_HANDLING[descriptor.periodicity_profile],  # ty: ignore[invalid-argument-type]
        )

    def forward(self, graph: Mapping[str, Any], source_feats: Tensor) -> Tensor:
        """The energy of each graph, in eV.

        Args:
            graph: The batch. Its ``cell`` is read as it is, strained or not.
            source_feats: ``[n_atoms, (multipole_max_l + 1)^2]``, each atom's
                multipoles in the ``e3nn`` component order.
        """
        from mace_torch.electrostatics.reference.kspace import compute_k_vectors_flat

        cell = graph["cell"].view(-1, 3, 3)
        rcell, volume = reciprocal_cell_and_volume(cell)
        k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
            self.kspace_cutoff, cell, rcell
        )
        return self.energy(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=source_feats,
            node_positions=graph["positions"],
            batch=graph["batch"],
            volume=volume,
            pbc=graph["pbc"].view(-1, 3),
        )


class ReferenceSolver:
    """The in-tree solver every other one is measured against.

    Plain torch, on any device torch runs on, differentiable twice through
    autograd: the long-range sum is an explicit function of the positions, the
    cell and the multipoles, so a force and its training gradient come from the
    same graph. It evaluates once; a self-consistent loop around it belongs to
    the model, so it offers no fused solve.
    """

    name = "reference"
    capabilities = SolverCapabilities(
        ops=frozenset({"long_range_energy"}),
        devices=frozenset({"cpu", "cuda", "xpu", "mps"}),
        dtypes=frozenset({"float64", "float32"}),
        slab_normals=frozenset({2}),
        field_flags=frozenset({SELF_INTERACTION}),
        supports_double_backward=True,
        bit_parity=True,
    )

    def long_range_energy(
        self, descriptor: ElectrostaticsSolverDescriptor
    ) -> nn.Module:
        return LongRangeEnergy(descriptor, self.name)

    def make_scf_solve(
        self, descriptor: ElectrostaticsSolverDescriptor
    ) -> nn.Module | NotImplementedType:
        del descriptor
        return NotImplemented


def build_long_range(
    descriptor: ElectrostaticsSolverDescriptor,
    *,
    solver: str = "reference",
    trains_derivatives: bool = False,
) -> nn.Module:
    """The long-range op for a model, resolved once.

    Args:
        descriptor: The solve.
        solver: The registered solver to build it with.
        trains_derivatives: Whether the model trains on forces or stress, which
            needs the solve differentiated twice.

    Raises:
        SolverNotAvailableError: If no solver has that name, or it did not
            import.
        UnsupportedSolveError: If the solver declines the solve, or cannot be
            differentiated twice and the model trains on derivatives. Nothing
            is substituted for it.
    """
    backend: ElectrostaticsSolver = get_solver(solver)
    backend.capabilities.require(descriptor, solver)
    if trains_derivatives:
        backend.capabilities.require_double_backward(
            solver, "training on forces or stress"
        )
    return backend.long_range_energy(descriptor)


def build_scf_solve(
    descriptor: ElectrostaticsSolverDescriptor, *, solver: str = "reference"
) -> nn.Module | None:
    """A solver's fused self-consistent solve, or ``None`` when it has none.

    ``None`` is the model's cue to run the loop itself, one long-range
    evaluation per iteration.
    """
    backend: ElectrostaticsSolver = get_solver(solver)
    backend.capabilities.require(descriptor, solver)
    span = backend.make_scf_solve(descriptor)
    return None if span is NotImplemented else span
