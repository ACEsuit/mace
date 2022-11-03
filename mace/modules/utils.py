###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the ASL License (see ASL.md)
###########################################################################################

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn
import torch.utils.data

from mace.tools import to_numpy
from mace.tools.scatter import scatter_sum

from .blocks import AtomicEnergiesBlock


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = True
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=False,  # For complete dissociation turn to true
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: Optional[torch.Tensor],
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(positions).expand(1, 1, 3)
    if compute_stress and cell is not None and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.einsum(
            "zi,zi->z", cell[:, 0, :], torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
        ).unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
    if forces is not None and virials is None:
        return -1 * forces, torch.zeros_like(positions).expand(1, 1, 3), None
    if forces is None and virials is not None:
        return torch.zeros_like(positions), -1 * virials, None
    if forces is not None and virials is not None:
        return -1 * forces, -1 * virials, stress
    return (
        torch.zeros_like(positions),
        torch.zeros_like(positions).expand(1, 1, 3),
        None,
    )


def compute_forces_site_virials(
    energy: torch.Tensor,
    node_energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    training: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )[0]
    site_virials = torch.autograd.grad(
        outputs=[node_energy],  # [n_nodes, ]
        inputs=[displacement],  # [n_graphs, 3, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )[0]
    if forces is not None and site_virials is None:
        forces = -1 * forces
    if site_virials is not None:
        site_virials = -1 * site_virials
    return forces, site_virials


def get_symmetric_displacement(
    positions: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3, 3, dtype=positions.dtype, device=positions.device,
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3), dtype=positions.dtype, device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum("be,bec->bc", unit_shifts, cell[batch[sender]],)
    return positions, shifts, displacement


def get_outputs(
    energy: torch.Tensor,
    node_energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: Optional[torch.Tensor],
    cell: Optional[torch.Tensor],
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
    compute_site_virials: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if compute_force and compute_virials and displacement is not None:
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=training,
        )
    elif compute_force and compute_site_virials and displacement is not None:
        forces, site_virials = compute_forces_site_virials(
            energy=energy,
            node_energy=node_energy,
            positions=positions,
            displacement=displacement,
            training=training,
        )
        stress = None
        return forces, site_virials, stress
    elif compute_force and not compute_stress:
        forces, virials, stress = (
            compute_forces(energy=energy, positions=positions, training=training),
            None,
            None,
        )
        stress = None
    else:
        forces, virials, stress = (None, None, None)
    return forces, virials, stress


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


def compute_mean_std_atomic_inter_energy(
    data_loader: torch.utils.data.DataLoader, atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        avg_atom_inter_es_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    std = to_numpy(torch.std(avg_atom_inter_es)).item()

    return mean, std


def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader, atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()

    return mean, rms


def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
    num_neighbors = []

    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()
