###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn
import torch.utils.data
from scipy.constants import c, e

from mace.tools import to_numpy
from mace.tools.scatter import scatter_mean, scatter_std, scatter_sum
from mace.tools.torch_geometric.batch import Batch

from .blocks import AtomicEnergiesBlock
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, ncols=55)
import torch.distributed as dist
import torch_geometric

from collections import defaultdict
from torch_geometric.utils import remove_isolated_nodes, contains_isolated_nodes

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
        allow_unused=True,  # For complete dissociation turn to true
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
    cell: torch.Tensor,
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
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.einsum(
            "zi,zi->z",
            cell[:, 0, :],
            torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
        ).unsqueeze(-1)
        stress = virials / (volume.view(-1, 1, 1) + 1e-16)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress


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
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
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
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return positions, shifts, displacement


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: Optional[torch.Tensor],
    cell: torch.Tensor,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (compute_virials or compute_stress) and displacement is not None:
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=training,
        )
    elif compute_force:
        forces, virials, stress = (
            compute_forces(energy=energy, positions=positions, training=training),
            None,
            None,
        )
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


def _check_non_zero(std):
    if np.any(std == 0):
        logging.warning(
            "Standard deviation of the scaling is zero, Changing to no scaling"
        )
        std[std == 0] = 1
    return std


def extract_invariant(x: torch.Tensor, num_layers: int, num_features: int, l_max: int):
    out = []
    for i in range(num_layers - 1):
        out.append(
            x[
                :,
                i
                * (l_max + 1) ** 2
                * num_features : (i * (l_max + 1) ** 2 + 1)
                * num_features,
            ]
        )
    out.append(x[:, -num_features:])
    return torch.cat(out, dim=-1)


def compute_mean_std_atomic_inter_energy(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []
    head_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=0, dim_size=batch.num_graphs
        )[torch.arange(batch.num_graphs), batch.head]
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        avg_atom_inter_es_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        head_list.append(batch.head)

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    head = torch.cat(head_list, dim=0)  # [total_n_graphs]
    # mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    # std = to_numpy(torch.std(avg_atom_inter_es)).item()
    mean = to_numpy(scatter_mean(src=avg_atom_inter_es, index=head, dim=0).squeeze(-1))
    std = to_numpy(scatter_std(src=avg_atom_inter_es, index=head, dim=0).squeeze(-1))
    std = _check_non_zero(std)

    return mean, std


def _compute_mean_std_atomic_inter_energy(
    batch: Batch,
    atomic_energies_fn: AtomicEnergiesBlock,
) -> Tuple[torch.Tensor, torch.Tensor]:
    head = batch.head
    node_e0 = atomic_energies_fn(batch.node_attrs)
    graph_e0s = scatter_sum(
        src=node_e0, index=batch.batch, dim=0, dim_size=batch.num_graphs
    )[torch.arange(batch.num_graphs), head]
    graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
    atom_energies = (batch.energy - graph_e0s) / graph_sizes
    return atom_energies

# For head
def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
    rank=0,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []
    head_list = []
    head_batch = []

    if rank == 0:
        data_iter = tqdm(data_loader)
    else:
        data_iter = data_loader
    for batch in data_iter:
        head = batch.head
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=0, dim_size=batch.num_graphs
        )[torch.arange(batch.num_graphs), head]
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }
        head_list.append(head)
        head_batch.append(head[batch.batch])

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }
    head = torch.cat(head_list, dim=0)  # [total_n_graphs]
    head_batch = torch.cat(head_batch, dim=0)  # [total_n_graphs]
    
    mean = to_numpy(scatter_mean(src=atom_energies, index=head, dim=0))
    rms = to_numpy(
        torch.sqrt(
            scatter_mean(src=torch.square(forces), index=head_batch, dim=0).mean(-1)
        )
    )
    rms = _check_non_zero(rms)

    return mean, rms

# def compute_mean_rms_energy_forces(
#     data_loader: torch.utils.data.DataLoader,
#     atomic_energies: np.ndarray,
#     rank=0,
# ) -> Tuple[float, float]:
#     atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

#     atom_energy_list = []
#     forces_list = []

#     if rank == 0:
#         data_iter = tqdm(data_loader)
#     else:
#         data_iter = data_loader
#     for batch in data_iter:
#         node_e0 = atomic_energies_fn(batch.node_attrs)
#         graph_e0s = scatter_sum(
#             src=node_e0, index=batch.batch, dim=0, dim_size=batch.num_graphs
#         )[:,0]
#         graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
#         atom_energy_list.append(
#             (batch.energy - graph_e0s) / graph_sizes
#         )  # {[n_graphs], }
#         forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

#     atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
#     forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

#     mean = to_numpy(torch.mean(atom_energies)).item()
#     rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()
#     rms = _check_non_zero(rms)

#     return mean, rms

def _compute_mean_rms_energy_forces(
    batch: Batch,
    atomic_energies_fn: AtomicEnergiesBlock,
) -> Tuple[torch.Tensor, torch.Tensor]:
    head = batch.head
    node_e0 = atomic_energies_fn(batch.node_attrs)
    graph_e0s = scatter_sum(
        src=node_e0, index=batch.batch, dim=0, dim_size=batch.num_graphs
    )[torch.arange(batch.num_graphs), head]
    graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
    atom_energies = (batch.energy - graph_e0s) / graph_sizes  # {[n_graphs], }
    forces = batch.forces  # {[n_graphs*n_atoms,3], }

    return atom_energies, forces


def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader, rank=0) -> float:
    num_neighbors = []

    if rank == 0:
        data_iter = tqdm(data_loader)
    else:
        data_iter = data_loader
    
    for batch in data_iter:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()

def compute_avg_num_neighbors_per_elem(data_loader: torch.utils.data.DataLoader, rank=0) -> float:
    num_neighbors = []
    node_attrs = []

    if rank == 0:
        data_iter = tqdm(data_loader)
    else:
        data_iter = data_loader

    isolated_flag = False
    
    for batch in data_iter:
        iso_removed_edge_index, _, mask = remove_isolated_nodes(batch.edge_index, num_nodes=batch.node_attrs.size(0))

        _, receivers = iso_removed_edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        element_onehot_counts = counts.unsqueeze(-1) * batch.node_attrs[mask]
        num_neighbors.append(element_onehot_counts)
        node_attrs.append(batch.node_attrs)

    sum_num_neighbors_per_elem = torch.sum(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype()), dim=0, keepdim=False
    )
    num_elem = torch.sum(torch.cat(node_attrs, dim=0).type(torch.get_default_dtype()), dim=0)
    avg_num_neighbors_per_elem = sum_num_neighbors_per_elem / num_elem
    return to_numpy(avg_num_neighbors_per_elem) #to_numpy(torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype()))

def raw_num_neighbors_per_elem(data_loader: torch.utils.data.DataLoader, rank=0) -> float:
    num_neighbors = []
    node_attrs = []

    if rank == 0:
        data_iter = tqdm(data_loader)
    else:
        data_iter = data_loader

    isolated_flag = False
    
    for batch in data_iter:
        iso_removed_edge_index, _, mask = remove_isolated_nodes(batch.edge_index, num_nodes=batch.node_attrs.size(0))

        _, receivers = iso_removed_edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        element_onehot_counts = counts.unsqueeze(-1) * batch.node_attrs[mask]
        num_neighbors.append(element_onehot_counts)
        node_attrs.append(batch.node_attrs)

    raw = torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    num_elem = torch.sum(torch.cat(node_attrs, dim=0).type(torch.get_default_dtype()), dim=0)
    return to_numpy(raw), to_numpy(num_elem)

def compute_statistics(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float, float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []
    num_neighbors = []
    head_list = []
    head_batch = []

    for batch in data_loader:
        head = batch.head
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=0, dim_size=batch.num_graphs
        )[torch.arange(batch.num_graphs), head]
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }
        head_list.append(head)  # {[n_graphs], }
        head_batch.append(head[batch.batch])

        # for avg neighbour counting
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }
    head = torch.cat(head_list, dim=0)  # [total_n_graphs]
    head_batch = torch.cat(head_batch, dim=0)  # [total_n_graphs*n_atoms]

    mean = to_numpy(scatter_mean(src=atom_energies, index=head, dim=0).squeeze(-1))
    rms = to_numpy(
        torch.sqrt(
            scatter_mean(src=torch.square(forces), index=head_batch, dim=0).mean(-1)
        )
    )

    rms = _check_non_zero(rms)

    avg_num_neighbors = to_numpy(
        torch.mean(
            torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
        )
    )

    return avg_num_neighbors.item(), mean.item(), rms.item()


def compute_rms_dipoles(
    data_loader: torch.utils.data.DataLoader,
) -> Tuple[float, float]:
    dipoles_list = []
    for batch in data_loader:
        dipoles_list.append(batch.dipole)  # {[n_graphs,3], }

    dipoles = torch.cat(dipoles_list, dim=0)  # {[total_n_graphs,3], }
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(dipoles)))).item()
    rms = _check_non_zero(rms)
    return rms


def compute_fixed_charge_dipole(
    charges: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mu = positions * charges.unsqueeze(-1) / (1e-11 / c / e)  # [N_atoms,3]
    return scatter_sum(
        src=mu, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs
    )  # [N_graphs,3]
