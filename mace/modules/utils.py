###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from scipy.constants import c, e

from mace.tools import to_numpy
from mace.tools.scatter import scatter_mean, scatter_std, scatter_sum
from mace.tools.torch_geometric.batch import Batch

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
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
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


@torch.jit.unused
def compute_hessians_vmap(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    forces_flatten = forces.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            -1 * forces_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements).to(forces.device)
    try:
        chunk_size = 1 if num_elements < 64 else 16
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
            I_N
        )[0]
    except RuntimeError:
        gradient = compute_hessians_loop(forces, positions)
    if gradient is None:
        return torch.zeros((positions.shape[0], forces.shape[0], 3, 3))
    return gradient


@torch.jit.unused
def compute_hessians_loop(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    hessian = []
    for grad_elem in forces.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-1 * grad_elem],
            inputs=[positions],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hess_row = hess_row.detach()  # this makes it very slow? but needs less memory
        if hess_row is None:
            hessian.append(torch.zeros_like(positions))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)
    return hessian


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    displacement: Optional[torch.Tensor],
    vectors: Optional[torch.Tensor] = None,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
    compute_hessian: bool = False,
    compute_edge_forces: bool = False,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if (compute_virials or compute_stress) and displacement is not None:
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=(training or compute_hessian or compute_edge_forces),
        )
    elif compute_force:
        forces, virials, stress = (
            compute_forces(
                energy=energy,
                positions=positions,
                training=(training or compute_hessian or compute_edge_forces),
            ),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)
    if compute_hessian:
        assert forces is not None, "Forces must be computed to get the hessian"
        hessian = compute_hessians_vmap(forces, positions)
    else:
        hessian = None
    if compute_edge_forces and vectors is not None:
        edge_forces = compute_forces(
            energy=energy,
            positions=vectors,
            training=(training or compute_hessian),
        )
        if edge_forces is not None:
            edge_forces = -1 * edge_forces  # Match LAMMPS sign convention
    else:
        edge_forces = None
    return forces, virials, stress, hessian, edge_forces


def get_atomic_virials_stresses(
    edge_forces: torch.Tensor,  # [n_edges, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    vectors: torch.Tensor,  # [n_edges, 3]
    num_atoms: int,
    batch: torch.Tensor,
    cell: torch.Tensor,  # [n_graphs, 3, 3]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute atomic virials and optionally atomic stresses from edge forces and vectors.
    From pobo95 PR #528.
    Returns:
        Tuple of:
            - Atomic virials [num_atoms, 3, 3]
            - Atomic stresses [num_atoms, 3, 3] (None if not computed)
    """
    edge_virial = torch.einsum("zi,zj->zij", edge_forces, vectors)
    atom_virial_sender = scatter_sum(
        src=edge_virial, index=edge_index[0], dim=0, dim_size=num_atoms
    )
    atom_virial_receiver = scatter_sum(
        src=edge_virial, index=edge_index[1], dim=0, dim_size=num_atoms
    )
    atom_virial = (atom_virial_sender + atom_virial_receiver) / 2
    atom_virial = (atom_virial + atom_virial.transpose(-1, -2)) / 2
    atom_stress = None
    cell = cell.view(-1, 3, 3)
    volume = torch.linalg.det(cell).abs().unsqueeze(-1)
    atom_volume = volume[batch].view(-1, 1, 1)
    atom_stress = atom_virial / atom_volume
    atom_stress = torch.where(
        torch.abs(atom_stress) < 1e10, atom_stress, torch.zeros_like(atom_stress)
    )
    return -1 * atom_virial, atom_stress


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
    out.append(x[:, :num_features])
    for i in range(1, num_layers):
        out.append(
            x[
                :,
                i
                * (l_max + 1) ** 2
                * num_features : (i * (l_max + 1) ** 2 + 1)
                * num_features,
            ]
        )
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


def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []
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
        head_list.append(head)
        head_batch.append(head[batch.batch])

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }
    head = torch.cat(head_list, dim=0)  # [total_n_graphs]
    head_batch = torch.cat(head_batch, dim=0)  # [total_n_graphs]

    # mean = to_numpy(torch.mean(atom_energies)).item()
    # rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()
    mean = to_numpy(scatter_mean(src=atom_energies, index=head, dim=0).squeeze(-1))
    rms = to_numpy(
        torch.sqrt(
            scatter_mean(src=torch.square(forces), index=head_batch, dim=0).mean(-1)
        )
    )
    rms = _check_non_zero(rms)

    return mean, rms


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
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }
    head = torch.cat(head_list, dim=0)  # [total_n_graphs]
    head_batch = torch.cat(head_batch, dim=0)  # [total_n_graphs]

    # mean = to_numpy(torch.mean(atom_energies)).item()
    mean = to_numpy(scatter_mean(src=atom_energies, index=head, dim=0).squeeze(-1))
    rms = to_numpy(
        torch.sqrt(
            scatter_mean(src=torch.square(forces), index=head_batch, dim=0).mean(-1)
        )
    )

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )

    return to_numpy(avg_num_neighbors).item(), mean, rms


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


def compute_fixed_charge_dipole_polar(
    charges: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mu = positions * charges.unsqueeze(
        -1
    )  # / (1e-11 / c / e)  # [N_atoms,3] = 0.20819...
    return scatter_sum(src=mu, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs)


@torch.jit.ignore
def compute_dielectric_gradients(
    dielectric: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.tensor, torch.tensor]:
    dielectric_flatten = dielectric.view(-1)

    def get_vjp(v):
        return torch.autograd.grad(
            dielectric_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )

    try:
        I_N = torch.eye(dielectric.shape[-1]).to(dielectric.device)
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0)(I_N)[0]
    except RuntimeError:
        gradient = compute_dielectric_gradients_loop(dielectric, positions).detach()
    if gradient is None:
        return torch.zeros((positions.shape[0], dielectric.shape[-1], 3))
    return gradient


def compute_dielectric_gradients_loop(
    dielectric: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    gradients = []
    for i in range(dielectric.shape[-1]):
        grad_elem = dielectric[:, i]
        hess_row = torch.autograd.grad(
            grad_elem,
            positions,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        gradients.append(hess_row)
    gradients = torch.stack(gradients)
    return gradients


class InteractionKwargs(NamedTuple):
    lammps_class: Optional[torch.Tensor]
    lammps_natoms: Tuple[int, int] = (0, 0)


class GraphContext(NamedTuple):
    is_lammps: bool
    num_graphs: int
    num_atoms_arange: torch.Tensor
    displacement: Optional[torch.Tensor]
    positions: torch.Tensor
    vectors: torch.Tensor
    lengths: torch.Tensor
    cell: torch.Tensor
    node_heads: torch.Tensor
    interaction_kwargs: InteractionKwargs


def prepare_graph(
    data: Dict[str, torch.Tensor],
    compute_virials: bool = False,
    compute_stress: bool = False,
    compute_displacement: bool = False,
    lammps_mliap: bool = False,
) -> GraphContext:
    if torch.jit.is_scripting():
        lammps_mliap = False

    node_heads = (
        data["head"][data["batch"]]
        if "head" in data
        else torch.zeros_like(data["batch"])
    )

    if lammps_mliap:
        n_real, n_total = data["natoms"][0], data["natoms"][1]
        num_graphs = 2
        num_atoms_arange = torch.arange(n_real, device=data["node_attrs"].device)
        displacement = None
        positions = torch.zeros(
            (int(n_real), 3),
            dtype=data["vectors"].dtype,
            device=data["vectors"].device,
        )
        cell = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["vectors"].dtype,
            device=data["vectors"].device,
        )
        vectors = data["vectors"].requires_grad_(True)
        lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        ikw = InteractionKwargs(data["lammps_class"], (n_real, n_total))
    else:
        data["positions"].requires_grad_(True)
        positions = data["positions"]
        cell = data["cell"]
        num_atoms_arange = torch.arange(positions.shape[0], device=positions.device)
        num_graphs = int(data["ptr"].numel() - 1)
        displacement = torch.zeros(
            (num_graphs, 3, 3), dtype=positions.dtype, device=positions.device
        )
        if compute_virials or compute_stress or compute_displacement:
            p, s, displacement = get_symmetric_displacement(
                positions=positions,
                unit_shifts=data["unit_shifts"],
                cell=cell,
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )
            data["positions"], data["shifts"] = p, s
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        ikw = InteractionKwargs(None, (0, 0))

    return GraphContext(
        is_lammps=lammps_mliap,
        num_graphs=num_graphs,
        num_atoms_arange=num_atoms_arange,
        displacement=displacement,
        positions=positions,
        vectors=vectors,
        lengths=lengths,
        cell=cell,
        node_heads=node_heads,
        interaction_kwargs=ikw,
    )
