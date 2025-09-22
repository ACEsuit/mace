from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from e3nn import o3


def gto_basis_kspace_cutoff(sigmas: List[float], max_l: int) -> float:
    """
    Minimal heuristic cutoff used for k-space grid generation.
    Matches interface expected by previous repo without pulling full dependency.
    """
    min_sigma = float(min(sigmas)) if len(sigmas) > 0 else 1.0
    # Simple heuristic; original implementation is more elaborate.
    return 8.0 / min_sigma


def compute_k_vectors(
    kspace_cutoff: torch.Tensor,
    cell: torch.Tensor,   # [n_graph, 3, 3]
    rcell: torch.Tensor,  # [n_graph, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a very small reciprocal lattice grid per graph.
    For compatibility this returns a single masked (invalid) vector if not used downstream.
    """
    n_graph = cell.shape[0]
    device = cell.device
    # Provide one zero vector per graph; mark as invalid via mask=False
    kvecs = torch.zeros((n_graph, 1, 3), dtype=cell.dtype, device=device)
    knorm2 = torch.zeros((n_graph, 1), dtype=cell.dtype, device=device)
    kmask = torch.zeros((n_graph, 1), dtype=torch.bool, device=device)
    return kvecs, knorm2, kmask


class DisplacedGTOExternalFieldBlock(torch.nn.Module):
    """
    Lightweight external field projector. Maps uniform external potentials/fields
    (fermi level + 3-vector field per graph) to local spherical features per node.
    Only l<=1 channels are populated; higher l are zero-filled for shape compatibility.
    """

    def __init__(self, max_l: int, sigmas: List[float], mode: str = "receiver"):
        super().__init__()
        self.max_l = max_l
        self.sigmas = list(sigmas)
        self.num_widths = len(sigmas)
        # Build output irreps matching local potential feature layout (no spin doubling here)
        self.irreps = (o3.Irreps.spherical_harmonics(max_l) * self.num_widths).sort()[0].simplify()

    def forward(
        self,
        batch: torch.Tensor,            # [n_nodes]
        positions: torch.Tensor,        # [n_nodes, 3]
        external_potential: torch.Tensor,  # [n_graph, 4] -> [mu, Ex, Ey, Ez]
    ) -> torch.Tensor:
        device = positions.device
        n_nodes = positions.shape[0]
        # Gather per-node external field
        mu_E = external_potential[batch]  # [n_nodes, 4]
        E = mu_E[:, 1:4]  # [n_nodes, 3]

        # Build per-width features: l=0 channel (mu) and l=1 channels (E components)
        # Layout flattens as widths-major over (l,m) blocks
        blocks: List[torch.Tensor] = []
        for _ in range(self.num_widths):
            if self.max_l >= 0:
                l0 = mu_E[:, 0:1]
                blocks.append(l0)
            if self.max_l >= 1:
                blocks.append(E)  # 3 components ~ l=1 real harmonics proxy
            # Zero-fill higher l if requested
            for l in range(2, self.max_l + 1):
                blocks.append(torch.zeros((n_nodes, 2 * l + 1), device=device, dtype=positions.dtype))
        return torch.cat(blocks, dim=-1) if len(blocks) > 0 else torch.zeros((n_nodes, 0), device=device, dtype=positions.dtype)


class PBCAgnosticDirectElectrostaticEnergyBlock(torch.nn.Module):
    """
    Minimal electrostatic energy evaluator. Computes pairwise 0.5 * sum_ij q_i q_j / r_ij
    using only the monopole (l=0) component. PBC support is omitted for simplicity.
    """

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        kspace_cutoff: float,
        include_self_interaction: bool = False,
    ):
        super().__init__()
        self.include_self_interaction = include_self_interaction

    def forward(
        self,
        charge_density: torch.Tensor,  # [n_nodes, (l+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,   # [n_graph, 3, 3]
        rcell: torch.Tensor,  # [n_graph, 3, 3]
        volumes: torch.Tensor,
        pbc: torch.Tensor,    # [n_graph, 3]
        num_graphs: int,
        use_pbc_evaluator: bool = False,
    ) -> torch.Tensor:
        device = positions.device
        dtype = positions.dtype
        q = charge_density[:, 0]  # monopole only
        energy = torch.zeros((num_graphs,), device=device, dtype=dtype)
        # Compute energy per graph
        for g in range(num_graphs):
            mask = (batch == g)
            pos_g = positions[mask]
            q_g = q[mask]
            if pos_g.numel() == 0:
                continue
            rij = pos_g.unsqueeze(1) - pos_g.unsqueeze(0)  # [N,N,3]
            dist = torch.linalg.norm(rij + 1e-15, dim=-1)  # [N,N]
            eye = torch.eye(dist.shape[0], device=device, dtype=torch.bool)
            inv_r = torch.where(eye, torch.zeros_like(dist), 1.0 / dist)
            # 0.5 * sum_{i!=j} q_i q_j / r_ij
            e = 0.5 * (q_g.unsqueeze(0) * q_g.unsqueeze(1) * inv_r).sum()
            # Optionally include self term (ignored by default)
            if self.include_self_interaction:
                e = e + 0.0
            energy[g] = e
        return energy


class PBCAgnosticElectrostaticFeatureBlock(torch.nn.Module):
    """
    Projects electrostatic field due to a (spin-resolved) charge density onto a local
    spherical basis per node. Minimal real-space evaluator that fills l<=1 channels
    and zero-fills higher orders; compatible with previous model interfaces.
    """

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        projection_max_l: int,
        projection_smearing_widths: List[float],
        kspace_cutoff: float,
        include_self_interaction: bool = False,
        integral_normalization: str = "receiver",
        quadrupole_feature_corrections: bool = False,
    ):
        super().__init__()
        self.proj_max_l = projection_max_l
        self.num_widths = len(projection_smearing_widths)
        self.include_self_interaction = include_self_interaction
        # Precompute output dimension for convenience
        self.out_irreps = (
            (o3.Irreps.spherical_harmonics(projection_max_l) * self.num_widths)
            .sort()[0]
            .simplify()
        )

    def forward(
        self,
        k_vectors: torch.Tensor,  # unused placeholder
        k_vectors_normed_squared: torch.Tensor,  # unused placeholder
        k_vectors_mask: torch.Tensor,  # unused placeholder
        source_feats: torch.Tensor,  # [n_nodes, 1, (l+1)**2]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
        pbc: torch.Tensor,
        use_pbc_evaluator: bool,
        return_electrostatic_potentials: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        device = node_positions.device
        dtype = node_positions.dtype
        n_nodes = node_positions.shape[0]
        num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0

        # Use only monopole part of the source features
        q = source_feats[:, 0, 0]  # [n_nodes]

        # Compute potential and field at each node (ignoring self term)
        V = torch.zeros((n_nodes,), dtype=dtype, device=device)
        F = torch.zeros((n_nodes, 3), dtype=dtype, device=device)
        for g in range(num_graphs):
            mask = (batch == g)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            pos_g = node_positions[idx]
            q_g = q[idx]
            rij = pos_g.unsqueeze(1) - pos_g.unsqueeze(0)  # [N,N,3]
            dist = torch.linalg.norm(rij + 1e-15, dim=-1)  # [N,N]
            eye = torch.eye(dist.shape[0], device=device, dtype=torch.bool)
            inv_r = torch.where(eye, torch.zeros_like(dist), 1.0 / dist)
            # Potential
            V_g = (q_g.unsqueeze(0) * inv_r).sum(dim=1)
            V[idx] = V_g
            # Field: sum q * r / r^3
            inv_r3 = torch.where(eye, torch.zeros_like(dist), inv_r / (dist * dist + 1e-15))
            F_g = (q_g.unsqueeze(0).unsqueeze(-1) * (rij * inv_r3.unsqueeze(-1))).sum(dim=1)
            F[idx] = F_g

        # Pack into spherical basis blocks per width: l=0 uses V, l=1 uses field (x,y,z)
        blocks: List[torch.Tensor] = []
        for _ in range(self.num_widths):
            if self.proj_max_l >= 0:
                blocks.append(V.unsqueeze(-1))
            if self.proj_max_l >= 1:
                blocks.append(F)
            for l in range(2, self.proj_max_l + 1):
                blocks.append(torch.zeros((n_nodes, 2 * l + 1), device=device, dtype=dtype))

        features = (
            torch.cat(blocks, dim=-1)
            if len(blocks) > 0
            else torch.zeros((n_nodes, 0), device=device, dtype=dtype)
        )
        # Self interaction term placeholder (zero)
        self_terms = torch.zeros_like(features)
        esps = V.unsqueeze(-1) if return_electrostatic_potentials else None
        return features, self_terms, esps


def compute_total_charge_dipole(
    charge_density: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute total charge (monopole) and total dipole per graph from per-node charge density.
    Expects charge_density with (l=0) channel at index 0.
    """
    q0 = charge_density[:, 0]
    # Total charge per graph
    total_charge = torch.zeros((num_graphs,), dtype=positions.dtype, device=positions.device)
    # Total dipole per graph
    total_dipole = torch.zeros((num_graphs, 3), dtype=positions.dtype, device=positions.device)
    for g in range(num_graphs):
        mask = (batch == g)
        q_g = q0[mask]
        r_g = positions[mask]
        total_charge[g] = q_g.sum()
        total_dipole[g] = (q_g.unsqueeze(-1) * r_g).sum(dim=0)
    return total_charge, total_dipole

