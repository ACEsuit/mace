# Adapted from graph_longrange/realspace_electrostatics.py by Will Baldwin (MIT).
# https://github.com/WillBaldwin0/graph_electrostatics
# See LICENSE.md and this package's __init__.py for provenance and local changes.

import torch
from scipy.constants import pi

from .gto_utils import (
    GTOSelfInteractionBlock,
    get_Cl_sigma,
)
from .utils import FIELD_CONSTANT, scatter_sum


@torch.no_grad()
def batch_complete_graph_excluding_self_duplicates_vector(
    batch: torch.Tensor,
    N: int,
    sizes: tuple[int, ...] | None = None,
    dense: bool = False,
) -> torch.Tensor:
    """
    Duplicate each node N times, then for each graph build directed
    edges between every pair of duplicates *unless* they share the same
    original node ID.

    Args:
        batch (LongTensor): shape [M], graph ID of each original node.
        N (int): number of duplicates per node.
        sizes: the atoms of each graph, as host integers, when the caller has
            them. Then nothing is read off the device and every size the
            graph has is known when it is traced.
        dense: join every duplicate of the batch to every duplicate of
            another node, across graphs too, so the edges depend on the node
            count alone. A sum over them masks the pairs of two graphs.

    Returns:
        edge_index (LongTensor[2, E])

    The edges are ordered by graph, then by sender duplicate, then by receiver
    duplicate, each in ascending node order. Built without a loop over graphs,
    so without ``sizes`` the host waits on the device once, for the graph sizes.
    """
    device = batch.device
    batch = batch.long()
    num_nodes = batch.size(0)
    if num_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    if dense:
        return batch_complete_graph_excluding_self_duplicates_vector(
            torch.zeros_like(batch), N, (num_nodes,)
        )
    if sizes is None:
        sizes = tuple(torch.bincount(batch).tolist())
    graph_sizes = list(sizes)
    while graph_sizes and graph_sizes[-1] == 0:
        graph_sizes.pop()
    # D duplicates in a graph, each keeping every duplicate of another node.
    duplicates = [size * N for size in graph_sizes]
    widths = [count - N for count in duplicates]
    per_graph = [count * width for count, width in zip(duplicates, widths, strict=True)]
    num_edges = sum(per_graph)
    if num_edges == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # The duplicates of graph g in ascending order are a contiguous run of
    # this permutation, starting at its graph's offset.
    order = torch.argsort(batch.repeat_interleave(N), stable=True)
    graph_starts = torch.tensor(
        [0, *duplicates[:-1]], dtype=torch.long, device=device
    ).cumsum(0)
    edge_starts = torch.tensor(
        [0, *per_graph[:-1]], dtype=torch.long, device=device
    ).cumsum(0)
    width = torch.tensor(widths, dtype=torch.long, device=device)
    graph = torch.repeat_interleave(
        torch.arange(len(graph_sizes), device=device),
        torch.tensor(per_graph, dtype=torch.long, device=device),
        output_size=num_edges,
    )
    local = torch.arange(num_edges, device=device) - edge_starts[graph]
    edge_width = width[graph]
    sender = local // edge_width
    kept = local % edge_width
    # The k-th kept receiver skips the N duplicates of the sender's own node.
    own = (sender // N) * N
    receiver = kept + N * (kept >= own)
    start = graph_starts[graph]
    return torch.stack([order[start + sender], order[start + receiver]], dim=0)


def charges_energy_from_graph(
    charges,  # [n_atoms]
    positions,
    edge_index,
    batch,
    density_smearing_width,
    num_graphs: int | None = None,
    same_graph_only: bool = False,
):
    """
    Computes the energy of a collection of charges considering only specifed edges.
    normalization of the charges is multipoles. ``num_graphs`` sizes the result
    when the caller knows it, rather than reading it off ``batch``, and
    ``same_graph_only`` drops the edges joining two graphs.
    """
    sender, receiver = edge_index

    R_ij = positions[receiver] - positions[sender]  # [N_edges,3]
    d_ij = torch.linalg.norm(R_ij, dim=-1)  # [N_edges,1]
    smooth_reciprocal = torch.erf(d_ij * 0.5 / density_smearing_width) / (
        torch.abs(d_ij) + 1e-6
    )

    # charge part
    edge_energy = (
        0.5
        * FIELD_CONSTANT
        * smooth_reciprocal
        * charges[sender]
        * charges[receiver]
        / (4 * pi)
    )
    if same_graph_only:
        edge_energy = edge_energy * (batch[sender] == batch[receiver])
    # handle the case with no edges
    if edge_energy.numel() == 0:
        return torch.zeros(
            (batch.max() + 1 if num_graphs is None else num_graphs,),
            dtype=charges.dtype,
            device=charges.device,
        )
    node_energies = scatter_sum(
        src=edge_energy.squeeze(-1), index=receiver, dim=-1, dim_size=charges.shape[0]
    )
    return scatter_sum(
        src=node_energies, index=batch, dim=-1, dim_size=num_graphs
    )  # [n_graphs]


class RealSpaceFiniteDiffereneEnergy(torch.nn.Module):
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        include_self_interaction: bool = False,
        offset=0.02,
    ):
        if density_max_l > 1:
            raise ValueError(
                "RealSpaceFiniteDiffereneEnergy only supports l=0 and l=1."
            )

        super().__init__()
        self.density_max_l = density_max_l
        self.density_smearing_width = density_smearing_width
        self.include_self_interaction = include_self_interaction
        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            density_max_l,
            [density_smearing_width],
            "multipoles",
            "multipoles",
        )

        self.offset = offset
        self.register_buffer(
            "x", torch.tensor([offset, 0.0, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "y", torch.tensor([0.0, offset, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "z", torch.tensor([0.0, 0.0, offset], dtype=torch.get_default_dtype())
        )

    def energy_l0(
        self,
        source_feats: torch.Tensor,  # [n_node, (max_l_s+1)**2]
        positions: torch.Tensor,  # [n_node, 3]
        batch: torch.Tensor,  # [n_node]
        sizes: tuple[int, ...] | None = None,
        dense_pairs: bool = False,
    ) -> torch.Tensor:

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(
            batch, 1, sizes, dense_pairs
        )
        num_graphs = None if sizes is None else len(sizes)

        energy = charges_energy_from_graph(
            source_feats.squeeze(-1),
            positions,
            edge_index,
            batch,
            density_smearing_width=self.density_smearing_width,
            num_graphs=num_graphs,
            same_graph_only=dense_pairs,
        )

        # self interaction
        if self.include_self_interaction:
            self_fields = self.self_interaction(source_feats)  # [n_node, (l+1)^2]
            node_energies = torch.einsum("nb,nb->n", source_feats, self_fields)
            self_energy = scatter_sum(
                src=node_energies, index=batch, dim=-1, dim_size=num_graphs
            )
            energy += self_energy * 0.5

        return energy

    def energy_l1(
        self,
        source_feats: torch.Tensor,  # [n_node, (max_l_s+1)**2]
        positions: torch.Tensor,  # [n_node, 3]
        batch: torch.Tensor,  # [n_node]
        sizes: tuple[int, ...] | None = None,
        dense_pairs: bool = False,
    ) -> torch.Tensor:
        extended_positions = positions.repeat_interleave(4, dim=0)
        extended_positions[1::4] += self.x
        extended_positions[2::4] += self.y
        extended_positions[3::4] += self.z

        extended_batch = batch.repeat_interleave(4)
        charges = torch.zeros_like(extended_positions[:, 0])

        charges[1::4] = source_feats[:, 3] / self.offset
        charges[2::4] = source_feats[:, 1] / self.offset
        charges[3::4] = source_feats[:, 2] / self.offset
        charges[0::4] = source_feats[:, 0] - (
            charges[1::4] + charges[2::4] + charges[3::4]
        )

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(
            batch, 4, sizes, dense_pairs
        )
        num_graphs = None if sizes is None else len(sizes)

        energy = charges_energy_from_graph(
            charges,
            extended_positions,
            edge_index,
            extended_batch,
            density_smearing_width=self.density_smearing_width,
            num_graphs=num_graphs,
            same_graph_only=dense_pairs,
        )

        # self interaction
        if self.include_self_interaction:
            self_fields = self.self_interaction(source_feats)  # [n_node, (l+1)^2]
            node_energies = torch.einsum("nb,nb->n", source_feats, self_fields)
            self_energy = scatter_sum(
                src=node_energies, index=batch, dim=-1, dim_size=num_graphs
            )
            energy += self_energy * 0.5

        return energy

    def forward(
        self,
        source_feats: torch.Tensor,  # [n_node, (max_l_s+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
        sizes: tuple[int, ...] | None = None,
        dense_pairs: bool = False,
    ) -> torch.Tensor:
        if self.density_max_l == 0:
            return self.energy_l0(source_feats, positions, batch, sizes, dense_pairs)
        else:
            return self.energy_l1(source_feats, positions, batch, sizes, dense_pairs)


def charges_features_from_graph(
    charges,  # [n_atoms]
    positions,
    edge_index,
    batch,
    total_width_factors,  # [1, n_radial]
    same_graph_only: bool = False,
):
    """
    Computes the features from a collection of charges, on set of scalar features,
    considering only specified edges, or only those within one graph with
    ``same_graph_only``.
    normalization of the charges is multipoles.
    """
    num_nodes = positions.shape[0]
    sender, receiver = edge_index
    R_ij = positions[sender] - positions[receiver]  # [N_edges,3]
    d_ij = torch.norm(R_ij, dim=-1, keepdim=True)  # [N_edges,1]
    smooth_reciprocal = torch.erf(0.5 * d_ij / total_width_factors) / (d_ij + 1e-6)
    if same_graph_only:
        smooth_reciprocal = smooth_reciprocal * (
            batch[sender] == batch[receiver]
        ).unsqueeze(-1)

    features = scatter_sum(
        charges[sender].unsqueeze(-1) * smooth_reciprocal,
        receiver,
        dim=0,
        dim_size=num_nodes,
    )  # [n_nodes, n_radial]

    features = FIELD_CONSTANT * features / (4 * pi)
    return features


class RealSpaceFiniteDifferenceElectrostaticFeatures(torch.nn.Module):
    """Computes field features for L=0,1 charges and features.
    vector charges and features are represented by displaced scalars."""

    l0_factors: torch.Tensor
    l1_factors: torch.Tensor
    total_width_factors: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        projection_max_l: int,
        projection_smearing_widths: list[float],
        include_self_interaction=False,
        integral_normalization="receiver",
        offset: float = 0.1,
    ):
        super().__init__()

        self.density_max_l = density_max_l
        self.projection_max_l = projection_max_l
        self.include_self_interaction = include_self_interaction
        self.density_smearing_width = density_smearing_width
        self.projection_smearing_widths = projection_smearing_widths
        self.num_radial = len(projection_smearing_widths)

        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            projection_max_l,
            projection_smearing_widths,
            "multipoles",
            integral_normalization,
        )

        projection_smearing_widths_tensor = torch.tensor(
            projection_smearing_widths, dtype=torch.get_default_dtype()
        )
        total_width_factors = torch.pow(
            (density_smearing_width**2 + projection_smearing_widths_tensor**2) / 2, 0.5
        )
        self.register_buffer("total_width_factors", total_width_factors)

        self.offset = offset
        self.register_buffer(
            "x", torch.tensor([offset, 0.0, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "y", torch.tensor([0.0, offset, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "z", torch.tensor([0.0, 0.0, offset], dtype=torch.get_default_dtype())
        )

        l0_factors = [
            get_Cl_sigma(0, sigma, normalize=integral_normalization)
            / get_Cl_sigma(0, sigma, normalize="multipoles")
            for sigma in projection_smearing_widths
        ]
        self.register_buffer(
            "l0_factors", torch.tensor(l0_factors, dtype=torch.get_default_dtype())
        )
        l1_factors = [
            3**0.5
            * sigma**2
            * (
                get_Cl_sigma(1, sigma, normalize=integral_normalization)
                / get_Cl_sigma(0, sigma, normalize="multipoles")
            )
            / self.offset
            for sigma in projection_smearing_widths
        ]
        self.register_buffer(
            "l1_factors", torch.tensor(l1_factors, dtype=torch.get_default_dtype())
        )

    def call_density_0_feats_0(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (max_l_s+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
        sizes: tuple[int, ...] | None = None,
        dense_pairs: bool = False,
    ) -> torch.Tensor:
        edge_long_index = batch_complete_graph_excluding_self_duplicates_vector(
            batch, 1, sizes, dense_pairs
        )
        feats = charges_features_from_graph(
            charges=source_feats[:, 0],
            positions=positions,
            edge_index=edge_long_index,
            batch=batch,
            total_width_factors=self.total_width_factors.unsqueeze(0),
            same_graph_only=dense_pairs,
        )  # [n_atoms, n_radial]
        return self.l0_factors * feats

    def call_density_1_feats_1(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (max_l_s+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
        sizes: tuple[int, ...] | None = None,
        dense_pairs: bool = False,
    ) -> torch.Tensor:
        extended_positions = positions.repeat_interleave(4, dim=0)
        extended_positions[1::4] += self.x
        extended_positions[2::4] += self.y
        extended_positions[3::4] += self.z

        extended_batch = batch.repeat_interleave(4)
        charges = torch.zeros_like(extended_positions[:, 0])

        charges[1::4] = source_feats[:, 3] / self.offset
        charges[2::4] = source_feats[:, 1] / self.offset
        charges[3::4] = source_feats[:, 2] / self.offset
        charges[0::4] = source_feats[:, 0] - (
            charges[1::4] + charges[2::4] + charges[3::4]
        )

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(
            batch, 4, sizes, dense_pairs
        )

        scalar_features = charges_features_from_graph(
            charges=charges,
            positions=extended_positions,
            edge_index=edge_index,
            batch=extended_batch,
            total_width_factors=self.total_width_factors.unsqueeze(0),
            same_graph_only=dense_pairs,
        )  # [all_nodes, num_radial]

        all_features = source_feats.new_zeros((batch.size(0), 4 * self.num_radial))

        all_features[:, : self.num_radial] = self.l0_factors * scalar_features[0::4]
        all_features[:, self.num_radial :: 3] = self.l1_factors * (
            scalar_features[2::4] - scalar_features[0::4]
        )
        all_features[:, self.num_radial + 1 :: 3] = self.l1_factors * (
            scalar_features[3::4] - scalar_features[0::4]
        )
        all_features[:, self.num_radial + 2 :: 3] = self.l1_factors * (
            scalar_features[1::4] - scalar_features[0::4]
        )

        return all_features

    def forward(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (max_l_s+1)**2]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        sizes: tuple[int, ...] | None = None,
        dense_pairs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if self.density_max_l == 0 and self.projection_max_l == 0:
            features = self.call_density_0_feats_0(
                source_feats, node_positions, batch, sizes, dense_pairs
            )
        elif self.density_max_l == 1 and self.projection_max_l == 0:
            all_feats = self.call_density_1_feats_1(
                source_feats, node_positions, batch, sizes, dense_pairs
            )
            features = all_feats[:, : self.num_radial]
        elif self.density_max_l == 0 and self.projection_max_l == 1:
            padded_source_feats = torch.zeros(
                source_feats.shape[0],
                4,
                dtype=source_feats.dtype,
                device=source_feats.device,
            )
            padded_source_feats[:, 0] = source_feats[:, 0]
            features = self.call_density_1_feats_1(
                padded_source_feats, node_positions, batch, sizes, dense_pairs
            )
        else:
            features = self.call_density_1_feats_1(
                source_feats, node_positions, batch, sizes, dense_pairs
            )

        self_interaction_terms = self.self_interaction(source_feats)
        if self.include_self_interaction:
            features += self_interaction_terms

        return features, self_interaction_terms, None
