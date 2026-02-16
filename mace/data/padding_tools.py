"""Helpers for fixed-shape graph padding."""

import torch

from mace.tools import torch_geometric


def _zeros_with_size0(tensor: torch.Tensor, size0: int) -> torch.Tensor:
    if tensor.dim() == 0:
        return torch.zeros_like(tensor)
    shape = list(tensor.shape)
    shape[0] = size0
    return torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)


_GRAPH_LEVEL_TENSOR_KEYS = {
    "cell",
    "pbc",
    "stress",
    "virials",
    "dipole",
    "polarizability",
    "polarizability_sh",
    "polarizability_weight",
    "dipole_weight",
}


def build_fake_padding_graph(
    reference_graph, num_atoms: int, num_edges: int, r_max: float
):
    """Build a fake graph used to pad a batch to fixed atom/edge counts."""
    if num_atoms <= 0:
        raise ValueError("Padding graph requires at least one fake atom.")

    real_num_atoms = int(reference_graph["node_attrs"].shape[0])
    real_num_edges = int(reference_graph["edge_index"].shape[1])

    fake_graph = torch_geometric.data.Data.from_dict(
        {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in reference_graph.to_dict().items()
        }
    )

    # Pad explicit mandatory node fields first.
    for key in ("node_attrs", "positions", "forces", "charges"):
        if key in fake_graph and torch.is_tensor(fake_graph[key]) and fake_graph[key].dim() > 0:
            fake_graph[key] = _zeros_with_size0(fake_graph[key], num_atoms)
    fake_graph["node_attrs"][:, 0] = 1.0

    # Pad other per-node / per-edge tensor fields generically so new AtomicData keys
    # usually don't require changing this helper.
    for key, value in fake_graph.to_dict().items():
        if (
            key in {"edge_index", "node_attrs", "positions", "forces", "charges", "shifts", "unit_shifts"}
            or key in _GRAPH_LEVEL_TENSOR_KEYS
            or not torch.is_tensor(value)
            or value.dim() == 0
        ):
            continue
        if value.shape[0] == real_num_edges and (
            value.shape[0] != real_num_atoms or "edge" in key
        ):
            fake_graph[key] = _zeros_with_size0(value, num_edges)
        elif value.shape[0] == real_num_atoms:
            fake_graph[key] = _zeros_with_size0(value, num_atoms)

    edge_index = torch.zeros(
        (2, num_edges),
        dtype=reference_graph["edge_index"].dtype,
        device=reference_graph["edge_index"].device,
    )
    if num_edges > 0:
        edge_ids = torch.arange(
            num_edges, dtype=edge_index.dtype, device=edge_index.device
        )
        edge_index[0] = torch.remainder(edge_ids, num_atoms)
        edge_index[1] = torch.remainder(edge_ids + 1, num_atoms)
    fake_graph["edge_index"] = edge_index

    for key in ("shifts", "unit_shifts"):
        if key in fake_graph and torch.is_tensor(fake_graph[key]):
            fake_graph[key] = _zeros_with_size0(fake_graph[key], num_edges)

    cell_scale = max(float(r_max) * 2.0, 1.0)
    cell_ref = reference_graph["cell"]
    fake_graph["cell"] = torch.eye(
        3, dtype=cell_ref.dtype, device=cell_ref.device
    ) * cell_scale
    if "pbc" in fake_graph and torch.is_tensor(fake_graph["pbc"]):
        fake_graph["pbc"] = torch.zeros_like(fake_graph["pbc"], dtype=torch.bool)

    if (
        num_edges > 0
        and "unit_shifts" in fake_graph
        and torch.is_tensor(fake_graph["unit_shifts"])
        and "shifts" in fake_graph
        and torch.is_tensor(fake_graph["shifts"])
    ):
        fake_graph["unit_shifts"][:, 0] = 1.0
        fake_graph["shifts"][:, 0] = cell_scale

    if "total_charge" in fake_graph and torch.is_tensor(fake_graph["total_charge"]):
        fake_graph["total_charge"] = torch.zeros_like(fake_graph["total_charge"])
    if "total_spin" in fake_graph and torch.is_tensor(fake_graph["total_spin"]):
        fake_graph["total_spin"] = torch.ones_like(fake_graph["total_spin"])

    fake_graph.num_nodes = num_atoms
    return fake_graph
