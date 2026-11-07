# Adapted from graph_longrange/kspace.py by Will Baldwin (MIT).
# https://github.com/WillBaldwin0/graph_electrostatics
# See LICENSE.md and this package's __init__.py for provenance and local changes.

import torch
from scipy.constants import pi


def compute_k_vectors_flat(
    cutoff: float,
    cell_vectors: torch.Tensor,  # [n_graphs, 3, 3]
    r_cell_vectors: torch.Tensor,  # [n_graphs, 3, 3]
):
    """
    Given a batch of cells, computes graph-grouped flattened k-vectors: the
    selection of :func:`k_vector_selection` evaluated by
    :func:`k_vectors_from_selection`.

    returns:
        - kvectors
            flattened tensor [n_k_total, 3], grouped by graph id
        - k_vectors_normed_squared
            flattened tensor [n_k_total] of squared norms
        - kvector_batch
            tensor [n_k_total] (long) giving graph id for each k-vector
        - k0_mask
            float tensor [n_k_total], 1.0 where k=(0,0,0), else 0.0
    """
    coefficients, kept = k_vector_selection(cutoff, cell_vectors, r_cell_vectors)
    return k_vectors_from_selection(coefficients, kept, r_cell_vectors)


@torch.no_grad()
def k_vector_selection(
    cutoff: float,
    cell_vectors: torch.Tensor,  # [n_graphs, 3, 3]
    r_cell_vectors: torch.Tensor,  # [n_graphs, 3, 3]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Which k-vectors of a batch of cells are inside the cutoff.
    The search-space heuristic follows the previous implementation.

    note that this function is meant to run on gpu, so it cannot do the matrix
    inverse internally

        - cutoff
            reciporical space cutoff
        - cell_vectors [n_graph, 3, 3]
            where the -2 dimension indexes the different lattice
            vectors, and the -1 index is the spatial dimension.

    returns:
        - coefficients
            [n_kvec, 3], in the dtype of the reciprocal cell: the integer
            coefficients of every candidate, a superset shared by all graphs
            whose first row is k = 0
        - kept
            long tensor [n_k_total]: the rows, in the graph-major flattening
            of [n_graphs, n_kvec], of the candidates inside the cutoff

    Both depend on the cells alone, and the host waits on the device for
    them, so a caller with a fixed cell computes them once.
    """
    device = cell_vectors.device

    # compute normed real lattice vectors
    norms = torch.norm(cell_vectors, dim=-1)  # [n_graph, 3]
    normed_lattice_vectors = cell_vectors / norms.unsqueeze(-1)  # [n_graph, 3, 3]

    dot_products = torch.einsum(
        "bij,bij->bi", r_cell_vectors, normed_lattice_vectors
    )  # [n_graph, 3]
    max_ns = cutoff * torch.pow(dot_products, -1)  # [n_graph, 3]
    max_ns = torch.ceil(max_ns).type(torch.int64)

    max_max_ns = torch.max(max_ns, dim=0).values  # [3]
    n1max, n2max, n3max = max_max_ns.tolist()

    # make a single superset of all coeficients
    origin = torch.cartesian_prod(
        torch.arange(0, 1, 1, device=device),
        torch.arange(0, 1, 1, device=device),
        torch.arange(0, 1, 1, device=device),
    ).type(torch.float32)

    open_half_sphere = torch.cartesian_prod(
        torch.arange(1, n1max, 1, device=device),
        torch.arange(-n2max, n2max, 1, device=device),
        torch.arange(-n3max, n3max, 1, device=device),
    ).type(torch.float32)

    open_half_plane = torch.cartesian_prod(
        torch.arange(0, 1, 1, device=device),
        torch.arange(1, n2max, 1, device=device),
        torch.arange(-n3max, n3max, 1, device=device),
    ).type(torch.float32)

    open_half_line = torch.cartesian_prod(
        torch.arange(0, 1, 1, device=device),
        torch.arange(0, 1, 1, device=device),
        torch.arange(1, n3max, 1, device=device),
    ).type(torch.float32)

    kvecs = torch.cat(
        (origin, open_half_line, open_half_plane, open_half_sphere), dim=-2
    )
    kvecs = kvecs.to(r_cell_vectors.dtype)

    # produce all kvectors for all graphs with the largest superset of coeficients
    k_vectors = torch.einsum(
        "ni,bij->bnj", kvecs, r_cell_vectors
    )  # [n_graphs, n_kvec, 3]

    # find a mask of all of them which are too large
    k_vectors_normed_squared = torch.einsum(
        "bni,bni->bn", k_vectors, k_vectors
    )  # [n_graphs, n_kvec]
    mask = k_vectors_normed_squared.le(cutoff * cutoff)  # [n_graphs, n_kvec]

    # One mask over the graph-major flattening keeps the vectors grouped by
    # graph.
    return kvecs, mask.reshape(-1).nonzero().squeeze(-1)


def k_vectors_from_selection(
    coefficients: torch.Tensor,  # [n_kvec, 3]
    kept: torch.Tensor,  # [n_k_total]
    r_cell_vectors: torch.Tensor,  # [n_graphs, 3, 3]
):
    """
    The k-vectors :func:`k_vector_selection` chose, from the reciprocal cells.

    Differentiable in the cells, and with every shape fixed by the selection,
    so nothing here is read off the device. Returns what
    :func:`compute_k_vectors_flat` does.
    """
    k_vectors = torch.einsum(
        "ni,bij->bnj", coefficients, r_cell_vectors
    )  # [n_graphs, n_kvec, 3]
    k_vectors_normed_squared = torch.einsum(
        "bni,bni->bn", k_vectors, k_vectors
    )  # [n_graphs, n_kvec]
    # k = 0 is always kept and is each graph's first vector, so the k0 mask
    # marks the kept vectors at local index 0.
    num_kvec = coefficients.shape[0]
    k_vectors_flat = k_vectors.reshape(-1, 3)[kept]  # [n_k_total, 3]
    k_norm2_flat = k_vectors_normed_squared.reshape(-1)[kept]  # [n_k_total]
    k_batch_flat = kept // num_kvec  # [n_k_total]
    k0_mask_flat = (kept % num_kvec == 0).to(k_norm2_flat.dtype)  # [n_k_total]
    return k_vectors_flat, k_norm2_flat, k_batch_flat, k0_mask_flat


def compute_k_vectors(
    cutoff: float,
    cell_vectors: torch.Tensor,  # [n_graphs, 3, 3]
    r_cell_vectors: torch.Tensor,  # [n_graphs, 3, 3]
):
    """
    given a batch of cells, computes a tensor of k-vectors [n_graph, n_max_k, 3],
    and a mask [n_graph, n_max_k] describing which k vectors should be included.
    The method for determining the initial search space is one I just made up by
    looking at parallelograms.

    note that this function is meant to run on gpu, so it cannot do the matrix
    inverse internally

        - cutoff
            reciporical space cutoff
        - cell_vectors [n_graph, 3, 3]
            where the -2 dimension indexes the different lattice
            vectors, and the -1 index is the spatial dimension.

    returns:
        - kvectors
            a tensor [n_graph, n_max_k, 3] of the k-vectors associated with each graph
        - k_vectors_normed_squared
            a tensor [n_graph, n_max_k] of the squared norms of the k-vectors
        - mask
            mask describing which of these shuold be included for each graph.
    """
    device = cell_vectors.device

    # compute normed real lattice vectors
    norms = torch.norm(cell_vectors, dim=-1)  # [n_graph, 3]
    normed_lattice_vectors = cell_vectors / norms.unsqueeze(-1)  # [n_graph, 3, 3]

    dot_products = torch.einsum(
        "bij,bij->bi", r_cell_vectors, normed_lattice_vectors
    )  # [n_graph, 3]
    max_ns = cutoff * torch.pow(dot_products, -1)  # [n_graph, 3]
    max_ns = torch.ceil(max_ns).type(torch.int64)

    max_max_ns = torch.max(max_ns, dim=0).values  # [3]
    n1max = int(max_max_ns[0])
    n2max = int(max_max_ns[1])
    n3max = int(max_max_ns[2])

    # make a single superset of all coeficients
    origin = torch.cartesian_prod(
        torch.arange(0, 1, 1, device=device),
        torch.arange(0, 1, 1, device=device),
        torch.arange(0, 1, 1, device=device),
    ).type(torch.float32)

    open_half_sphere = torch.cartesian_prod(
        torch.arange(1, n1max, 1, device=device),
        torch.arange(-n2max, n2max, 1, device=device),
        torch.arange(-n3max, n3max, 1, device=device),
    ).type(torch.float32)

    open_half_plane = torch.cartesian_prod(
        torch.arange(0, 1, 1, device=device),
        torch.arange(1, n2max, 1, device=device),
        torch.arange(-n3max, n3max, 1, device=device),
    ).type(torch.float32)

    open_half_line = torch.cartesian_prod(
        torch.arange(0, 1, 1, device=device),
        torch.arange(0, 1, 1, device=device),
        torch.arange(1, n3max, 1, device=device),
    ).type(torch.float32)

    kvecs = torch.cat(
        (origin, open_half_line, open_half_plane, open_half_sphere), dim=-2
    )
    kvecs = kvecs.to(r_cell_vectors.dtype)

    # produce all kvectors for all graphs with the largest superset of coeficients
    k_vectors = torch.einsum(
        "ni,bij->bnj", kvecs, r_cell_vectors
    )  # [n_graphs, n_kvec, 3]

    # find a mask of all of them which are too large
    k_vectors_normed_squared = torch.einsum(
        "bni,bni->bn", k_vectors, k_vectors
    )  # [n_graphs, n_kvec]
    mask = k_vectors_normed_squared.le(cutoff * cutoff)  # [n_graphs, n_kvec]

    return k_vectors, k_vectors_normed_squared, mask


def evaluate_fourier_series_at_points_flat(
    k_vectors: torch.Tensor,  # [n_k_total, 3]
    k_vector_batch: torch.Tensor,  # [n_k_total]
    fourier_coefficients: torch.Tensor,  # [n_k_total, 2]
    sample_points: torch.Tensor,  # [n_points, 3]
    sample_batch: torch.Tensor,  # [n_points]
    k0_mask: torch.Tensor,  # [n_k_total]
) -> torch.Tensor:
    """
    Evaluate a real-valued field from its Fourier coefficients at arbitrary points.

    \rho(r) = (1/(2\\pi)^3) * sum_k [2 Re{rho_k} cos(k·r) - 2 Im{rho_k} sin(k·r)],
    with the k=0 term not doubled.
    """
    inner_products = torch.matmul(k_vectors, sample_points.t())  # [n_k_total, n_points]
    mask = k_vector_batch[:, None] == sample_batch[None, :]
    mask_f = mask.to(dtype=inner_products.dtype)
    cosines = torch.cos(inner_products) * mask_f
    sines = torch.sin(inner_products) * mask_f

    summand = (
        2.0 * fourier_coefficients[:, 0].unsqueeze(-1) * cosines
        - 2.0 * fourier_coefficients[:, 1].unsqueeze(-1) * sines
    )
    k0_factor = torch.ones_like(k0_mask)
    k0_factor[k0_mask > 0.0] = 0.5
    summand = summand * k0_factor.unsqueeze(-1)
    return torch.sum(summand, dim=0) / (2 * pi) ** 3


def fourier_series_dot(
    series_1: torch.Tensor,  # [n_graph, max_n_k, 2]
    series_2: torch.Tensor,  # [n_graph, max_n_k, 2]
    volumes: torch.Tensor,  # [n_graph]
) -> torch.Tensor:
    """Dot product between two real Fourier series, handling k=0."""
    values_at_zero = torch.einsum("bj,bj->b", series_1[:, 0, :], series_2[:, 0, :])
    values = 2 * torch.einsum("bkj,bkj->b", series_1[:, 1:, :], series_2[:, 1:, :])
    return torch.multiply(values + values_at_zero, volumes) / (2 * pi) ** 6
