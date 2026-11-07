# Adapted from graph_longrange/utils.py by Will Baldwin (MIT).
# https://github.com/WillBaldwin0/graph_electrostatics
# See LICENSE.md and this package's __init__.py for provenance and local changes.

import torch

# internal units are (electon, Volt, Angstrom)
# 5.526349406 * 1e-3 is the value of epsilon_0 in (electon, Volt, Angstrom) units.
# see docs.
FIELD_CONSTANT = 1 / (5.526349406 * 1e-3)

# madelung constant is used for the non-periodic monopole correction term
CUBIC_MADELUNG = 2.837297


# Adapted from PyTorch Geometric's to_dense_batch (Matthias Fey et al., MIT):
# https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/utils/to_dense_batch.py
# Retained here through graph_longrange's utility implementation.
def to_dense_batch(
    x: torch.Tensor,
    batch: torch.Tensor | None = None,
    fill_value: float = 0.0,
    max_num_nodes: int | None = None,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    if batch is None and max_num_nodes is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    num_nodes = scatter_sum(
        batch.new_ones(x.size(0)), batch, dim=0, dim_size=batch_size
    )
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes, *list(x.size())[1:]]
    out = x.new_full(size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes, *list(x.size())[1:]])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool, device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask


# _broadcast, scatter_sum and scatter_mean adapt pytorch_scatter (Matthias Fey,
# MIT), retained through graph_longrange:
# https://github.com/rusty1s/pytorch_scatter/blob/2.0.9/torch_scatter/utils.py
# https://github.com/rusty1s/pytorch_scatter/blob/2.0.9/torch_scatter/scatter.py
def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = _broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


def any_periodic(pbc: torch.Tensor, periodic: tuple[bool, ...] | None) -> bool:
    """Whether any structure of the batch is periodic along any axis.

    From ``periodic``, the host's own answer per structure, when the caller
    has it, and otherwise read off ``pbc``, which waits on the device.
    """
    if periodic is not None:
        return any(periodic)
    return bool(torch.any(pbc))


def open_atoms(
    sizes: tuple[int, ...], periodic: tuple[bool, ...], device: torch.device
) -> tuple[torch.Tensor | None, tuple[int, ...]]:
    """The atoms of the structures periodic along no axis, from host counts.

    Returns:
        Their rows in the batch, or ``None`` when there are none, and every
        structure's count of them: its size when it is open, zero otherwise.
    """
    counts = tuple(
        0 if flag else int(size) for size, flag in zip(sizes, periodic, strict=True)
    )
    ranges, start = [], 0
    for size, count in zip(sizes, counts, strict=True):
        if count:
            ranges.append(torch.arange(start, start + count, device=device))
        start += int(size)
    if not ranges:
        return None, counts
    return torch.cat(ranges), counts
