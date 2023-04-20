import os
import os.path as osp
import ssl
import urllib
import zipfile
from typing import Optional, Tuple
import torch

from torch import Tensor
from mace.tools.scatter import scatter_sum


def makedirs(dir):
    os.makedirs(dir, exist_ok=True)


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition("/")[2].split("?")[0]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print("Using exist file", filename)
        return path

    if log:
        print("Downloading", url)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        f.write(data.read())

    return path


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)


def to_dense_batch(
    x: Tensor,
    batch: Optional[Tensor] = None,
    fill_value: float = 0.0,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Given a sparse batch of node features
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
    :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
    dense node feature tensor
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).
    In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
    N_{\max}}` is returned, holding information about the existence of
    fake-nodes in the dense representation.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered. (default: :obj:`None`)
        fill_value (float, optional): The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)
        batch_size (int, optional) The batch size. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`BoolTensor`)

    Examples:

        >>> x = torch.arange(12).view(6, 2)
        >>> x
        tensor([[ 0,  1],
                [ 2,  3],
                [ 4,  5],
                [ 6,  7],
                [ 8,  9],
                [10, 11]])

        >>> out, mask = to_dense_batch(x)
        >>> mask
        tensor([[True, True, True, True, True, True]])

        >>> batch = torch.tensor([0, 0, 1, 2, 2, 2])
        >>> out, mask = to_dense_batch(x, batch)
        >>> out
        tensor([[[ 0,  1],
                [ 2,  3],
                [ 0,  0]],
                [[ 4,  5],
                [ 0,  0],
                [ 0,  0]],
                [[ 6,  7],
                [ 8,  9],
                [10, 11]]])
        >>> mask
        tensor([[ True,  True, False],
                [ True, False, False],
                [ True,  True,  True]])

        >>> out, mask = to_dense_batch(x, batch, max_num_nodes=4)
        >>> out
        tensor([[[ 0,  1],
                [ 2,  3],
                [ 0,  0],
                [ 0,  0]],
                [[ 4,  5],
                [ 0,  0],
                [ 0,  0],
                [ 0,  0]],
                [[ 6,  7],
                [ 8,  9],
                [10, 11],
                [ 0,  0]]])

        >>> mask
        tensor([[ True,  True, False, False],
                [ True, False, False, False],
                [ True,  True,  True, False]])
    """
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

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = x.new_full(size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool, device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask


def to_dense_adj(
    edge_index: Tensor,
    batch: Optional[Tensor] = None,
    edge_attr: Optional[Tensor] = None,
    max_num_nodes: Optional[int] = None,
) -> Tensor:
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:

        >>> edge_index = torch.tensor([[0, 0, 1, 2, 3],
        ...                            [0, 1, 0, 3, 0]])
        >>> batch = torch.tensor([0, 0, 1, 1])
        >>> to_dense_adj(edge_index, batch)
        tensor([[[1., 1.],
                [1., 0.]],
                [[0., 1.],
                [1., 0.]]])

        >>> to_dense_adj(edge_index, batch, max_num_nodes=4)
        tensor([[[1., 1., 0., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 1., 0., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]]])

        >>> edge_attr = torch.Tensor([1, 2, 3, 4, 5])
        >>> to_dense_adj(edge_index, batch, edge_attr)
        tensor([[[1., 2.],
                [3., 0.]],
                [[0., 4.],
                [5., 0.]]])
    """
    if batch is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        batch = edge_index.new_zeros(num_nodes)

    batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter_sum(one, batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()

    elif (idx1.numel() > 0 and idx1.max() >= max_num_nodes) or (
        idx2.numel() > 0 and idx2.max() >= max_num_nodes
    ):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    adj = torch.zeros(size, dtype=edge_attr.dtype, device=edge_index.device)

    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = adj.view([flattened_size] + list(adj.size())[3:])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    scatter_sum(edge_attr, idx, dim=0, out=adj)
    adj = adj.view(size)

    return adj
