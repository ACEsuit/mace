"""The dispatched ops, as torch custom operators.

Each is a ``torch.library.custom_op`` with a meta implementation and a
registered autograd rule. That shape buys three things the frozen tree does not
have:

* ``torch.compile(fullgraph=True)`` sees one opaque node per op instead of
  tracing into it, so a backend can swap its body without the compiler
  noticing and without a graph break.
* The node count enters as a symbolic dimension. The meta implementations below
  build their outputs from ``num_nodes`` as an ``int``, never by reading a
  device tensor and never from ``edge_index.max()``, so one compiled frame
  serves every batch size.
* The backward is registered rather than inferred, and is written in ordinary
  differentiable torch, so differentiating it again works. Training on forces
  needs that second derivative, and a backward that is not itself
  differentiable produces wrong forces rather than an error.

No data-dependent host reads anywhere in a body: a `.item()` or a `bool()` on a
device tensor would break CUDA-graph capture under `reduce-overhead`.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter

import torch
from torch import Tensor

__all__ = [
    "channelwise_tp_conv",
    "monomial_basis",
    "segment_sum",
    "symmetric_contraction",
]


# ---------------------------------------------------------------------------
# segment_sum
# ---------------------------------------------------------------------------


@torch.library.custom_op("mace::segment_sum", mutates_args=())
def segment_sum(values: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Sum ``values`` into ``num_segments`` segments given by ``index``.

    No irreps semantics. The same op reduces messages onto nodes, site energies
    onto graphs, and pair-repulsion terms onto whichever of the two.

    Args:
        values: ``[n, ...]``. The leading axis is what gets reduced.
        index: ``[n]``, int64, the segment each row belongs to.
        num_segments: How many segments, as a plain int so it can be symbolic.
    """
    out = values.new_zeros((num_segments, *values.shape[1:]))
    return out.index_add(0, index, values)


@segment_sum.register_fake
def _(values: Tensor, index: Tensor, num_segments: int) -> Tensor:
    return values.new_empty((num_segments, *values.shape[1:]))


def _segment_sum_setup(ctx, inputs, output) -> None:
    _, index, _ = inputs
    ctx.save_for_backward(index)


def _segment_sum_backward(ctx, grad):
    (index,) = ctx.saved_tensors
    # Gathering is differentiable, so this backward can itself be
    # differentiated, which is what force training needs.
    return grad.index_select(0, index), None, None


segment_sum.register_autograd(_segment_sum_backward, setup_context=_segment_sum_setup)


# ---------------------------------------------------------------------------
# symmetric_contraction
# ---------------------------------------------------------------------------


def monomial_basis(basis: Tensor, order: int) -> Tensor:
    """A body order's basis, rewritten over the symmetric monomials of its input.

    ``basis`` is ``[n_paths, dim_out, dim_in ** order]``, the reduced basis with
    its input axes flattened. Contracted with the outer power, it only ever sees
    that power's symmetric part, so it can be rewritten over the monomials
    ``x_i x_j ...`` with ``i <= j <= ...``: each row is the sum of the basis
    over every ordering of its indices. That is ``C(dim + order - 1, order)``
    rows instead of ``dim ** order``, 816 against 4096 at ``dim = 16`` and
    order 3, and it is what lets :func:`symmetric_contraction` avoid forming the
    power at all.

    Returns ``[n_monomials, n_paths, dim_out]``, rows in the order
    :func:`_monomials` builds them. Computed once, at build time.
    """
    paths, dim_out, flat = basis.shape
    dim = round(flat ** (1.0 / order))
    full = basis.reshape(paths, dim_out, *([dim] * order))
    axes = range(2, 2 + order)
    symmetric = torch.stack(
        [full.permute(0, 1, *each) for each in itertools.permutations(axes)]
    ).mean(0)
    indices = torch.tensor(_monomial_indices(dim, order)).T
    values = symmetric[(slice(None), slice(None), *indices)]
    counts = [Counter(monomial) for monomial in _monomial_indices(dim, order)]
    multiplicity = torch.tensor(
        [
            math.factorial(order) / math.prod(math.factorial(c) for c in each.values())
            for each in counts
        ],
        dtype=basis.dtype,
    )
    return (values * multiplicity).permute(2, 0, 1).contiguous()


def _monomial_indices(dim: int, degree: int) -> list[tuple[int, ...]]:
    """The degree-``degree`` monomials over ``dim`` variables, as sorted index
    tuples, **grouped by their highest variable, ascending**.

    That grouping is the property everything relies on: the monomials whose
    variables are all at most ``k`` are then a prefix of the list, of length
    ``C(k + degree, degree)``, so a monomial of one degree higher is a prefix
    entry times ``x_k``.
    """
    if degree == 0:
        return [()]
    lower = _monomial_indices(dim, degree - 1)
    return [
        (*prefix, k)
        for k in range(dim)
        for prefix in lower[: math.comb(k + degree - 1, degree - 1)]
    ]


def _blocks(dim: int, degree: int) -> list[tuple[int, int]]:
    """``(start, width)`` of each group of degree-``degree`` monomials, one group
    per highest variable ``k``. The group is the degree-``degree - 1`` monomials
    up to ``k``, each times ``x_k``."""
    return [
        (math.comb(k - 1 + degree, degree), math.comb(k + degree - 1, degree - 1))
        for k in range(dim)
    ]


def _monomials(
    columns: Tensor, degree: int, differentiable: bool = False
) -> list[Tensor]:
    """The monomials of the input of every degree up to ``degree``.

    ``columns`` is the input transposed, ``[dim, R]``: one row per input
    component, one column per node and channel. Entry ``d`` of the list is
    ``[C(dim + d - 1, d), R]``, rows in the order :func:`_monomial_indices`
    gives, and each degree is built from the one below, one group per highest
    variable. Monomials run along the first axis so that every group is a
    contiguous block of rows.

    Each group is written straight into its rows, which saves a copy of the
    largest intermediate. That write is not differentiable, so a caller that
    needs the graph, the backward under a second derivative, asks for the
    concatenating form instead. The values are the same.
    """
    powers = [columns.new_ones(1, columns.shape[1]), columns]
    for degree_now in range(2, degree + 1):
        lower = powers[-1]
        blocks = _blocks(columns.shape[0], degree_now)
        if differentiable:
            powers.append(
                torch.cat(
                    [
                        lower[:width] * columns[k : k + 1]
                        for k, (_, width) in enumerate(blocks)
                    ]
                )
            )
            continue
        count = math.comb(columns.shape[0] + degree_now - 1, degree_now)
        upper = columns.new_empty((count, columns.shape[1]))
        for k, (start, width) in enumerate(blocks):
            torch.mul(
                lower[:width], columns[k : k + 1], out=upper[start : start + width]
            )
        powers.append(upper)
    return powers[: degree + 1]


def _joined_bases(bases: list[Tensor], orders: int, order: int) -> Tensor:
    """Every output irrep's basis for one body order, side by side:
    ``[n_monomials, sum of n_paths * dim_out]``."""
    return torch.cat(
        [basis.reshape(basis.shape[0], -1) for basis in bases[order - 1 :: orders]],
        dim=1,
    )


def _horner_basis(joined: Tensor, dim: int, degree: int) -> Tensor:
    """The highest order's basis, one slice per input component.

    ``joined`` is ``[n_monomials, columns]`` at ``degree``. A monomial of that
    degree is one of degree ``degree - 1`` times its highest component ``x_k``,
    so the basis splits into one ``[columns, C(dim + degree - 2, degree - 1)]``
    slice per ``k``, zero past that group's width. Stacked, that is
    ``[dim * columns, n_lower]``: one matrix product against the monomials one
    degree down gives every ``k`` at once, and contracting the result with
    ``x`` is the last Horner step.
    """
    blocks = _blocks(dim, degree)
    lower = blocks[-1][1]
    padded = joined.new_zeros((dim, lower, joined.shape[1]))
    for k, (start, width) in enumerate(blocks):
        padded[k, :width] = joined[start : start + width]
    return padded.transpose(1, 2).reshape(dim * joined.shape[1], lower)


@torch.library.custom_op("mace::symmetric_contraction", mutates_args=())
def symmetric_contraction(
    features: Tensor,
    weights: list[Tensor],
    bases: list[Tensor],
    element: Tensor,
    orders: int,
) -> Tensor:
    """The many-body contraction over the reduced Clebsch-Gordan basis.

    Args:
        features: ``[n_nodes, num_features, dim_in]``.
        weights: One ``[num_elements, n_paths, num_features]`` array per output
            irrep and body order, output irrep outermost and body order
            ascending within it. Canonical ``[Z, A, mul]``, over the basis
            order :mod:`mace_core.clebsch_gordan` pins.
        bases: One ``[n_monomials, n_paths, dim_out]`` array per output irrep
            and body order, in the same order, from :func:`monomial_basis`.
            Constant model state.
        element: ``[n_nodes]``, int64, which element each node is.
        orders: The body orders per output irrep, which is the correlation.

    Returns:
        ``[n_nodes, num_features, sum of dim_out]``: each output irrep's sum
        over body orders, concatenated in the order given.

    Every order below the highest is one matrix product of the input's
    monomials against every output irrep's basis at once. The highest order's
    monomials are never formed: its basis is contracted with the monomials one
    degree down and then with the input, a Horner step, so the largest
    intermediate is ``dim_in`` projections per node and channel. The weights are
    applied per node after the projection, so nothing grows with the number of
    elements.
    """
    nodes, channels, dim = features.shape
    columns = features.reshape(-1, dim).T.contiguous()
    powers = _monomials(columns, max(orders - 1, 1))
    projections = []
    for order in range(1, orders + 1):
        joined = _joined_bases(bases, orders, order)
        if order < orders or order == 1:
            projections.append(joined.T @ powers[order])
            continue
        partial = _horner_basis(joined, dim, order) @ powers[order - 1]
        partial = partial.view(dim, joined.shape[1], columns.shape[1])
        partial = partial.mul_(columns[:, None])
        projections.append(partial.sum(0))
    return _weighted(projections, weights, bases, element, orders, nodes, channels)


def _weighted(
    projections: list[Tensor],
    weights: list[Tensor],
    bases: list[Tensor],
    element: Tensor,
    orders: int,
    nodes: int,
    channels: int,
) -> Tensor:
    """Each output irrep's projections weighted per node and summed over body
    orders, concatenated: ``[n_nodes, num_features, sum of dim_out]``."""
    pieces: list[Tensor] = []
    offsets = [0] * orders
    for target in range(len(bases) // orders):
        dim_out = bases[target * orders].shape[2]
        total = projections[0].new_zeros((nodes, channels, dim_out))
        for order in range(1, orders + 1):
            paths = bases[target * orders + order - 1].shape[1]
            start = offsets[order - 1]
            offsets[order - 1] += paths * dim_out
            if paths == 0:
                continue
            projected = projections[order - 1][start : start + paths * dim_out]
            per_node = weights[target * orders + order - 1].index_select(0, element)
            total = total + torch.einsum(
                "aonc,nac->nco",
                projected.reshape(paths, dim_out, nodes, channels),
                per_node,
            )
        pieces.append(total)
    return torch.cat(pieces, dim=-1)


@symmetric_contraction.register_fake
def _(
    features: Tensor,
    weights: list[Tensor],
    bases: list[Tensor],
    element: Tensor,
    orders: int,
) -> Tensor:
    width = sum(basis.shape[2] for basis in bases[::orders])
    return features.new_empty((features.shape[0], features.shape[1], width))


def _symmetric_contraction_setup(ctx, inputs, output) -> None:
    features, weights, bases, element, orders = inputs
    ctx.save_for_backward(features, element, *weights, *bases)
    ctx.count = len(weights)
    ctx.orders = orders


def _symmetric_contraction_backward(ctx, grad):
    saved = list(ctx.saved_tensors)
    features, element = saved[0], saved[1]
    weights = saved[2 : 2 + ctx.count]
    bases = saved[2 + ctx.count :]
    orders = ctx.orders
    nodes, channels, dim = features.shape
    rows = nodes * channels
    columns = features.reshape(-1, dim).T.contiguous()
    targets = len(bases) // orders

    # The monomials are recomputed rather than saved. Every operation from here
    # on is differentiable, in place or not, which is what makes the second
    # derivative work.
    powers = _monomials(
        columns, max(orders - 1, 1), differentiable=torch.is_grad_enabled()
    )
    grad_powers: dict[int, Tensor] = {}

    def accumulate(degree: int, value: Tensor) -> None:
        grad_powers[degree] = (
            value if degree not in grad_powers else grad_powers[degree] + value
        )

    grads_out = torch.split(
        grad, [bases[target * orders].shape[2] for target in range(targets)], dim=-1
    )
    grad_weights: dict[int, Tensor] = {}
    for order in range(1, orders + 1):
        joined = _joined_bases(bases, orders, order)
        per_target = [
            (target * orders + order - 1, bases[target * orders + order - 1])
            for target in range(targets)
        ]
        grad_projected = torch.cat(
            [
                torch.einsum(
                    "nco,nac->aonc",
                    grads_out[target],
                    weights[i].index_select(0, element),
                ).reshape(basis.shape[1] * basis.shape[2], rows)
                for target, (i, basis) in enumerate(per_target)
            ]
        )
        if order < orders or order == 1:
            projected = joined.T @ powers[order]
            accumulate(order, joined @ grad_projected)
        else:
            horner = _horner_basis(joined, dim, order)
            partial = (horner @ powers[order - 1]).view(dim, joined.shape[1], rows)
            projected = (partial * columns[:, None]).sum(0)
            accumulate(1, (partial * grad_projected[None]).sum(1))
            spread = columns[:, None] * grad_projected[None]
            accumulate(order - 1, horner.T @ spread.reshape(horner.shape[0], rows))
        offset = 0
        for target, (i, basis) in enumerate(per_target):
            paths, dim_out = basis.shape[1], basis.shape[2]
            block = projected[offset : offset + paths * dim_out]
            offset += paths * dim_out
            grad_per_node = torch.einsum(
                "nco,aonc->nac",
                grads_out[target],
                block.reshape(paths, dim_out, nodes, channels),
            )
            grad_weights[i] = torch.zeros_like(weights[i]).index_add(
                0, element, grad_per_node
            )

    # Down through the monomials: degree d is degree d - 1 times one component.
    grad_columns = grad_powers[1]
    for degree in range(orders - 1, 1, -1):
        lower, upper = powers[degree - 1], grad_powers[degree]
        for k, (start, width) in enumerate(_blocks(dim, degree)):
            group = upper[start : start + width]
            grad_columns[k] += (group * lower[:width]).sum(0)
            grad_powers[degree - 1][:width].addcmul_(group, columns[k : k + 1])

    # The structure has to mirror the inputs exactly, lists included: a bare
    # None where the signature has a list is rejected by the autograd shim.
    return (
        grad_columns.T.reshape(features.shape),
        [grad_weights[i] for i in range(len(weights))],
        [None] * len(bases),
        None,
        None,
    )


symmetric_contraction.register_autograd(
    _symmetric_contraction_backward, setup_context=_symmetric_contraction_setup
)


# ---------------------------------------------------------------------------
# channelwise_tp_conv
# ---------------------------------------------------------------------------


@torch.library.custom_op("mace::channelwise_tp_conv", mutates_args=())
def channelwise_tp_conv(
    node_features: Tensor,
    edge_attributes: Tensor,
    radial_weights: Tensor,
    coefficients: Tensor,
    path_widths: list[int],
    sender: Tensor,
    receiver: Tensor,
    num_nodes: int,
) -> Tensor:
    """The message-passing tensor product, reduced onto the nodes.

    Args:
        node_features: ``[n_nodes, num_features, dim_in]``.
        edge_attributes: ``[n_edges, dim_edge]``, normally spherical harmonics
            of the edge direction.
        radial_weights: ``[n_edges, n_paths, num_features]``, from the radial
            MLP, which is external to this op.
        coefficients: ``[dim_out, dim_in, dim_edge]``, the Clebsch-Gordan
            coefficients of every path. Constant model state.
        path_widths: How many output components each path owns, in order. The
            paths own consecutive, disjoint blocks of ``dim_out``, which is
            what lets one array hold all of their coefficients.
        sender: ``[n_edges]``, int64.
        receiver: ``[n_edges]``, int64.
        num_nodes: The node count, as a plain int so it stays symbolic under
            compile. Never ``edge_index.max()``, which would be a
            data-dependent host read and a recompile on every batch.

    Returns:
        ``[n_nodes, num_features, dim_out]``. **Always node-level.** Whether
        the reduction is fused into the kernel is a backend's business, which
        is what removes the six ``conv_fusion`` branches the frozen tree
        carries inside its interaction blocks.

    The coupling with the edge attributes carries no channel, so it is done
    first, for every edge at once: ``[n_edges, dim_out, dim_in]``. One batched
    product with the senders' features then gives every path's message, held
    ``[n_edges, dim_out, num_features]`` so that the channels are the long
    side of each product and each path's radial weights scale one contiguous
    block. The messages are the largest intermediate, the same array the
    frozen tree forms before its scatter.
    """
    channels = node_features.shape[1]
    gathered = node_features.index_select(0, sender)
    coupling = _edge_coupling(edge_attributes, coefficients)
    messages = torch.bmm(coupling, gathered.transpose(1, 2))
    start = 0
    for path, width in enumerate(path_widths):
        messages[:, start : start + width].mul_(radial_weights[:, path, None, :])
        start += width
    out = messages.new_zeros((num_nodes, coefficients.shape[0], channels))
    return out.index_add_(0, receiver, messages).transpose(1, 2).contiguous()


def _edge_coupling(edge_attributes: Tensor, coefficients: Tensor) -> Tensor:
    """The coefficients contracted with each edge's attributes:
    ``[n_edges, dim_out, dim_in]``."""
    dim_out, dim_in, dim_edge = coefficients.shape
    flat = coefficients.reshape(dim_out * dim_in, dim_edge)
    return (edge_attributes @ flat.T).reshape(-1, dim_out, dim_in)


@channelwise_tp_conv.register_fake
def _(
    node_features: Tensor,
    edge_attributes: Tensor,
    radial_weights: Tensor,
    coefficients: Tensor,
    path_widths: list[int],
    sender: Tensor,
    receiver: Tensor,
    num_nodes: int,
) -> Tensor:
    return node_features.new_empty(
        (num_nodes, node_features.shape[1], coefficients.shape[0])
    )


def _conv_setup(ctx, inputs, output) -> None:
    (
        node_features,
        edge_attributes,
        radial_weights,
        coefficients,
        path_widths,
        sender,
        receiver,
        _,
    ) = inputs
    ctx.save_for_backward(
        node_features, edge_attributes, radial_weights, coefficients, sender, receiver
    )
    ctx.path_widths = list(path_widths)


def _conv_backward(ctx, grad):
    (
        node_features,
        edge_attributes,
        radial_weights,
        coefficients,
        sender,
        receiver,
    ) = ctx.saved_tensors
    dim_out, dim_in, dim_edge = coefficients.shape
    widths = ctx.path_widths
    if not widths:
        return (
            torch.zeros_like(node_features),
            torch.zeros_like(edge_attributes),
            torch.zeros_like(radial_weights),
            None,
            None,
            None,
            None,
            None,
        )
    edges = edge_attributes.shape[0]
    gathered = node_features.index_select(0, sender)
    coupling = _edge_coupling(edge_attributes, coefficients)
    # One contiguous gather of the gradient per edge, ``[n_edges, dim_out,
    # num_features]``, the layout the forward held its messages in.
    incoming = grad.transpose(1, 2).contiguous().index_select(0, receiver)
    # Each path's block scaled by its radial weights. Built by slices rather
    # than by gathering the weights out to every component, which would need
    # an index tensor made on the host inside the backward.
    weighted = torch.cat(
        [
            piece * radial_weights[:, path, None, :]
            for path, piece in enumerate(torch.split(incoming, widths, dim=1))
        ],
        dim=1,
    )

    # Every operation is differentiable, which is what the second derivative
    # needs.
    grad_gathered = _edge_product(coupling.transpose(1, 2), weighted).transpose(1, 2)
    grad_coupling = _edge_product(weighted, gathered)
    unweighted = torch.bmm(coupling, gathered.transpose(1, 2))
    grad_radial = torch.stack(
        [piece.sum(1) for piece in torch.split(incoming * unweighted, widths, dim=1)],
        dim=1,
    )
    grad_nodes = torch.zeros_like(node_features).index_add(0, sender, grad_gathered)
    flat = coefficients.reshape(dim_out * dim_in, dim_edge)
    grad_edges = grad_coupling.reshape(edges, dim_out * dim_in) @ flat
    return (
        grad_nodes,
        grad_edges,
        grad_radial,
        None,
        None,
        None,
        None,
        None,
    )


def _edge_product(left: Tensor, right: Tensor) -> Tensor:
    """``left @ right`` edge by edge, where one side has only a few rows or
    columns: the input irreps' components.

    In float32 on CUDA that shape gets a poor kernel. Measured on an A100 at
    153714 edges, ``[40 x 128] @ [128 x 4]`` per edge takes 37.6 ms as one
    batched product and 11.0 ms as four products of one column each, and
    ``[4 x 40] @ [40 x 128]`` 13.9 against 10.7 ms one row at a time. In
    float64 the single product is the fast one, 6.9 and 6.3 ms against 22.0
    and 21.3, and on the CPU splitting only multiplies the per-edge loop. So
    only float32 on CUDA takes the short side one slice at a time.
    """
    if left.dtype != torch.float32 or not left.is_cuda:
        return torch.bmm(left, right)
    if right.shape[2] <= left.shape[1]:
        return torch.cat(
            [torch.bmm(left, right[:, :, k : k + 1]) for k in range(right.shape[2])],
            dim=2,
        )
    return torch.cat(
        [torch.bmm(left[:, k : k + 1], right) for k in range(left.shape[1])], dim=1
    )


channelwise_tp_conv.register_autograd(_conv_backward, setup_context=_conv_setup)


# ---------------------------------------------------------------------------
# equivariant_linear
# ---------------------------------------------------------------------------


@torch.library.custom_op("mace::equivariant_linear", mutates_args=())
def equivariant_linear(
    features: Tensor,
    weights: Tensor,
    row: Tensor,
    column: Tensor,
    source: Tensor,
    bias: Tensor,
    bias_row: Tensor,
    dim_out: int,
) -> Tensor:
    """An equivariant linear map, plus a bias on the scalar outputs.

    Args:
        features: ``[n, dim_in]``.
        weights: ``[n_weights]``, flat and canonical.
        row: ``[n_entries]``, the output component each entry writes to.
        column: ``[n_entries]``, the input component it reads from.
        source: ``[n_entries]``, which weight it uses. One weight appears
            ``2l+1`` times, once per component of its irrep, which is what
            makes the map equivariant rather than a free matrix.
        bias: ``[n_bias]``, added to the scalar outputs. Only ``0e`` may carry
            one without breaking equivariance.
        bias_row: ``[n_bias]``, which output component each bias adds to.
        dim_out: The output width, as a plain int.

    The structure travels as index tensors rather than as Python constants.
    That is what lets one registered operator serve every irreps declaration in
    a model: an op body cannot read a build-time table, and reading one from a
    device tensor would be a host read on the hot path.
    """
    dense = features.new_zeros((dim_out, features.shape[-1]))
    dense = dense.index_put((row, column), weights.index_select(0, source))
    out = features @ dense.transpose(0, 1)
    if bias.numel():
        addition = out.new_zeros((dim_out,)).index_add(0, bias_row, bias)
        out = out + addition
    return out


@equivariant_linear.register_fake
def _(
    features: Tensor,
    weights: Tensor,
    row: Tensor,
    column: Tensor,
    source: Tensor,
    bias: Tensor,
    bias_row: Tensor,
    dim_out: int,
) -> Tensor:
    return features.new_empty((*features.shape[:-1], dim_out))


def _linear_setup(ctx, inputs, output) -> None:
    features, weights, row, column, source, bias, bias_row, _ = inputs
    ctx.save_for_backward(features, weights, row, column, source, bias, bias_row)


def _linear_backward(ctx, grad):
    features, weights, row, column, source, bias, bias_row = ctx.saved_tensors
    dense = features.new_zeros((grad.shape[-1], features.shape[-1]))
    dense = dense.index_put((row, column), weights.index_select(0, source))

    grad_features = grad @ dense
    outer = grad.reshape(-1, grad.shape[-1]).transpose(0, 1) @ features.reshape(
        -1, features.shape[-1]
    )
    per_entry = outer[row, column]
    grad_weights = torch.zeros_like(weights).index_add(0, source, per_entry)
    grad_bias = (
        grad.reshape(-1, grad.shape[-1]).sum(0).index_select(0, bias_row)
        if bias.numel()
        else torch.zeros_like(bias)
    )
    return grad_features, grad_weights, None, None, None, grad_bias, None, None


equivariant_linear.register_autograd(_linear_backward, setup_context=_linear_setup)
