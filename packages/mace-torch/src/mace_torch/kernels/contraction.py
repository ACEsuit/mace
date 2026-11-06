"""The V1 fused symmetric contraction over build-time monomial bases.

The public primitive in ``ops`` keeps the uncompressed single-irrep contract.
This fused form shares monomials across output irreps and avoids forming the
highest outer power; its weights retain the same canonical path order.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter

import torch
from torch import Tensor


def monomial_basis(basis: Tensor, order: int) -> Tensor:
    """A body order's basis, rewritten over the symmetric monomials of its input.

    ``basis`` is ``[n_paths, dim_out, dim_in ** order]``, the reduced basis with
    its input axes flattened. Contracted with the outer power, it only ever sees
    that power's symmetric part, so it can be rewritten over the monomials
    ``x_i x_j ...`` with ``i <= j <= ...``: each row is the sum of the basis
    over every ordering of its indices. That is ``C(dim + order - 1, order)``
    rows instead of ``dim ** order``, 816 against 4096 at ``dim = 16`` and
    order 3, and it is what lets :func:`symmetric_contraction_grouped` avoid forming the
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


def symmetric_contraction_grouped_aten(
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


symmetric_contraction_grouped = torch.library.custom_op(
    "mace::symmetric_contraction_grouped", mutates_args=()
)(symmetric_contraction_grouped_aten)


@symmetric_contraction_grouped.register_fake
def _(
    features: Tensor,
    weights: list[Tensor],
    bases: list[Tensor],
    element: Tensor,
    orders: int,
) -> Tensor:
    width = sum(basis.shape[2] for basis in bases[::orders])
    return features.new_empty((features.shape[0], features.shape[1], width))


def _symmetric_contraction_grouped_setup(ctx, inputs, output) -> None:
    features, weights, bases, element, orders = inputs
    ctx.save_for_backward(features, element, *weights, *bases)
    ctx.count = len(weights)
    ctx.orders = orders


def _symmetric_contraction_grouped_backward(ctx, grad):
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


symmetric_contraction_grouped.register_autograd(
    _symmetric_contraction_grouped_backward,
    setup_context=_symmetric_contraction_grouped_setup,
)
