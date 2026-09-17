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

import torch
from torch import Tensor

__all__ = [
    "channelwise_tp_conv",
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


def _outer_power(features: Tensor, order: int) -> Tensor:
    """``features`` raised to the ``order``-fold outer power, flattened.

    Shape ``[n, mul, dim]`` in, ``[n, mul, dim ** order]`` out. Written as
    repeated multiplication rather than an einsum over a variable number of
    axes, because the rank is a build-time constant and this keeps one code
    path for every body order.
    """
    power = features
    for _ in range(order - 1):
        power = (power.unsqueeze(-1) * features.unsqueeze(-2)).flatten(-2)
    return power


@torch.library.custom_op("mace::symmetric_contraction", mutates_args=())
def symmetric_contraction(
    features: Tensor,
    weights: list[Tensor],
    bases: list[Tensor],
    element: Tensor,
) -> Tensor:
    """The many-body contraction over the reduced Clebsch-Gordan basis.

    Args:
        features: ``[n_nodes, num_features, dim_in]``.
        weights: One ``[num_elements, n_paths, num_features]`` array per body
            order, ascending. Canonical ``[Z, A, mul]``, over the basis order
            :mod:`mace_core.clebsch_gordan` pins.
        bases: One ``[n_paths, dim_out, dim_in ** order]`` array per body
            order, ascending. Constant model state, never learned.
        element: ``[n_nodes]``, int64, which element each node is.

    Returns:
        ``[n_nodes, num_features, dim_out]``, the sum over body orders.

    The sum is written out rather than accumulated by a Horner cascade. Both
    compute the same thing; this one keeps each body order's contribution an
    independent term, which is what lets an unreachable order contribute
    nothing instead of needing its weights forced to zero.
    """
    total: Tensor | None = None
    for order, (weight, basis) in enumerate(zip(weights, bases, strict=True), start=1):
        if basis.shape[0] == 0:
            continue
        power = _outer_power(features, order)
        projected = torch.einsum("aof,nmf->nmao", basis, power)
        per_node = weight.index_select(0, element)
        term = torch.einsum("nmao,nam->nmo", projected, per_node)
        total = term if total is None else total + term
    if total is None:
        return features.new_zeros(
            (features.shape[0], features.shape[1], bases[0].shape[1])
        )
    return total


@symmetric_contraction.register_fake
def _(
    features: Tensor, weights: list[Tensor], bases: list[Tensor], element: Tensor
) -> Tensor:
    return features.new_empty((features.shape[0], features.shape[1], bases[0].shape[1]))


def _symmetric_contraction_setup(ctx, inputs, output) -> None:
    features, weights, bases, element = inputs
    ctx.save_for_backward(features, element, *weights, *bases)
    ctx.order_count = len(weights)


def _symmetric_contraction_backward(ctx, grad):
    saved = list(ctx.saved_tensors)
    features, element = saved[0], saved[1]
    count = ctx.order_count
    weights = saved[2 : 2 + count]
    bases = saved[2 + count :]

    # Recomputed rather than saved: the outer powers are the large intermediate
    # and recomputing them is cheaper than holding dim**order per node through
    # the whole backward. Every operation here is differentiable, which is what
    # makes the second derivative work.
    grad_features = torch.zeros_like(features)
    grad_weights = []
    for order, (weight, basis) in enumerate(zip(weights, bases, strict=True), start=1):
        if basis.shape[0] == 0:
            grad_weights.append(torch.zeros_like(weight))
            continue
        power = _outer_power(features, order)
        projected = torch.einsum("aof,nmf->nmao", basis, power)
        per_node = weight.index_select(0, element)

        grad_per_node = torch.einsum("nmo,nmao->nam", grad, projected)
        grad_weight = torch.zeros_like(weight).index_add(0, element, grad_per_node)
        grad_weights.append(grad_weight)

        grad_projected = torch.einsum("nmo,nam->nmao", grad, per_node)
        grad_power = torch.einsum("nmao,aof->nmf", grad_projected, basis)
        grad_features = grad_features + _outer_power_backward(
            features, grad_power, order
        )
    # The structure has to mirror the inputs exactly, lists included: a bare
    # None where the signature has a list is rejected by the autograd shim.
    return grad_features, grad_weights, [None] * len(bases), None


def _outer_power_backward(features: Tensor, grad_power: Tensor, order: int) -> Tensor:
    """The derivative of the outer power, by the product rule.

    The ``order``-fold outer power differentiates into ``order`` terms, each
    contracting the gradient against the power of one degree less on every axis
    but one. Written with einsum over an explicitly reshaped gradient, so it is
    itself differentiable.
    """
    if order == 1:
        return grad_power
    dim = features.shape[-1]
    shaped = grad_power.reshape(*grad_power.shape[:-1], *([dim] * order))
    lower = _outer_power(features, order - 1).reshape(
        *features.shape[:-1], *([dim] * (order - 1))
    )
    total = torch.zeros_like(features)
    letters = "abcdefgh"[: order - 1]
    for axis in range(order):
        held = letters[:axis] + "z" + letters[axis:]
        total = total + torch.einsum(f"nm{held},nm{letters}->nmz", shaped, lower)
    return total


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
        coefficients: ``[n_paths, dim_out, dim_in, dim_edge]``, the
            Clebsch-Gordan coefficients. Constant model state.
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
    """
    gathered = node_features.index_select(0, sender)
    messages = torch.einsum(
        "poid,emi,ed,epm->emo", coefficients, gathered, edge_attributes, radial_weights
    )
    out = node_features.new_zeros(
        (num_nodes, node_features.shape[1], coefficients.shape[1])
    )
    return out.index_add(0, receiver, messages)


@channelwise_tp_conv.register_fake
def _(
    node_features: Tensor,
    edge_attributes: Tensor,
    radial_weights: Tensor,
    coefficients: Tensor,
    sender: Tensor,
    receiver: Tensor,
    num_nodes: int,
) -> Tensor:
    return node_features.new_empty(
        (num_nodes, node_features.shape[1], coefficients.shape[1])
    )


def _conv_setup(ctx, inputs, output) -> None:
    (
        node_features,
        edge_attributes,
        radial_weights,
        coefficients,
        sender,
        receiver,
        _,
    ) = inputs
    ctx.save_for_backward(
        node_features, edge_attributes, radial_weights, coefficients, sender, receiver
    )


def _conv_backward(ctx, grad):
    (
        node_features,
        edge_attributes,
        radial_weights,
        coefficients,
        sender,
        receiver,
    ) = ctx.saved_tensors
    grad_messages = grad.index_select(0, receiver)
    gathered = node_features.index_select(0, sender)

    grad_gathered = torch.einsum(
        "poid,emo,ed,epm->emi",
        coefficients,
        grad_messages,
        edge_attributes,
        radial_weights,
    )
    grad_nodes = torch.zeros_like(node_features).index_add(0, sender, grad_gathered)
    grad_edges = torch.einsum(
        "poid,emo,emi,epm->ed",
        coefficients,
        grad_messages,
        gathered,
        radial_weights,
    )
    grad_radial = torch.einsum(
        "poid,emo,emi,ed->epm", coefficients, grad_messages, gathered, edge_attributes
    )
    return grad_nodes, grad_edges, grad_radial, None, None, None, None


channelwise_tp_conv.register_autograd(_conv_backward, setup_context=_conv_setup)
