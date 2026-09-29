"""The example backend's own kernels, registered with ``torch.library``.

Three ops, each a ``custom_op`` with the three registrations a backend of its
own has to write: the forward, a ``register_fake`` that says what the output
looks like without computing it, and a ``register_autograd`` whose backward is
written in differentiable torch operations.

That last part is the requirement a vendor kernel hides. A model trained on
forces differentiates the energy, and then differentiates that derivative to
get at the weights, so the backward is itself differentiated. A backward that
is correct but not differentiable raises nothing: it gives a second derivative
of zero along whatever it treats as constant, and the forces train wrong. Here
every backward is built from ops autograd can see through, so the second
derivative is autograd's.

Shapes: ``N`` nodes, ``E`` edges, ``C`` channels, ``I`` components of one
channel's input, ``J`` of an edge's attributes, ``O`` of one channel's output,
``P`` paths, ``A`` basis entries of one output irrep at one body order and
``Z`` elements.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["contraction", "segment_sum", "tp_conv"]


def _add_into(rows: int, index: Tensor, values: Tensor) -> Tensor:
    """``rows`` rows, row ``index[i]`` the sum of every ``values[i]`` sent to it.

    Accumulated by sorting the indices rather than with atomics, so the sum
    comes out the same on every call on a GPU too. That is what
    ``gradgradcheck`` asks of a backward before it will compare it with a
    difference quotient: an accumulation whose order changes from call to
    call fails it at float64 however right it is. Slower than ``index_add``,
    and the trade this backend makes.

    The gathers are written as indexing, ``values[index]``, for the same
    reason: its derivative is this accumulation, and the derivative of this is
    that gather, so every order of derivative stays deterministic. The
    derivative of ``index_select`` is an ``index_add``, which is not.
    """
    return values.new_zeros((rows, *values.shape[1:])).index_put(
        (index,), values, accumulate=True
    )


# ---------------------------------------------------------------------------
# The reduction into segments
# ---------------------------------------------------------------------------


@torch.library.custom_op("mace_backend_example::segment_sum", mutates_args=())
def segment_sum(values: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Row ``i`` of ``values`` added into row ``index[i]`` of the result.

    ``num_segments`` is a host integer, never read off a device tensor, so a
    compiled graph keeps it symbolic.
    """
    return _add_into(num_segments, index, values)


@segment_sum.register_fake
def _(values: Tensor, index: Tensor, num_segments: int) -> Tensor:
    return values.new_empty((num_segments, *values.shape[1:]))


def _segment_sum_setup(ctx, inputs, output) -> None:
    _, index, _ = inputs
    ctx.save_for_backward(index)


def _segment_sum_backward(ctx, grad: Tensor):
    (index,) = ctx.saved_tensors
    return grad[index], None, None


segment_sum.register_autograd(_segment_sum_backward, setup_context=_segment_sum_setup)


# ---------------------------------------------------------------------------
# The message-passing tensor product
# ---------------------------------------------------------------------------


def _edge_coupling(coupling: Tensor, edge: Tensor) -> Tensor:
    """``[E, O, I]``: the coupling coefficients against each edge's attributes.

    Every product here has two operands. A product of three is contracted in
    an order chosen from the sizes, and under ``torch.compile`` choosing it
    reads the sizes as numbers, which is one compilation per size.
    """
    return torch.einsum("oij,ej->eoi", coupling, edge)


def _unweighted(coupling: Tensor, gathered: Tensor, edge: Tensor) -> Tensor:
    """``[E, C, O]``: each edge's coupled product, before its radial weight."""
    return torch.einsum("eoi,eci->eco", _edge_coupling(coupling, edge), gathered)


@torch.library.custom_op("mace_backend_example::tp_conv", mutates_args=())
def tp_conv(
    node: Tensor,
    edge: Tensor,
    weights: Tensor,
    coupling: Tensor,
    path: Tensor,
    sender: Tensor,
    receiver: Tensor,
    num_nodes: int,
) -> Tensor:
    """``[N, C, O]``: the coupled, weighted messages summed at each receiver.

    Args:
        node: ``[N, C, I]``, channel-major node features.
        edge: ``[E, J]``, the edge attributes.
        weights: ``[E, P, C]``, one radial weight per edge, path and channel.
        coupling: ``[O, I, J]``, the coupling coefficients of every path.
        path: ``[O]``, which path each output component belongs to.
        sender: ``[E]``.
        receiver: ``[E]``.
        num_nodes: A host integer.
    """
    spread = weights[:, path].transpose(1, 2)
    messages = _unweighted(coupling, node[sender], edge) * spread
    return _add_into(num_nodes, receiver, messages)


@tp_conv.register_fake
def _(
    node: Tensor,
    edge: Tensor,
    weights: Tensor,
    coupling: Tensor,
    path: Tensor,
    sender: Tensor,
    receiver: Tensor,
    num_nodes: int,
) -> Tensor:
    return node.new_empty((num_nodes, node.shape[1], coupling.shape[0]))


def _tp_conv_setup(ctx, inputs, output) -> None:
    node, edge, weights, coupling, path, sender, receiver, _ = inputs
    ctx.save_for_backward(node, edge, weights, coupling, path, sender, receiver)


def _tp_conv_backward(ctx, grad: Tensor):
    node, edge, weights, coupling, path, sender, receiver = ctx.saved_tensors
    at_edge = grad[receiver]
    gathered = node[sender]
    spread = weights[:, path].transpose(1, 2)

    # The weights: each path's weight scales its own output components, so
    # its gradient sums those components back into the path.
    per_component = at_edge * _unweighted(coupling, gathered, edge)
    grad_weights = _add_into(
        weights.shape[1], path, per_component.permute(2, 0, 1)
    ).permute(1, 0, 2)

    through = at_edge * spread
    grad_node = _add_into(
        node.shape[0],
        sender,
        torch.einsum("eoi,eco->eci", _edge_coupling(coupling, edge), through),
    )
    outer = torch.einsum("eco,eci->eoi", through, gathered)
    grad_edge = torch.einsum("eoi,oij->ej", outer, coupling)
    return grad_node, grad_edge, grad_weights, None, None, None, None, None


tp_conv.register_autograd(_tp_conv_backward, setup_context=_tp_conv_setup)


# ---------------------------------------------------------------------------
# The symmetric contraction
# ---------------------------------------------------------------------------


def _contract(basis: Tensor, features: Tensor, times: int) -> Tensor:
    """``basis`` with its last ``times`` input axes contracted with the
    features, channel by channel.

    ``basis`` is ``[A, O, I, ..., I]``; the result is ``[N, C, A, O, I, ...]``
    with the input axes that are left. The basis is symmetric in its input
    axes, so which ones are contracted does not matter.
    """
    if times == 0:
        return basis.expand(features.shape[0], features.shape[1], *basis.shape)
    reduced = torch.einsum("ao...i,nci->ncao...", basis, features)
    for _ in range(times - 1):
        reduced = torch.einsum("ncao...i,nci->ncao...", reduced, features)
    return reduced


def _orders(bases: list[Tensor]) -> list[int]:
    return [basis.dim() - 2 for basis in bases]


@torch.library.custom_op("mace_backend_example::contraction", mutates_args=())
def contraction(
    features: Tensor,
    weights: list[Tensor],
    bases: list[Tensor],
    element: Tensor,
    targets: list[int],
    widths: list[int],
) -> Tensor:
    """``[N, C, sum(widths)]``: the weighted symmetric powers of the features.

    Args:
        features: ``[N, C, I]``, channel-major.
        weights: One ``[Z, A, C]`` per basis.
        bases: One ``[A, O, I, ..., I]`` per output irrep and body order, with
            as many input axes as the body order.
        element: ``[N]``, each node's element index.
        targets: For each basis, which output irrep it writes.
        widths: Each output irrep's dimension, in output order.
    """
    outputs = [
        features.new_zeros((features.shape[0], features.shape[1], width))
        for width in widths
    ]
    for weight, basis, target, order in zip(
        weights, bases, targets, _orders(bases), strict=True
    ):
        per_node = weight[element]
        term = torch.einsum(
            "nac,ncao->nco", per_node, _contract(basis, features, order)
        )
        outputs[target] = outputs[target] + term
    return torch.cat(outputs, dim=-1)


@contraction.register_fake
def _(
    features: Tensor,
    weights: list[Tensor],
    bases: list[Tensor],
    element: Tensor,
    targets: list[int],
    widths: list[int],
) -> Tensor:
    return features.new_empty((features.shape[0], features.shape[1], sum(widths)))


def _contraction_setup(ctx, inputs, output) -> None:
    features, weights, bases, element, targets, widths = inputs
    ctx.count = len(weights)
    ctx.targets = list(targets)
    ctx.widths = list(widths)
    ctx.num_elements = [weight.shape[0] for weight in weights]
    ctx.save_for_backward(features, element, *weights, *bases)


def _contraction_backward(ctx, grad: Tensor):
    features, element, *rest = ctx.saved_tensors
    weights, bases = rest[: ctx.count], rest[ctx.count :]
    by_target = torch.split(grad, ctx.widths, dim=-1)
    grad_features = torch.zeros_like(features)
    grad_weights = []
    for weight, basis, target, order in zip(
        weights, bases, ctx.targets, _orders(bases), strict=True
    ):
        upstream = by_target[target]
        per_node = weight[element]
        full = _contract(basis, features, order)
        grad_weights.append(
            _add_into(
                weight.shape[0], element, torch.einsum("nco,ncao->nac", upstream, full)
            )
        )
        # One factor fewer, times the order: the derivative of a symmetric
        # form of degree k is k times the form of degree k - 1.
        partial = _contract(basis, features, order - 1)
        weighted = torch.einsum("nco,nac->ncao", upstream, per_node)
        grad_features = grad_features + order * torch.einsum(
            "ncao,ncaoi->nci", weighted, partial
        )
    # One entry per input, lists included: the bases are constants.
    return grad_features, grad_weights, [None] * len(bases), None, None, None


contraction.register_autograd(_contraction_backward, setup_context=_contraction_setup)
