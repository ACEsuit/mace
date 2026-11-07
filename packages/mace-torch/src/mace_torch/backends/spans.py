"""Public stateless interaction-span call contract, kernel spec 1.2.

The model owns every tensor. ``member_weights`` contains, in order:
cutoff radius; basis tensors (Bessel frequencies/prefactor, Gaussian centers,
none for Chebyshev); transform tensors (Agnesi amplitude/q/p/radii or Soft
steepness/radii); then the radial network's canonical tensors. A normalized
SiLU network supplies its matrices in layer order; a layer-norm SiLU network
supplies matrix/bias pairs, then norm-weight/norm-bias pairs. These are live
views, never detached checkpoint tensors.

When present, the density network's tensors follow the radial network's in
the same format. Element-conditioned spans then receive sender and receiver
tables, each ``[num_elements, element_embedding_width]``, including the element
linear maps' normalization. ``edges.node_elements`` indexes these tables.
They are differentiable views of the model's parameters.

``edge_runs`` groups receivers if requested, otherwise senders if requested,
and is None without either requirement. Both views are available on ``edges``
when a backend declares both. No span sorts or resolves a backend in forward.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

from torch import Tensor, nn

from mace_torch.graph.edges import EdgeRuns
from mace_torch.nn.edges import EdgeInputs

__all__ = [
    "EdgeInputs",
    "EdgeRuns",
    "InteractionSpan",
    "SpanCapabilities",
    "validate_span",
]


class SpanCapabilities(NamedTuple):
    double_backward: bool
    weight_gradients: bool


class InteractionSpan(Protocol):
    capabilities: SpanCapabilities

    def __call__(
        self,
        node_features: Tensor,
        edges: EdgeInputs,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
        member_weights: tuple[Tensor, ...],
        edge_runs: EdgeRuns | None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Receiver-summed convolution in the chain layout.

        With a density member, return ``(message, density)``. Density has shape
        ``[num_nodes, 1]`` and sums ``tanh(network(input)**2)`` over receivers.
        Its normalization is applied by the interaction outside the span.
        """
        ...


def validate_span(span: nn.Module, backend_name: str) -> None:
    """Refuse held state and undeclared capabilities before any model runs."""
    if not isinstance(span, nn.Module):
        raise TypeError(f"backend {backend_name!r} must return a span module or None.")
    held = set(span.state_dict()) | {name for name, _ in span.named_buffers()}
    held |= {name for name, _ in span.named_parameters()}
    if held:
        raise ValueError(
            f"backend {backend_name!r}: span {type(span).__name__} "
            f"holds {sorted(held)}; a span must have no parameters or buffers. "
            "Pass member weights on each call."
        )
    if not isinstance(span.capabilities, SpanCapabilities) or any(
        type(value) is not bool for value in span.capabilities
    ):
        raise TypeError(
            f"backend {backend_name!r}: declare SpanCapabilities with two booleans."
        )
