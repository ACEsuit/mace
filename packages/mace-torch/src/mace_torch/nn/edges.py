"""What the interactions read off the edges, computed once and only if read.

The backbone forms the edge vectors once, from the positions, the edge index
and the shifts, and hands every interaction block one :class:`EdgeInputs`.
Everything derived from the vectors is computed the first time something reads
it and cached for that forward, so a quantity no op reads is never allocated.
A backend's fused span, for instance, reads the vectors and computes the
harmonics and the radial embedding inside its kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from mace_torch.graph.edges import EdgeRuns

if TYPE_CHECKING:
    from mace_torch.nn.embedding import RadialEmbeddingBlock

__all__ = ["EdgeInputs"]


class EdgeInputs:
    """The edges' inputs to the interactions, for one forward.

    Built by the backbone and never by a block. Read-only for its consumers:
    the cached values are shared by every layer.

    Args:
        vectors: ``[n_edges, 3]`` in Angstrom, receiver minus sender plus the
            periodic shift. A strain applied upstream is already in them.
        node_atomic_numbers: ``[n_nodes]``, int64, which a distance transform
            reads through each pair's covalent radii.
        edge_index: ``[2, n_edges]``, int64, row 0 the sender.
        harmonics: The spherical harmonics op, ``[n_edges, 3]`` directions in.
        radial_embedding: The radial embedding block, which applies the
            envelope, the distance transform and the basis.
        differentiated_twice: The derivative engine's Python flag for force
            training or a Hessian; independent of the module's training mode.
        receiver_runs: The receiver grouping requested by the backend, if any.
        sender_runs: The sender grouping requested by the backend, if any.
    """

    def __init__(
        self,
        vectors: Tensor,
        node_atomic_numbers: Tensor,
        edge_index: Tensor,
        harmonics: nn.Module,
        radial_embedding: nn.Module,
        *,
        differentiated_twice: bool = False,
        receiver_runs: EdgeRuns | None = None,
        sender_runs: EdgeRuns | None = None,
    ) -> None:
        self.vectors = vectors
        self.node_atomic_numbers = node_atomic_numbers
        self.edge_index = edge_index
        self._harmonics = harmonics
        self._radial_embedding = radial_embedding
        self.differentiated_twice = differentiated_twice
        self.receiver_runs = receiver_runs
        self.sender_runs = sender_runs
        self._lengths: Tensor | None = None
        self._attributes: Tensor | None = None
        self._radial: Tensor | None = None
        self._envelope: Tensor | None = None

    def to(self, dtype: torch.dtype) -> EdgeInputs:
        """A span's precision boundary; no derived per-edge inputs are evaluated."""
        return EdgeInputs(
            self.vectors.to(dtype),
            self.node_atomic_numbers,
            self.edge_index,
            self._harmonics,
            self._radial_embedding,
            differentiated_twice=self.differentiated_twice,
            receiver_runs=self.receiver_runs,
            sender_runs=self.sender_runs,
        )

    def span_weights(self) -> tuple[Tensor, ...]:
        """Live radial basis/transform tensors; this does not evaluate the basis."""
        return cast("RadialEmbeddingBlock", self._radial_embedding).span_weights()

    @property
    def lengths(self) -> Tensor:
        """``[n_edges, 1]``, in Angstrom."""
        if self._lengths is None:
            self._lengths = self.vectors.norm(dim=-1, keepdim=True)
        return self._lengths

    @property
    def attributes(self) -> Tensor:
        """``[n_edges, (lmax + 1)^2]``, the spherical harmonics of the directions."""
        if self._attributes is None:
            self._attributes = self._harmonics(self.vectors)
        return self._attributes

    @property
    def radial(self) -> Tensor:
        """``[n_edges, num_basis]``, the radial embedding.

        The basis on the transformed lengths, times the cutoff envelope on the
        raw ones when the model applies it there.
        """
        if self._radial is None:
            self._build_radial()
        assert self._radial is not None
        return self._radial

    @property
    def envelope(self) -> Tensor:
        """``[n_edges, 1]``, the cutoff envelope on the raw lengths."""
        if self._envelope is None:
            self._build_radial()
        assert self._envelope is not None
        return self._envelope

    def _build_radial(self) -> None:
        self._radial, self._envelope = self._radial_embedding(
            self.lengths, self.node_atomic_numbers, self.edge_index
        )

    def built(self) -> frozenset[str]:
        """Which derived quantities this forward has computed so far."""
        made = {
            "lengths": self._lengths,
            "attributes": self._attributes,
            "radial": self._radial,
        }
        return frozenset(name for name, value in made.items() if value is not None)
