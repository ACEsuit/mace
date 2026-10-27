"""What the interactions read off the edges, computed once and only if read.

The backbone forms the edge vectors once, from the positions, the edge index
and the shifts, and hands every interaction block one :class:`EdgeInputs`.
Everything derived from the vectors is computed the first time something reads
it and cached for that forward, so a quantity no op reads is never allocated.
A backend's fused span, for instance, reads the vectors and computes the
harmonics and the radial embedding inside its kernel.
"""

from __future__ import annotations

from torch import Tensor, nn

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
    """

    def __init__(
        self,
        vectors: Tensor,
        node_atomic_numbers: Tensor,
        edge_index: Tensor,
        harmonics: nn.Module,
        radial_embedding: nn.Module,
    ) -> None:
        self.vectors = vectors
        self.node_atomic_numbers = node_atomic_numbers
        self.edge_index = edge_index
        self._harmonics = harmonics
        self._radial_embedding = radial_embedding
        self._lengths: Tensor | None = None
        self._attributes: Tensor | None = None
        self._radial: Tensor | None = None
        self._envelope: Tensor | None = None

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
