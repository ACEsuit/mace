"""Node and edge embeddings: the element embedding and the radial embedding.

Both are the first thing a MACE model does with a graph. The element embedding
maps each atom's element to a vector of scalar channels; the radial embedding
maps each edge's length to a vector of radial features and the smooth cutoff
envelope that goes with them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import SupportsIndex, cast

import torch
from mace_core.config import (
    AgnesiTransformConfig,
    BesselBasisConfig,
    ChebyshevBasisConfig,
    DistanceTransformConfig,
    GaussianBasisConfig,
    NoDistanceTransformConfig,
    PolynomialCutoffConfig,
    RadialBasisConfig,
    SoftTransformConfig,
)
from mace_core.elements import model_atomic_number_table
from mace_core.kernels.precision import Precision
from mace_core.kernels.specs import RadialBasisSpec
from torch import Tensor

from mace_torch.backends.reference import TORCH_DTYPES
from mace_torch.kernels.initialization import standard_normal_like
from mace_torch.nn.radial import (
    AgnesiTransform,
    SoftTransform,
)

__all__ = [
    "DistanceTransform",
    "ElementEmbedding",
    "ElementEncoding",
    "RadialEmbeddingBlock",
    "build_distance_transform",
    "radial_basis_spec",
]

DistanceTransform = AgnesiTransform | SoftTransform


@dataclass(frozen=True)
class ElementEncoding:
    """Each node's element, in the three forms a model reads it in.

    Attributes:
        index: ``[n_nodes]``, int64, each node's column of the element table.
        one_hot: ``[n_nodes, n_elements]``, the index as a one-hot, in the
            model's precision.
        features: ``[n_nodes, num_channels]``, the learned embedding of each
            node's element, the scalars the first layer reads.
    """

    index: Tensor
    one_hot: Tensor
    features: Tensor


class ElementEmbedding(torch.nn.Module):
    """The model's element table and the learned embedding of each element.

    It saves both: the table, ascending, is fitted from the data or inherited
    from a foundation model, and the embedding is learned. A node's column is
    found by searching the table, so the table this module holds, built with or
    loaded, is the one every forward uses, and nothing is derived from it and
    kept. Atomic number 0, a padding atom, sorts below every element and takes
    the first column.

    The embedding is ``one_hot @ weight / sqrt(n_elements)``, legacy's
    equivariant linear map on scalars with its fan-in applied in the forward.

    Args:
        atomic_numbers: The element table, strictly ascending.
        num_channels: The width of each element's embedding.
        precision: The dtype name the embedding and the one-hot are in.
    """

    #: ``[n_elements]``, int64, the element table.
    atomic_numbers: Tensor

    def __init__(
        self,
        atomic_numbers: Iterable[SupportsIndex],
        num_channels: int,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        try:
            table = model_atomic_number_table(atomic_numbers).zs
        except ValueError as error:
            raise ValueError(f"atomic_numbers: {error}") from None
        self.num_elements = len(table)
        self.normalization = 1.0 / math.sqrt(self.num_elements)
        self.register_buffer("atomic_numbers", torch.tensor(table, dtype=torch.int64))
        self.weight = torch.nn.Parameter(
            torch.zeros(self.num_elements, num_channels, dtype=TORCH_DTYPES[precision])
        )

    def forward(self, node_atomic_numbers: Tensor) -> ElementEncoding:
        """``node_atomic_numbers`` is ``[n_nodes]``, int64: 0 for a padding
        atom, an element of the table otherwise."""
        index = torch.searchsorted(self.atomic_numbers, node_atomic_numbers)
        one_hot = torch.nn.functional.one_hot(index, self.num_elements).to(
            self.weight.dtype
        )
        features = (one_hot @ self.weight) * self.normalization
        return ElementEncoding(index=index, one_hot=one_hot, features=features)

    def initialize_weights(self, seed: int) -> None:
        """A standard normal, unscaled, since the forward applies the fan-in."""
        with torch.no_grad():
            self.weight.copy_(standard_normal_like(self.weight, seed))

    def to_canonical(self) -> dict[str, Tensor]:
        """The element table and the embedding, ``[n_elements, num_channels]``,
        its rows in the table's order."""
        return {
            "atomic_numbers": self.atomic_numbers.detach(),
            "weight": self.weight.detach(),
        }

    def load_canonical(self, state: Mapping[str, Tensor]) -> None:
        """Put a saved table and embedding back.

        Raises:
            ValueError: If the saved table is not a model's: a forward would
                search it and find the wrong columns. Nothing of this module
                is loaded then.
        """
        try:
            model_atomic_number_table(state["atomic_numbers"].tolist())
        except ValueError as error:
            raise ValueError(f"atomic_numbers: {error}") from None
        with torch.no_grad():
            self.atomic_numbers.copy_(state["atomic_numbers"])
            self.weight.copy_(state["weight"])

    def extra_repr(self) -> str:
        return (
            f"atomic_numbers={self.atomic_numbers.tolist()}, "
            f"num_channels={self.weight.shape[1]}"
        )


class RadialEmbeddingBlock(torch.nn.Module):
    """The radial basis and the cutoff envelope of every edge, with an optional
    distance transform.

    The basis and its envelope are one op, the one the backend built from a
    :class:`~mace_core.kernels.specs.RadialBasisSpec`; the block adds the
    distance transform, which the model holds, and hands the envelope on to the
    interaction blocks.

    The forward runs three steps in a fixed order that is load-bearing:

    1. the cutoff envelope is computed on the **raw** edge lengths;
    2. the distance transform, if there is one, is applied to the lengths;
    3. the basis is evaluated on the (transformed) lengths.

    Both results are returned, always: the radial features ``[n_edges, num_basis]``
    and the envelope ``[n_edges, 1]``. With ``apply_cutoff`` (legacy's
    ``--apply_cutoff``, the default) the features are already ``basis * cutoff``;
    without it they are the bare basis and the consumer multiplies the envelope
    in later, after its radial MLP.
    """

    def __init__(
        self,
        radial_basis: torch.nn.Module,
        distance_transform: DistanceTransform | None,
    ):
        super().__init__()
        self.radial = radial_basis
        self.distance_transform = distance_transform
        self.kernel_spec = cast(RadialBasisSpec, radial_basis.kernel_spec)

    @property
    def num_basis(self) -> int:
        return self.kernel_spec.num_basis

    @property
    def apply_cutoff(self) -> bool:
        return self.kernel_spec.apply_cutoff

    @property
    def cutoff(self) -> torch.nn.Module:
        """The envelope, which the forward evaluates on the raw lengths."""
        return cast(torch.nn.Module, self.radial.cutoff)

    def forward(
        self,
        edge_lengths: torch.Tensor,
        node_atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``edge_lengths`` is ``[n_edges, 1]`` in Angstrom.

        Returns ``(edge_radial_features, edge_cutoff)``: the basis on the
        (transformed) lengths, ``[n_edges, num_basis]``, multiplied by the
        envelope when ``apply_cutoff`` is set, and the envelope on the raw
        lengths, ``[n_edges, 1]``.
        """
        edge_cutoff = self.cutoff(edge_lengths)  # on the raw lengths, always
        if self.distance_transform is not None:
            edge_lengths = self.distance_transform(
                edge_lengths, node_atomic_numbers, edge_index
            )
        edge_radial_features = self.radial(edge_lengths, envelope=edge_cutoff)
        return edge_radial_features, edge_cutoff

    def extra_repr(self) -> str:
        return f"apply_cutoff={self.apply_cutoff}"


# ---------------------------------------------------------------------------
# Config section -> specification or module
# ---------------------------------------------------------------------------


def radial_basis_spec(
    radial_basis: RadialBasisConfig,
    cutoff: PolynomialCutoffConfig,
    r_max: float,
    *,
    apply_cutoff: bool = True,
    precision: Precision = "float64",
) -> RadialBasisSpec:
    """The kernel specification of the basis and envelope the sections describe.

    ``r_max`` is in Angstrom and sets both; the Chebyshev basis does not use it.
    Every option of the basis section reaches the specification, so a backend
    sees the whole radial embedding it is asked to build.
    """
    return RadialBasisSpec(
        kind=radial_basis.kind,
        num_basis=radial_basis.num_basis,
        cutoff=r_max,
        cutoff_order=cutoff.polynomial_order,
        trainable=(
            radial_basis.trainable
            if isinstance(radial_basis, BesselBasisConfig | GaussianBasisConfig)
            else False
        ),
        include_constant=(
            radial_basis.include_constant
            if isinstance(radial_basis, ChebyshevBasisConfig)
            else False
        ),
        apply_cutoff=apply_cutoff,
        precision=precision,
    )


def build_distance_transform(
    config: DistanceTransformConfig,
) -> DistanceTransform | None:
    """The transform a config section describes; ``None`` for the kind ``none``."""
    match config:
        case NoDistanceTransformConfig():
            return None
        case AgnesiTransformConfig():
            return AgnesiTransform(
                exponent_q=config.exponent_q,
                exponent_p=config.exponent_p,
                amplitude=config.amplitude,
                trainable=config.trainable,
            )
        case SoftTransformConfig():
            return SoftTransform(steepness=config.steepness, trainable=config.trainable)
        case _:
            raise TypeError(
                f"no distance transform is built from {type(config).__name__}"
            )
