"""The backbone: node features in, node features out.

The first of the two layers the architecture is split into. It is a learned
descriptor and nothing else: **no readouts, no heads, no scale and shift, no
gradient calls**. Forces and stress are taken by a derivative engine around it,
which is what keeps a compiled graph whole. The model that holds it beside the
output layer is a separate class.

Four things it does not do, each replacing something the frozen tree does:

* It resolves no backend in ``forward``. Every op is built once from a
  kernel specification and held.
* It branches on no fusion attribute. A block asks the backend for a fused
  span once, at build, and the answer changes nothing it computes here.
* It carries no LAMMPS partitioning through its blocks. Ghost atoms are
  trailing node slots described by a :class:`LocalityInfo` handed to
  ``forward``, and the exchange between layers is one explicit hook, absent by
  default.
* It writes nothing into the graph it is given.

**What it saves** is what :func:`mace_torch.kernels.canonical_state`
collects from it: the element table and the learned weights, and the average
neighbour count in every convolution normalized by it. What the config fixes
is rebuilt from it.

**The graph.** A flat dict, read and never written. The backbone reads
``positions``, ``node_atomic_numbers``, ``edge_index`` and ``shifts``, and
nothing else. The element one-hot is formed here from the atomic numbers, and
the edge vectors from the positions and the shifts, so a strain applied to
both upstream reaches every edge quantity.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import SupportsIndex, assert_never

import torch
from mace_core.clebsch_gordan.irreps import Irrep, Irreps
from mace_core.config.model import (
    DistanceTransformConfig,
    ModelConfig,
    NoDistanceTransformConfig,
)
from mace_core.kernels.precision import Precision
from mace_core.kernels.specs import (
    DistanceTransformSpec,
    KernelSpec,
    RadialBasisSpec,
    SphericalHarmonicsSpec,
)
from torch import Tensor, nn

from mace_torch.backends import require_built
from mace_torch.backends.reference import TORCH_DTYPES
from mace_torch.nn.edges import EdgeInputs
from mace_torch.nn.embedding import (
    ElementEmbedding,
    RadialEmbeddingBlock,
    build_distance_transform,
    radial_basis_spec,
)
from mace_torch.nn.interaction import (
    InteractionBlock,
    NonLinearInteractionBlock,
    ResidualInteractionBlock,
)
from mace_torch.nn.product_basis import EquivariantProductBasisBlock

__all__ = [
    "GhostExchange",
    "LocalityInfo",
    "MACEBackbone",
    "NodeFeatures",
]


@dataclass(frozen=True)
class LocalityInfo:
    """Which node slots are local and which are ghosts, for one call.

    Per-call data, never build-time configuration: the graph is one domain of
    a decomposed run, its first ``n_local`` nodes owned here and the next
    ``n_ghost`` copies of nodes owned by a neighbouring domain.
    """

    n_local: int
    n_ghost: int

    def __post_init__(self) -> None:
        if self.n_local < 1 or self.n_ghost < 0:
            raise ValueError(
                f"LocalityInfo(n_local={self.n_local}, n_ghost={self.n_ghost}): a "
                f"domain owns at least one node and has no fewer than zero ghosts."
            )


#: The exchange between two layers of a decomposed run: the local nodes'
#: features ``[n_local, dim]`` in, every node's ``[n_local + n_ghost, dim]``
#: out, the ghosts filled from the domains that own them.
GhostExchange = Callable[[Tensor, LocalityInfo], Tensor]


@dataclass(frozen=True)
class NodeFeatures:
    """Every layer's node features, the backbone's whole output.

    Attributes:
        layers: One ``[n_nodes, dim_layer]`` tensor per interaction, grouped
            by irrep in the canonical ``mul_ir``. The last layer carries
            ``last_layer_irreps``, its scalars by default. Under a
            :class:`LocalityInfo`, the local nodes only.
        node_attrs: ``[n_nodes, n_elements]``, the element one-hot the
            backbone formed, in the model's precision.
        element_index: ``[n_nodes]``, int64, each node's column of the
            element table, the index the one-hot was formed from.

    What reads the elements after the backbone reads these two and forms no
    lookup of its own. Under a :class:`LocalityInfo`, both hold the local
    nodes only.
    """

    layers: tuple[Tensor, ...]
    node_attrs: Tensor
    element_index: Tensor


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Build modules that take their dtype from the process default in ``dtype``.

    The radial embedding's constants, the Bessel frequencies and the covalent
    radii among them, are computed at construction in the default dtype. A
    float64 model built under a float32 default would compute them in float32
    and cast them, which is a different model from the first digit that
    matters.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


# ---------------------------------------------------------------------------
# Construction helpers. Each answers one question about the configuration.
# ---------------------------------------------------------------------------


def _edge_members(
    radial: RadialBasisSpec,
    harmonics: SphericalHarmonicsSpec,
    transform: DistanceTransformConfig,
    precision: Precision,
) -> tuple[KernelSpec, ...]:
    """The specifications of what every edge convolution starts from.

    The radial basis with its envelope, the harmonics, and the distance
    transform when there is one. The radial basis states whether the envelope
    multiplies the basis or the radial networks' output, so a backend that
    handles only one of the two declines the other.
    """
    members: tuple[KernelSpec, ...] = (radial, harmonics)
    if not isinstance(transform, NoDistanceTransformConfig):
        members += (DistanceTransformSpec(kind=transform.kind, precision=precision),)
    return members


def _split_channels(irreps: str) -> tuple[str, int]:
    """A whole declaration as one channel's and the channel count.

    Its terms share one multiplicity: ``ModelConfig`` refuses ``edge_irreps``
    otherwise.
    """
    terms = Irreps.parse(irreps).terms
    return "+".join(str(ir) for _, ir in terms), terms[0][0]


def _nonlinear_up_projection(
    config: ModelConfig, first: bool, irreps_node: str, num_features: int
) -> tuple[str, int]:
    """What a ``nonlinear`` layer convolves: one channel's irreps and the width.

    Three cases:

    * Without ``edge_irreps``, and in the first layer unless
      ``use_edge_irreps_first`` is set, the node features as they come in.
    * In the first layer with ``use_edge_irreps_first``, the scalars of
      ``edge_irreps`` only.
    * In every later layer, ``edge_irreps`` whole.
    """
    if config.edge_irreps is None or (first and not config.use_edge_irreps_first):
        return irreps_node, num_features
    if first:
        return "0e", Irreps.parse(config.edge_irreps).multiplicity(Irrep(0, 1))
    return _split_channels(config.edge_irreps)


class MACEBackbone(nn.Module):
    """The equivariant message-passing backbone, a learned descriptor.

    Three parts, built in this order in ``__init__`` and run in it in
    :meth:`forward`:

    * **Edges.** The spherical harmonics and the radial embedding of every
      edge vector, computed once and read by every layer.
    * **Nodes.** The element embedding: each node's column of the element
      table, its one-hot, and the learned scalars the first layer reads.
    * **Layers.** ``num_interactions`` pairs of an interaction block and a
      product basis block. Every layer's features are kept: they are the
      descriptors.

    Args:
        backend: The kernel backend. Consulted at construction only.
        config: The architecture. Its ``clebsch_gordan_basis`` reaches every
            symmetric contraction's specification unchanged.
        atomic_numbers: The element table, strictly ascending, the columns of
            the element one-hot. The element embedding holds and saves it, so
            a model built from a saved state reads it from there.
        avg_num_neighbors: The dataset's average neighbour count, the
            normalization of every block without a learned density.
        precision: The dtype name every op is built and computes in.
        ghost_exchange: The exchange that fills the ghosts' features between
            layers of a decomposed run. Absent by default, and when absent a
            forward without a :class:`LocalityInfo` is exactly the hook-free
            one.
    """

    def __init__(
        self,
        backend,
        config: ModelConfig,
        atomic_numbers: Iterable[SupportsIndex],
        avg_num_neighbors: float = 1.0,
        precision: Precision = "float64",
        ghost_exchange: GhostExchange | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.precision = precision
        self.ghost_exchange = ghost_exchange
        self.num_features = config.num_channels

        # Edges.
        harmonics = SphericalHarmonicsSpec(lmax=config.max_ell, precision=precision)
        radial = radial_basis_spec(
            config.radial_basis,
            config.cutoff,
            config.r_max,
            apply_cutoff=config.apply_cutoff,
            precision=precision,
        )
        self.spherical_harmonics = require_built(backend, harmonics)
        # The distance transform is the model's own module, not an op, so it
        # takes the precision from the default dtype it is built under.
        with _default_dtype(TORCH_DTYPES[precision]):
            self.radial_embedding = RadialEmbeddingBlock(
                require_built(backend, radial),
                build_distance_transform(config.distance_transform),
            )
        # What every convolution produces and every product basis reads.
        target_irreps = str(Irreps.spherical_harmonics(config.max_ell))
        edge_members = _edge_members(
            radial, harmonics, config.distance_transform, precision
        )

        # Nodes.
        self.element_embedding = ElementEmbedding(
            atomic_numbers, self.num_features, precision
        )

        # Layers. The first reads the element embedding's scalars, every later
        # one the hidden irreps, and the last returns `last_layer_irreps`.
        hidden_irreps = str(Irreps.spherical_harmonics(config.max_L))
        last = config.num_interactions - 1
        interactions, products, layer_irreps = [], [], []
        for layer in range(config.num_interactions):
            irreps_in = "0e" if layer == 0 else hidden_irreps
            irreps_out = config.last_layer_irreps if layer == last else hidden_irreps
            interactions.append(
                self._build_interaction(
                    backend,
                    layer,
                    irreps_in,
                    irreps_out,
                    target_irreps,
                    edge_members,
                    avg_num_neighbors,
                )
            )
            products.append(self._build_product(backend, target_irreps, irreps_out))
            layer_irreps.append(
                str(Irreps.parse(irreps_out).with_channels(self.num_features))
            )
        self.interactions = nn.ModuleList(interactions)
        self.products = nn.ModuleList(products)
        #: Each layer's output declaration, grouped.
        self.layer_irreps = tuple(layer_irreps)

    # ------------------------------------------------------------------
    # Construction. The builders read what the backbone keeps from ``self``
    # and take what ``__init__`` derives along the way as arguments.
    # ------------------------------------------------------------------

    def _build_interaction(
        self,
        backend,
        layer: int,
        irreps_in: str,
        irreps_out: str,
        target_irreps: str,
        edge_members: tuple[KernelSpec, ...],
        avg_num_neighbors: float,
    ) -> nn.Module:
        """The interaction block of layer ``layer``, of the kind it names.

        Args:
            irreps_in: One channel of the node features the layer reads.
            irreps_out: One channel of the features the layer returns, which
                is what a skip connection carries to the product basis.
            target_irreps: One channel of what the convolution produces.
        """
        config = self.config
        kind = config.interaction_of(layer)
        common = {
            "irreps_node": irreps_in,
            "irreps_edge": target_irreps,
            "irreps_target": target_irreps,
            "num_radial": self.radial_embedding.num_basis,
            "num_features": self.num_features,
            "num_elements": self.element_embedding.num_elements,
            "radial_hidden": tuple(config.radial_mlp),
            "precision": self.precision,
            "envelope_outside": not config.apply_cutoff,
            "edge_members": edge_members,
        }
        if kind == "nonlinear":
            irreps_up, num_up_features = _nonlinear_up_projection(
                config, layer == 0, irreps_in, self.num_features
            )
            return NonLinearInteractionBlock(
                backend,
                irreps_up=irreps_up,
                irreps_skip_out=irreps_out,
                num_up_features=num_up_features,
                **common,
            )
        elif kind in ("plain", "density"):
            return InteractionBlock(
                backend,
                avg_num_neighbors=avg_num_neighbors,
                learned_density=kind == "density",
                **common,
            )
        elif kind in ("residual", "density_residual"):
            return ResidualInteractionBlock(
                backend,
                irreps_skip_out=irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                learned_density=kind == "density_residual",
                **common,
            )
        assert_never(kind)

    def _build_product(
        self, backend, target_irreps: str, irreps_out: str
    ) -> EquivariantProductBasisBlock:
        """The product basis block from ``target_irreps`` to ``irreps_out``."""
        config = self.config
        return EquivariantProductBasisBlock(
            backend,
            irreps_in=target_irreps,
            irreps_out=irreps_out,
            correlation=config.correlation,
            num_elements=(
                1
                if config.use_agnostic_product
                else self.element_embedding.num_elements
            ),
            num_features=self.num_features,
            basis=config.clebsch_gordan_basis,
            precision=self.precision,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, graph: Mapping[str, Tensor], locality: LocalityInfo | None = None
    ) -> NodeFeatures:
        """Every layer's node features.

        Args:
            graph: The flat dict. Read, never written.
            locality: Which trailing node slots are ghosts, for one domain of
                a decomposed run. ``None`` for a whole system.
        """
        self._check_locality(locality)
        # Every node, ghosts included, takes part in a layer's convolution.
        # A ghost's message is incomplete, its neighbours living in another
        # domain, so only the local nodes go on to the product basis.
        local = slice(None) if locality is None else slice(locality.n_local)

        # Edges.
        sender, receiver = graph["edge_index"][0], graph["edge_index"][1]
        edges = self._edge_inputs(graph, sender, receiver)

        # Nodes.
        element = self.element_embedding(graph["node_atomic_numbers"])
        features = element.features

        # Layers.
        product_element = (
            torch.zeros_like(element.index)
            if self.config.use_agnostic_product
            else element.index
        )
        num_nodes = graph["positions"].shape[0]
        layers = []
        for layer, (interaction, product) in enumerate(
            zip(self.interactions, self.products, strict=True)
        ):
            if locality is not None and layer > 0:
                assert self.ghost_exchange is not None
                features = self.ghost_exchange(features, locality)
            message, carried = interaction(
                features, edges, element.one_hot, sender, receiver, num_nodes
            )
            features = product(
                message[local],
                product_element[local],
                None if carried is None else carried[local],
            )
            layers.append(features)
        return NodeFeatures(
            layers=tuple(layers),
            node_attrs=element.one_hot[local],
            element_index=element.index[local],
        )

    def _check_locality(self, locality: LocalityInfo | None) -> None:
        """Refuse a decomposed run that needs an exchange this backbone lacks."""
        if (
            locality is not None
            and self.ghost_exchange is None
            and len(self.interactions) > 1
        ):
            raise ValueError(
                f"a LocalityInfo was given to a backbone of "
                f"{len(self.interactions)} layers built without ghost_exchange. "
                f"Past the first layer the ghosts' features have to be exchanged, "
                f"and nothing here can do that; build the backbone with the "
                f"exchange its runtime provides."
            )

    def _edge_inputs(
        self, graph: Mapping[str, Tensor], sender: Tensor, receiver: Tensor
    ) -> EdgeInputs:
        """The edge vectors, with what every layer derives from them."""
        positions = graph["positions"]
        vectors = positions[receiver] - positions[sender] + graph["shifts"]
        return EdgeInputs(
            vectors,
            graph["node_atomic_numbers"],
            graph["edge_index"],
            self.spherical_harmonics,
            self.radial_embedding,
        )

    # ------------------------------------------------------------------
    # Atomic descriptors
    # ------------------------------------------------------------------

    def atomic_descriptors(
        self,
        features: NodeFeatures,
        num_layers: int = -1,
        invariants_only: bool = True,
    ) -> Tensor:
        """The atomic descriptors, from the layers' node features.

        Args:
            features: What :meth:`forward` returned.
            num_layers: How many layers to keep, from the first. ``-1`` keeps
                every one.
            invariants_only: Keep only each layer's scalar channels. A layer
                without scalars, such as a dipole model's last, gives none.

        Returns:
            ``[n_nodes, width]``, the kept layers side by side. With
            ``invariants_only`` the width is ``num_channels`` per layer that
            has scalars.
        """
        count = len(features.layers) if num_layers == -1 else num_layers
        if not 1 <= count <= len(features.layers):
            raise ValueError(
                f"num_layers={num_layers} is not a layer count of this backbone; "
                f"it is -1 or between 1 and {len(features.layers)}."
            )
        pieces = []
        for layer, irreps in zip(
            features.layers[:count], self.layer_irreps, strict=False
        ):
            if not invariants_only:
                pieces.append(layer)
                continue
            multiplicity, irrep = Irreps.parse(irreps).terms[0]
            if irrep.degree == 0 and irrep.parity == 1:
                pieces.append(layer[:, :multiplicity])
        if not pieces:
            raise ValueError(
                f"none of the first {count} layer(s) carries scalars, "
                f"{list(self.layer_irreps[:count])}; ask for more layers or "
                f"for invariants_only=False."
            )
        return torch.cat(pieces, dim=-1)
