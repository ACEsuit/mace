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

**The graph.** A flat dict, read and never written. The backbone reads
``positions``, ``node_atomic_numbers``, ``edge_index`` and ``shifts``, plus one
key per declared node input. The element one-hot is formed here from the atomic
numbers, and the edge vectors from the positions and the shifts, so a strain
applied to both upstream reaches every edge quantity.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import SupportsIndex, assert_never

import torch
from mace_core.clebsch_gordan.irreps import Irrep, Irreps
from mace_core.config.model import (
    DistanceTransformConfig,
    ModelConfig,
    NoDistanceTransformConfig,
)
from mace_core.elements import MAX_ATOMIC_NUMBER, model_atomic_number_table
from mace_core.kernels.precision import Precision
from mace_core.kernels.specs import (
    DistanceTransformSpec,
    KernelSpec,
    LinearSpec,
    RadialBasisSpec,
    SphericalHarmonicsSpec,
)
from mace_core.observables import InputSpec
from torch import Tensor, nn

from mace_torch.backends import require_built
from mace_torch.backends.reference import TORCH_DTYPES
from mace_torch.nn.edges import EdgeInputs
from mace_torch.nn.embedding import (
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
    "GRAPH_KEYS",
    "GhostExchange",
    "LocalityInfo",
    "MACEBackbone",
    "NodeFeatures",
]

#: The graph keys every backbone reads, whatever it declares.
GRAPH_KEYS = frozenset({"positions", "node_atomic_numbers", "edge_index", "shifts"})

#: One past the largest atomic number, the length of the element lookup.
_ATOMIC_NUMBER_COUNT = MAX_ATOMIC_NUMBER + 1


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
            by irrep in the canonical ``mul_ir``. The last layer carries only
            scalars. Under a :class:`LocalityInfo`, the local nodes only.
        node_attrs: ``[n_nodes, n_elements]``, the element one-hot the
            backbone formed, in the model's precision. What reads the elements
            after the backbone reads this one and forms none of its own. Under
            a :class:`LocalityInfo`, the local nodes only.
    """

    layers: tuple[Tensor, ...]
    node_attrs: Tensor


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


def _check_node_inputs(node_inputs: Sequence[InputSpec]) -> None:
    """Refuse a node input the backbone cannot mix in.

    Raises:
        ValueError: If one is per structure, takes a name the backbone already
            reads, or is declared twice.
    """
    names = [spec.name for spec in node_inputs]
    for spec in node_inputs:
        if not spec.per_atom:
            raise ValueError(
                f"node input {spec.name!r} is declared per structure; the "
                f"backbone mixes in per-atom inputs only."
            )
        if spec.name in GRAPH_KEYS:
            raise ValueError(
                f"node input {spec.name!r} names a key the backbone already "
                f"reads, {sorted(GRAPH_KEYS)}; declare it under its own name."
            )
        if names.count(spec.name) > 1:
            raise ValueError(f"node input {spec.name!r} is declared twice.")


def _element_table(atomic_numbers: Iterable[SupportsIndex]) -> tuple[int, ...]:
    """The element table, in column order, with the argument named on refusal."""
    try:
        return model_atomic_number_table(atomic_numbers).zs
    except ValueError as error:
        raise ValueError(f"atomic_numbers: {error}") from None


def _element_index_of(atomic_numbers: tuple[int, ...]) -> Tensor:
    """``[119]``, int64: each atomic number's column of the element one-hot.

    Zero, a padding atom, and any number outside the table map to the first
    column; the data checks refuse an unknown element before it reaches a
    model.
    """
    element_index_of = torch.zeros(_ATOMIC_NUMBER_COUNT, dtype=torch.int64)
    element_index_of[list(atomic_numbers)] = torch.arange(len(atomic_numbers))
    return element_index_of


def _embedding_irreps(node_inputs: Sequence[InputSpec]) -> str:
    """One channel of the node embedding: ``0e`` and every irrep an input adds.

    An equivariant map cannot create an irrep its input lacks, so the first
    layer has to read every irrep the declared inputs carry.
    """
    terms = ["0e"]
    for spec in node_inputs:
        for _, irrep in Irreps.parse(spec.irreps).terms:
            if str(irrep) not in terms:
                terms.append(str(irrep))
    return "+".join(terms)


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
    * **Nodes.** The element one-hot and the declared node inputs, each mapped
      by an equivariant linear onto the first layer's features.
    * **Layers.** ``num_interactions`` pairs of an interaction block and a
      product basis block. Every layer's features are kept: they are the
      descriptors.

    Args:
        backend: The kernel backend. Consulted at construction only.
        config: The architecture. Its ``clebsch_gordan_basis`` reaches every
            symmetric contraction's specification unchanged.
        atomic_numbers: The element table, the columns of the element one-hot.
            Atomic number 0, a padding atom, takes the first column. The state
            dict does not hold it: the model's metadata records it, as
            ``ModelMetadata.atomic_numbers``.
        avg_num_neighbors: The dataset's average neighbour count, the
            normalization of every block without a learned density.
        precision: The dtype name every op is built and computes in.
        node_inputs: Declared per-atom inputs mixed into the features before
            the first layer, each read from the graph by its name and brought
            in by an equivariant linear map from its declared irreps. Nothing
            about any of them is special-cased.
        ghost_exchange: The exchange that fills the ghosts' features between
            layers of a decomposed run. Absent by default, and when absent a
            forward without a :class:`LocalityInfo` is exactly the hook-free
            one.
    """

    #: Atomic number to column of the element one-hot, ``[119]``, int64.
    element_index_of: Tensor

    def __init__(
        self,
        backend,
        config: ModelConfig,
        atomic_numbers: Iterable[SupportsIndex],
        avg_num_neighbors: float = 1.0,
        precision: Precision = "float64",
        node_inputs: Sequence[InputSpec] = (),
        ghost_exchange: GhostExchange | None = None,
    ) -> None:
        super().__init__()
        _check_node_inputs(node_inputs)
        self.config = config
        self.atomic_numbers = _element_table(atomic_numbers)
        self.register_buffer(
            "element_index_of",
            _element_index_of(self.atomic_numbers),
            persistent=False,
        )
        self.precision = precision
        self.ghost_exchange = ghost_exchange
        self.node_input_names = tuple(spec.name for spec in node_inputs)
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
        embedding_irreps = _embedding_irreps(node_inputs)
        embedding_declaration = str(
            Irreps.parse(embedding_irreps).with_channels(self.num_features)
        )
        self.node_embedding = self._build_linear(
            backend, f"{len(self.atomic_numbers)}x0e", embedding_declaration
        )
        self.node_input_maps = nn.ModuleList(
            self._build_linear(backend, spec.irreps, embedding_declaration)
            for spec in node_inputs
        )

        # Layers. The first reads the embedding, every later one the hidden
        # irreps, and the last returns scalars only.
        hidden_irreps = str(Irreps.spherical_harmonics(config.max_L))
        last = config.num_interactions - 1
        interactions, products, layer_irreps = [], [], []
        for layer in range(config.num_interactions):
            irreps_in = embedding_irreps if layer == 0 else hidden_irreps
            irreps_out = "0e" if layer == last else hidden_irreps
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

    def _build_linear(self, backend, irreps_in: str, irreps_out: str) -> nn.Module:
        """An equivariant linear between two whole declarations."""
        return backend.make_linear(
            LinearSpec(
                irreps_in=irreps_in, irreps_out=irreps_out, precision=self.precision
            )
        )

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
            "num_elements": len(self.atomic_numbers),
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
            num_elements=1 if config.use_agnostic_product else len(self.atomic_numbers),
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
        element = self.element_index_of[graph["node_atomic_numbers"]]
        element_one_hot = nn.functional.one_hot(
            element, num_classes=len(self.atomic_numbers)
        ).to(TORCH_DTYPES[self.precision])
        features = self._embed_nodes(graph, element_one_hot)

        # Layers.
        product_element = (
            torch.zeros_like(element) if self.config.use_agnostic_product else element
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
                features, edges, element_one_hot, sender, receiver, num_nodes
            )
            features = product(
                message[local],
                product_element[local],
                None if carried is None else carried[local],
            )
            layers.append(features)
        return NodeFeatures(layers=tuple(layers), node_attrs=element_one_hot[local])

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

    def _embed_nodes(
        self, graph: Mapping[str, Tensor], element_one_hot: Tensor
    ) -> Tensor:
        """The first layer's node features: the elements plus every node input."""
        features = self.node_embedding(element_one_hot)
        for name, mapping in zip(
            self.node_input_names, self.node_input_maps, strict=True
        ):
            features = features + mapping(graph[name])
        return features

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
            invariants_only: Keep only each layer's scalar channels. The last
                layer carries nothing else, which is why its width differs.

        Returns:
            ``[n_nodes, width]``, the kept layers side by side. With
            ``invariants_only`` the width is ``num_channels`` per layer.
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
            assert irrep.degree == 0 and irrep.parity == 1, irreps
            pieces.append(layer[:, :multiplicity])
        return torch.cat(pieces, dim=-1)
