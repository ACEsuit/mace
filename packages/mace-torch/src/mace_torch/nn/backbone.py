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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.config.model import (
    DistanceTransformConfig,
    InteractionKind,
    ModelConfig,
    NoDistanceTransformConfig,
)
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

from mace_torch.backends.layout import expanded_irreps
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
    "Aggregation",
    "BetweenLayers",
    "LocalityInfo",
    "MACEBackbone",
    "NodeFeatures",
    "aggregate_atomic_descriptors",
]

#: The graph keys every backbone reads, whatever it declares.
GRAPH_KEYS = frozenset({"positions", "node_atomic_numbers", "edge_index", "shifts"})

#: One past the largest atomic number, the length of the element lookup.
_ATOMIC_NUMBER_COUNT = 119


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
BetweenLayers = Callable[[Tensor, LocalityInfo], Tensor]


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


#: How :func:`aggregate_atomic_descriptors` reduces per structure.
Aggregation = Literal["mean", "per_element_mean"] | None


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


def _natural_irreps(lmax: int) -> str:
    """``0e+1o+2e+...`` up to ``lmax``: one copy of each degree, natural parity."""
    return "+".join(
        f"{degree}{'e' if degree % 2 == 0 else 'o'}" for degree in range(lmax + 1)
    )


def _per_channel(irreps: str) -> tuple[str, int]:
    """A whole declaration as one channel's and the channel count.

    Raises:
        ValueError: If its terms do not share one multiplicity, which a
            convolution with one radial weight per path and channel needs.
    """
    terms = Irreps.parse(irreps).terms
    multiplicities = {mul for mul, _ in terms}
    if len(multiplicities) != 1:
        raise ValueError(
            f"edge_irreps={irreps!r}; every term needs the same multiplicity, "
            f"such as '128x0e+128x1o'."
        )
    return "+".join(str(ir) for _, ir in terms), multiplicities.pop()


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


class MACEBackbone(nn.Module):
    """The equivariant message-passing backbone, a learned descriptor.

    Node embedding, radial embedding and spherical harmonics, then
    ``num_interactions`` pairs of an interaction block and a product basis
    block. Every layer's features are kept: they are the descriptors.

    Args:
        backend: The kernel backend. Consulted at construction only.
        config: The architecture. Its ``clebsch_gordan_basis`` reaches every
            symmetric contraction's specification unchanged.
        atomic_numbers: The element table, the columns of the element one-hot.
            Atomic number 0, a padding atom, takes the first column.
        avg_num_neighbors: The dataset's average neighbour count, the
            normalization of every block without a learned density.
        precision: The dtype name every op is built and computes in.
        node_inputs: Declared per-atom inputs mixed into the features before
            the first layer, each read from the graph by its name and brought
            in by an equivariant linear map from its declared irreps. Nothing
            about any of them is special-cased.
        between_layers: The exchange that fills the ghosts' features between
            layers of a decomposed run. Absent by default, and when absent a
            forward without a :class:`LocalityInfo` is exactly the hook-free
            one.
    """

    def __init__(
        self,
        backend,
        config: ModelConfig,
        atomic_numbers: Sequence[int],
        avg_num_neighbors: float = 1.0,
        precision: Precision = "float64",
        node_inputs: Sequence[InputSpec] = (),
        between_layers: BetweenLayers | None = None,
    ) -> None:
        super().__init__()
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
        self.config = config
        self.atomic_numbers = tuple(atomic_numbers)
        in_range = all(1 <= number < _ATOMIC_NUMBER_COUNT for number in atomic_numbers)
        if not in_range or len(set(self.atomic_numbers)) != len(self.atomic_numbers):
            raise ValueError(
                f"atomic_numbers={list(self.atomic_numbers)} is not an element "
                f"table: every entry is an atomic number from 1 to "
                f"{_ATOMIC_NUMBER_COUNT - 1}, each once."
            )
        # Atomic number -> column of the one-hot. Zero, a padding atom, and
        # any number outside the table map to the first column; the data
        # checks refuse an unknown element before it reaches a model.
        element_index_of = torch.zeros(_ATOMIC_NUMBER_COUNT, dtype=torch.int64)
        element_index_of[list(self.atomic_numbers)] = torch.arange(
            len(self.atomic_numbers)
        )
        self.register_buffer("element_index_of", element_index_of, persistent=False)
        self.precision = precision
        self.between_layers = between_layers
        self.node_input_names = tuple(spec.name for spec in node_inputs)
        num_elements = len(self.atomic_numbers)
        num_features = config.num_channels
        self.num_features = num_features

        harmonics = SphericalHarmonicsSpec(lmax=config.max_ell, precision=precision)
        radial = radial_basis_spec(
            config.radial_basis,
            config.cutoff,
            config.r_max,
            apply_cutoff=config.apply_cutoff,
            precision=precision,
        )
        self.spherical_harmonics = backend.make_spherical_harmonics(harmonics)
        # The distance transform is the model's own module, not an op, so it
        # takes the precision from the default dtype it is built under.
        with _default_dtype(TORCH_DTYPES[precision]):
            self.radial_embedding = RadialEmbeddingBlock(
                backend.make_radial_basis(radial),
                build_distance_transform(config.distance_transform),
            )
        num_radial = self.radial_embedding.num_basis
        edge_members = _edge_members(
            radial, harmonics, config.distance_transform, precision
        )

        # The embedding is scalars, plus whatever irreps the declared node
        # inputs carry: an equivariant map cannot create an irrep its input
        # lacks, so the first layer reads them all.
        embedding_terms = ["0e"]
        for spec in node_inputs:
            for _, irrep in Irreps.parse(spec.irreps).terms:
                if str(irrep) not in embedding_terms:
                    embedding_terms.append(str(irrep))
        embedding_irreps = "+".join(embedding_terms)
        embedding_flat = expanded_irreps(embedding_irreps, num_features)
        self.node_embedding = backend.make_linear(
            LinearSpec(
                irreps_in=f"{num_elements}x0e",
                irreps_out=embedding_flat,
                precision=precision,
            )
        )
        self.node_input_maps = nn.ModuleList(
            backend.make_linear(
                LinearSpec(
                    irreps_in=spec.irreps,
                    irreps_out=embedding_flat,
                    precision=precision,
                )
            )
            for spec in node_inputs
        )

        hidden = _natural_irreps(config.max_L)
        target = _natural_irreps(config.max_ell)
        product_elements = 1 if config.use_agnostic_product else num_elements
        self.use_agnostic_product = config.use_agnostic_product
        interactions, products, layer_irreps = [], [], []
        for layer in range(config.num_interactions):
            first = layer == 0
            irreps_node = embedding_irreps if first else hidden
            irreps_out = "0e" if layer == config.num_interactions - 1 else hidden
            kind = config.interaction_first if first else config.interaction
            interactions.append(
                self._interaction(
                    backend,
                    kind,
                    first,
                    irreps_node=irreps_node,
                    irreps_edge=target,
                    irreps_skip_out=irreps_out,
                    num_radial=num_radial,
                    num_elements=num_elements,
                    avg_num_neighbors=avg_num_neighbors,
                    edge_members=edge_members,
                )
            )
            products.append(
                EquivariantProductBasisBlock(
                    backend,
                    irreps_in=target,
                    irreps_out=irreps_out,
                    correlation=config.correlation,
                    num_elements=product_elements,
                    num_features=num_features,
                    basis=config.clebsch_gordan_basis,
                    precision=precision,
                )
            )
            layer_irreps.append(expanded_irreps(irreps_out, num_features))
        self.interactions = nn.ModuleList(interactions)
        self.products = nn.ModuleList(products)
        #: Each layer's output declaration, grouped.
        self.layer_irreps = tuple(layer_irreps)

    def _interaction(
        self,
        backend,
        kind: InteractionKind,
        first: bool,
        *,
        irreps_node: str,
        irreps_edge: str,
        irreps_skip_out: str,
        num_radial: int,
        num_elements: int,
        avg_num_neighbors: float,
        edge_members: tuple[KernelSpec, ...] | None,
    ) -> nn.Module:
        """The interaction block a layer's kind names."""
        config = self.config
        common = {
            "irreps_node": irreps_node,
            "irreps_edge": irreps_edge,
            "irreps_target": irreps_edge,
            "num_radial": num_radial,
            "num_features": self.num_features,
            "num_elements": num_elements,
            "radial_hidden": tuple(config.radial_mlp),
            "precision": self.precision,
            "envelope_outside": not config.apply_cutoff,
            "edge_members": edge_members,
        }
        if kind == "nonlinear":
            irreps_up, num_up_features = irreps_node, self.num_features
            if config.edge_irreps is not None and (
                not first or config.use_edge_irreps_first
            ):
                irreps_up, num_up_features = _per_channel(config.edge_irreps)
                if first:
                    irreps_up = "0e"
                    num_up_features = sum(
                        mul
                        for mul, ir in Irreps.parse(config.edge_irreps).terms
                        if ir.degree == 0 and ir.parity == 1
                    )
            return NonLinearInteractionBlock(
                backend,
                irreps_up=irreps_up,
                irreps_skip_out=irreps_skip_out,
                num_up_features=num_up_features,
                **common,
            )
        if kind not in ("plain", "residual", "density", "density_residual"):
            raise ValueError(
                f"interaction kind {kind!r} is not one this builds; the kinds are "
                f"'plain', 'residual', 'density', 'density_residual' and "
                f"'nonlinear'."
            )
        learned_density = kind in ("density", "density_residual")
        if kind in ("plain", "density"):
            return InteractionBlock(
                backend,
                avg_num_neighbors=avg_num_neighbors,
                learned_density=learned_density,
                **common,
            )
        return ResidualInteractionBlock(
            backend,
            irreps_skip_out=irreps_skip_out,
            avg_num_neighbors=avg_num_neighbors,
            learned_density=learned_density,
            **common,
        )

    def forward(
        self, graph: Mapping[str, Tensor], locality: LocalityInfo | None = None
    ) -> NodeFeatures:
        """Every layer's node features.

        Args:
            graph: The flat dict. Read, never written.
            locality: Which trailing node slots are ghosts, for one domain of
                a decomposed run. ``None`` for a whole system.
        """
        positions = graph["positions"]
        element_index_of = cast(Tensor, self.element_index_of)
        node_element = element_index_of[graph["node_atomic_numbers"]]
        node_attrs = nn.functional.one_hot(
            node_element, num_classes=len(self.atomic_numbers)
        ).to(TORCH_DTYPES[self.precision])
        edge_index = graph["edge_index"]
        sender, receiver = edge_index[0], edge_index[1]
        num_nodes = positions.shape[0]
        if (
            locality is not None
            and self.between_layers is None
            and len(self.interactions) > 1
        ):
            raise ValueError(
                f"a LocalityInfo was given to a backbone of "
                f"{len(self.interactions)} layers built without between_layers. "
                f"Past the first layer the ghosts' features have to be exchanged, "
                f"and nothing here can do that; build the backbone with the "
                f"exchange its runtime provides."
            )

        vectors = (
            positions.index_select(0, receiver)
            - positions.index_select(0, sender)
            + graph["shifts"]
        )
        edges = EdgeInputs(
            vectors,
            graph["node_atomic_numbers"],
            edge_index,
            self.spherical_harmonics,
            self.radial_embedding,
        )
        element = (
            torch.zeros_like(node_element)
            if self.use_agnostic_product
            else node_element
        )

        features = self.node_embedding(node_attrs)
        for name, mapping in zip(
            self.node_input_names, self.node_input_maps, strict=True
        ):
            features = features + mapping(graph[name])

        layers = []
        local_node_attrs = (
            node_attrs if locality is None else node_attrs[: locality.n_local]
        )
        for layer, (interaction, product) in enumerate(
            zip(self.interactions, self.products, strict=True)
        ):
            if locality is not None and layer > 0:
                assert self.between_layers is not None
                features = self.between_layers(features, locality)
            message, carried = interaction(
                features, edges, node_attrs, sender, receiver, num_nodes
            )
            layer_element = element
            if locality is not None:
                # The messages of the ghosts are incomplete: their neighbours
                # live in another domain. Only the local nodes go on.
                message = message[: locality.n_local]
                carried = None if carried is None else carried[: locality.n_local]
                layer_element = element[: locality.n_local]
            features = product(message, layer_element, carried)
            layers.append(features)
        return NodeFeatures(layers=tuple(layers), node_attrs=local_node_attrs)

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


def aggregate_atomic_descriptors(
    atomic_descriptors: Tensor,
    batch: Tensor,
    num_graphs: int,
    aggregation: Aggregation,
    element: Tensor | None = None,
    atomic_numbers: Sequence[int] = (),
) -> Tensor | list[dict[int, Tensor]]:
    """Reduce atomic descriptors per structure.

    Args:
        atomic_descriptors: ``[n_nodes, width]``, from
            :meth:`MACEBackbone.atomic_descriptors`.
        batch: ``[n_nodes]``, int64, each node's structure.
        num_graphs: How many structures, ``ptr.numel() - 1``.
        aggregation: ``None`` returns the atomic descriptors as they are. ``"mean"``
            averages each structure's atoms, ``[num_graphs, width]``.
            ``"per_element_mean"`` averages each structure's atoms of each
            element present, one dict per structure keyed by atomic number.
        element: ``[n_nodes]``, int64, each node's column of the element
            table. Needed for ``"per_element_mean"`` only.
        atomic_numbers: The element table. Needed for ``"per_element_mean"``.
    """
    if aggregation is None:
        return atomic_descriptors
    if aggregation == "mean":
        totals = atomic_descriptors.new_zeros(num_graphs, atomic_descriptors.shape[1])
        totals = totals.index_add(0, batch, atomic_descriptors)
        counts = torch.bincount(batch, minlength=num_graphs).to(
            atomic_descriptors.dtype
        )
        return totals / counts.unsqueeze(-1)
    if aggregation == "per_element_mean":
        if element is None or not atomic_numbers:
            raise ValueError(
                "per_element_mean needs each node's element and the element table."
            )
        result: list[dict[int, Tensor]] = []
        for graph in range(num_graphs):
            in_graph = batch == graph
            means = {}
            for column, atomic_number in enumerate(atomic_numbers):
                mask = in_graph & (element == column)
                if bool(mask.any()):
                    means[int(atomic_number)] = atomic_descriptors[mask].mean(dim=0)
            result.append(means)
        return result
    raise ValueError(
        f"aggregation={aggregation!r} is not one this computes. The choices are "
        f"None, 'mean' and 'per_element_mean'."
    )
