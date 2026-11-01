"""The backbone: equivariance, a read-only forward, padding, locality,
descriptors, compile, and what it asks of the backend.

Every test builds through the reference backend and runs once per dtype, the
model built in the dtype the suite is running in.
"""

import inspect
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from conftest import assert_close, fp64_only
from mace_core.config import AgnesiTransformConfig, ModelConfig
from mace_core.kernels import (
    BackendCapabilities,
    ChannelwiseTPConvDescriptor,
    DistanceTransformDescriptor,
    Precision,
    RadialBasisDescriptor,
    RadialMLPDescriptor,
    SphericalHarmonicsDescriptor,
    UnsupportedDescriptorError,
)
from mace_core.observables import InputSpec
from mace_torch.backends.harmonics import spherical_harmonics
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.kernels import initialize_model_weights
from mace_torch.nn import backbone as backbone_module
from mace_torch.nn.backbone import (
    LocalityInfo,
    MACEBackbone,
    NodeFeatures,
    aggregate_descriptors,
)
from mace_torch.nn.edges import EdgeInputs
from mace_torch.nn.interaction import (
    InteractionBlock,
    NonLinearInteractionBlock,
    ResidualInteractionBlock,
)
from mace_torch.nn.product_basis import EquivariantProductBasisBlock

ELEMENTS = (1, 6, 8)
R_MAX = 3.0
NN = Path(backbone_module.__file__).parent

#: Every interaction variant in scope, as the two layer kinds that build it.
VARIANTS: dict[str, dict[str, Any]] = {
    "plain_then_residual": {"interaction_first": "plain", "interaction": "residual"},
    "residual": {"interaction_first": "residual", "interaction": "residual"},
    "density": {"interaction_first": "density", "interaction": "density_residual"},
    "nonlinear": {
        "interaction_first": "nonlinear",
        "interaction": "nonlinear",
        "edge_irreps": "4x0e+4x1o",
        "use_edge_irreps_first": True,
    },
}


@pytest.fixture(autouse=True)
def no_gradients():
    """Nothing here differentiates, and the comparisons read values."""
    with torch.no_grad():
        yield


def _config(**overrides: Any) -> ModelConfig:
    options: dict[str, Any] = {
        "r_max": R_MAX,
        "num_interactions": 2,
        "num_channels": 4,
        "max_L": 2,
        "max_ell": 3,
        "correlation": 2,
        "radial_mlp": (8,),
    }
    options.update(overrides)
    return ModelConfig(**options)


def _precision() -> Precision:
    names: dict[torch.dtype, Precision] = {
        torch.float64: "float64",
        torch.float32: "float32",
    }
    return names[torch.get_default_dtype()]


def _backbone(backend: Any = None, seed: int = 0, **overrides: Any) -> MACEBackbone:
    model = MACEBackbone(
        backend or ReferenceBackend(),
        _config(**overrides),
        ELEMENTS,
        avg_num_neighbors=3.0,
        precision=_precision(),
    )
    initialize_model_weights(model, seed)
    return model


def _graph(structures) -> dict[str, torch.Tensor]:
    """Aperiodic structures batched into the flat graph dict.

    Every ordered pair of atoms of one structure closer than the cutoff is an
    edge, which is what any neighbour list returns for a cluster with no
    cell, so the shifts are zero.
    """
    dtype = torch.get_default_dtype()
    positions, numbers, senders, receivers, batch, ptr = [], [], [], [], [], [0]
    for index, (atomic_numbers, coordinates) in enumerate(structures):
        offset = ptr[-1]
        coordinates = np.asarray(coordinates, dtype=float)
        distances = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=-1)
        sender, receiver = np.nonzero((distances < R_MAX) & (distances > 0.0))
        senders.append(sender + offset)
        receivers.append(receiver + offset)
        positions.append(coordinates)
        numbers.extend(atomic_numbers)
        batch.extend([index] * len(atomic_numbers))
        ptr.append(offset + len(atomic_numbers))
    atomic_numbers = torch.tensor(numbers, dtype=torch.int64)
    columns = torch.tensor([ELEMENTS.index(z) for z in numbers])
    edge_index = torch.tensor(
        np.stack([np.concatenate(senders), np.concatenate(receivers)]),
        dtype=torch.int64,
    )
    return {
        "positions": torch.tensor(np.concatenate(positions), dtype=dtype),
        "node_attrs": torch.nn.functional.one_hot(columns, len(ELEMENTS)).to(dtype),
        "node_atomic_numbers": atomic_numbers,
        "edge_index": edge_index,
        "shifts": torch.zeros(edge_index.shape[1], 3, dtype=dtype),
        "batch": torch.tensor(batch, dtype=torch.int64),
        "ptr": torch.tensor(ptr, dtype=torch.int64),
    }


def _cluster(seed: int = 0, n_atoms: int = 6) -> dict[str, torch.Tensor]:
    """A small aperiodic cluster, every pair within the cutoff an edge."""
    generator = np.random.default_rng(seed)
    positions = generator.uniform(-1.2, 1.2, size=(n_atoms, 3))
    return _graph([([8, 1, 1, 6, 1, 8, 1, 6][:n_atoms], positions)])


def _padded(graph, n_atoms: int, n_edges: int) -> dict[str, torch.Tensor]:
    """``graph`` with inert padding appended, as the backbone's contract has it.

    One padding structure of ``n_atoms`` atoms of the table's first element,
    and ``n_edges`` self-loops among them whose shift puts them at twice the
    cutoff, where the envelope is exactly zero.
    """
    dtype = graph["positions"].dtype
    n_real = graph["positions"].shape[0]
    padding_nodes = n_real + torch.arange(n_edges) % n_atoms
    first = torch.zeros(n_atoms, len(ELEMENTS), dtype=dtype)
    first[:, 0] = 1.0
    shift = torch.zeros(n_edges, 3, dtype=dtype)
    shift[:, 0] = 2.0 * R_MAX
    return {
        "positions": torch.cat(
            [graph["positions"], torch.zeros(n_atoms, 3, dtype=dtype)]
        ),
        "node_attrs": torch.cat([graph["node_attrs"], first]),
        "node_atomic_numbers": torch.cat(
            [graph["node_atomic_numbers"], torch.full((n_atoms,), ELEMENTS[0])]
        ),
        "edge_index": torch.cat(
            [graph["edge_index"], torch.stack([padding_nodes, padding_nodes])], dim=1
        ),
        "shifts": torch.cat([graph["shifts"], shift]),
        "batch": torch.cat(
            [graph["batch"], torch.full((n_atoms,), int(graph["batch"].max()) + 1)]
        ),
        "ptr": torch.cat([graph["ptr"], graph["ptr"][-1:] + n_atoms]),
    }


def _wigner_d(degree: int, rotation: torch.Tensor) -> torch.Tensor:
    """``D^l(R)`` in this project's real basis, read off the harmonics.

    ``Y(R x) = D(R) Y(x)`` for every direction, solved over enough random
    directions to fix it. The harmonics define the basis the features are in,
    so this is the matrix the features must transform by.
    """
    points = torch.randn(
        64, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64
    )
    start, stop = degree**2, (degree + 1) ** 2
    original = spherical_harmonics(points, degree)[:, start:stop]
    rotated = spherical_harmonics(points @ rotation.T, degree)[:, start:stop]
    return torch.linalg.lstsq(original, rotated).solution.T


def _transform_features(features: torch.Tensor, irreps: str, rotation, parity: int):
    """Apply ``D(R)`` and the parity to grouped features."""
    from mace_core.clebsch_gordan.irreps import Irreps

    pieces, start = [], 0
    for mul, irrep in Irreps.parse(irreps).terms:
        width = mul * irrep.dimension
        block = features[:, start : start + width].reshape(-1, mul, irrep.dimension)
        matrix = _wigner_d(irrep.degree, rotation).to(features.dtype)
        sign = irrep.parity**1 if parity == -1 else 1
        pieces.append((sign * block @ matrix.T).reshape(-1, width))
        start += width
    return torch.cat(pieces, dim=-1)


def _rotation(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rotation, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )
    return rotation if torch.det(rotation) > 0 else -rotation


def _moved(graph, positions=None, shifts=None) -> dict[str, torch.Tensor]:
    moved = dict(graph)
    if positions is not None:
        moved["positions"] = positions
    if shifts is not None:
        moved["shifts"] = shifts
    return moved


# ---------------------------------------------------------------------------
# Equivariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_rotated_inputs_give_wigner_rotated_features(variant):
    model = _backbone(**VARIANTS[variant])
    graph = _cluster()
    rotation = _rotation(11)
    dtype = graph["positions"].dtype
    rotated = _moved(
        graph,
        graph["positions"] @ rotation.T.to(dtype),
        graph["shifts"] @ rotation.T.to(dtype),
    )
    before, after = model(graph).layers, model(rotated).layers
    for layer, irreps in enumerate(model.layer_irreps):
        expected = _transform_features(before[layer], irreps, rotation, parity=1)
        assert_close(after[layer], expected, f"{variant} layer {layer}")
    # The invariant channels do not move at all.
    assert_close(
        model.descriptors(model(rotated)), model.descriptors(model(graph)), "invariants"
    )


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_inverted_inputs_flip_exactly_the_odd_irreps(variant):
    model = _backbone(**VARIANTS[variant])
    graph = _cluster(1)
    inverted = _moved(graph, -graph["positions"], -graph["shifts"])
    before, after = model(graph).layers, model(inverted).layers
    identity = torch.eye(3, dtype=torch.float64)
    for layer, irreps in enumerate(model.layer_irreps):
        expected = _transform_features(before[layer], irreps, identity, parity=-1)
        assert_close(after[layer], expected, f"{variant} layer {layer}")


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_translated_inputs_give_the_same_features(variant):
    model = _backbone(**VARIANTS[variant])
    graph = _cluster(2)
    translation = torch.tensor([[1.7, -0.4, 3.1]], dtype=graph["positions"].dtype)
    moved = _moved(graph, graph["positions"] + translation)
    for ours, theirs in zip(model(moved).layers, model(graph).layers, strict=True):
        assert_close(ours, theirs, f"{variant} translation")


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_permuted_nodes_give_permuted_features(variant):
    model = _backbone(**VARIANTS[variant])
    graph = _cluster(3)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    inverse = torch.argsort(permutation)
    permuted = dict(graph)
    for key in ("positions", "node_attrs", "node_atomic_numbers"):
        permuted[key] = graph[key][permutation]
    permuted["edge_index"] = inverse[graph["edge_index"]]
    for ours, theirs in zip(model(permuted).layers, model(graph).layers, strict=True):
        assert_close(ours, theirs[permutation], f"{variant} permutation")


def test_a_declared_node_input_is_read_by_name_and_rotates_with_the_rest():
    spec = InputSpec(name="magmom", irreps="1o", per_atom=True, units="muB")
    model = MACEBackbone(
        ReferenceBackend(),
        _config(),
        ELEMENTS,
        precision=_precision(),
        node_inputs=[spec],
    )
    initialize_model_weights(model, 4)
    graph = _cluster(4)
    dtype = graph["positions"].dtype
    graph["magmom"] = torch.randn(
        6, 3, generator=torch.Generator().manual_seed(1), dtype=dtype
    )
    rotation = _rotation(5)
    rotated = _moved(
        graph,
        graph["positions"] @ rotation.T.to(dtype),
        graph["shifts"] @ rotation.T.to(dtype),
    )
    rotated["magmom"] = graph["magmom"] @ rotation.T.to(dtype)
    before, after = model(graph).layers, model(rotated).layers
    for layer, irreps in enumerate(model.layer_irreps):
        expected = _transform_features(before[layer], irreps, rotation, parity=1)
        assert_close(after[layer], expected, f"magmom layer {layer}")
    changed = dict(graph)
    changed["magmom"] = graph["magmom"] * 2.0
    assert not torch.allclose(model(changed).layers[-1], model(graph).layers[-1])


def test_a_per_structure_input_is_refused_as_a_node_input():
    spec = InputSpec(name="total_charge", irreps="0e", per_atom=False, units="e")
    with pytest.raises(ValueError, match="per structure"):
        MACEBackbone(ReferenceBackend(), _config(), ELEMENTS, node_inputs=[spec])


@pytest.mark.parametrize("names", [("positions",), ("magmom", "magmom")])
def test_a_node_input_that_collides_is_refused(names):
    specs = [
        InputSpec(name=name, irreps="1o", per_atom=True, units="1") for name in names
    ]
    with pytest.raises(ValueError, match="node input"):
        MACEBackbone(ReferenceBackend(), _config(), ELEMENTS, node_inputs=specs)


# ---------------------------------------------------------------------------
# The forward reads, resolves nothing, and permutes nothing
# ---------------------------------------------------------------------------


def test_the_forward_never_writes_into_the_graph():
    model = _backbone()
    graph = _cluster()
    snapshot = {
        key: (value, value.clone())
        for key, value in graph.items()
        if torch.is_tensor(value)
    }
    keys = set(graph)
    model(graph)
    assert set(graph) == keys
    for key, (value, copy) in snapshot.items():
        assert graph[key] is value, key
        assert torch.equal(value, copy), key


class _Recording:
    """The reference backend, counting every factory call."""

    def __init__(self, span=None, bases=frozenset({"reduced", "full"})):
        self._backend = ReferenceBackend()
        self.calls: list[str] = []
        self.spans: list[tuple] = []
        self._span = span
        self._bases = bases
        self.name = "recording"

    def capabilities(self) -> BackendCapabilities:
        declared = self._backend.capabilities()
        return type(declared)(
            ops=declared.ops,
            devices=declared.devices,
            dtypes=declared.dtypes,
            max_lmax=declared.max_lmax,
            layouts=declared.layouts,
            bases=self._bases,
            supports_double_backward=declared.supports_double_backward,
        )

    def make_interaction_layer(self, descriptors):
        self.calls.append("make_interaction_layer")
        self.spans.append(descriptors)
        return self._span

    def __getattr__(self, name):
        factory = getattr(self._backend, name)

        def recorded(descriptor):
            self.calls.append(name)
            self.capabilities().require(descriptor, self.name)
            return factory(descriptor)

        return recorded


def test_the_forward_builds_and_resolves_nothing():
    backend = _Recording()
    model = _backbone(backend)
    made = len(backend.calls)
    model(_cluster())
    assert len(backend.calls) == made


def test_the_blocks_carry_no_dispatch_probe_or_layout_move():
    """The frozen tree's dispatch lives inside its blocks; none of it may."""
    forbidden = (
        "conv_fusion",
        "is_scripting",
        "compile_mode",
        "torch.jit",
        ".permute(",
        ".transpose(",
        "forward =",
    )
    for name in (
        "backbone.py",
        "interaction.py",
        "product_basis.py",
        "gate.py",
        "edges.py",
    ):
        source = (NN / name).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{name} contains {token!r}"


def test_every_block_takes_the_edges_as_one_object():
    expected = [
        "node_features",
        "edges",
        "node_attrs",
        "sender",
        "receiver",
        "num_nodes",
    ]
    for block in (
        InteractionBlock,
        ResidualInteractionBlock,
        NonLinearInteractionBlock,
    ):
        parameters = list(inspect.signature(block.forward).parameters)[1:]
        assert parameters == expected, block.__name__


# ---------------------------------------------------------------------------
# The edges are lazy
# ---------------------------------------------------------------------------


class _Counting(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.count = 0

    def forward(self, *args):
        self.count += 1
        return self.inner(*args)


def test_an_edge_quantity_nobody_reads_is_never_computed():
    model = _backbone()
    graph = _cluster()
    harmonics = _Counting(model.spherical_harmonics)
    radial = _Counting(model.radial_embedding)
    vectors = (
        graph["positions"][graph["edge_index"][1]]
        - graph["positions"][graph["edge_index"][0]]
    )
    edges = EdgeInputs(
        vectors, graph["node_atomic_numbers"], graph["edge_index"], harmonics, radial
    )
    assert edges.built() == frozenset()
    _ = edges.attributes
    _ = edges.attributes
    assert edges.built() == {"attributes"}
    assert (harmonics.count, radial.count) == (1, 0)
    _ = edges.radial
    _ = edges.envelope
    assert radial.count == 1
    assert edges.built() == {"attributes", "radial", "lengths"}


# ---------------------------------------------------------------------------
# The span request
# ---------------------------------------------------------------------------


def test_every_block_asks_for_its_span_once_naming_every_member():
    backend = _Recording()
    model = _backbone(
        backend, distance_transform=AgnesiTransformConfig(), **VARIANTS["density"]
    )
    assert backend.calls.count("make_interaction_layer") == len(model.interactions)
    for members in backend.spans:
        kinds = [type(member) for member in members]
        for required in (
            RadialBasisDescriptor,
            SphericalHarmonicsDescriptor,
            DistanceTransformDescriptor,
            RadialMLPDescriptor,
            ChannelwiseTPConvDescriptor,
        ):
            assert required in kinds, required.__name__


def test_a_convolution_with_an_undescribed_piece_does_not_ask():
    """With the envelope after the radial networks, no descriptor states the
    radial embedding, so the block builds op by op without asking."""
    backend = _Recording()
    _backbone(backend, apply_cutoff=False)
    assert "make_interaction_layer" not in backend.calls


def test_an_accepted_span_changes_nothing_the_reference_computes():
    graph = _cluster()
    plain = _backbone(seed=9)
    offered = _backbone(_Recording(span=torch.nn.Identity()), seed=9)
    for ours, theirs in zip(offered(graph).layers, plain(graph).layers, strict=True):
        assert torch.equal(ours, theirs)


def test_a_span_holding_state_is_refused():
    with pytest.raises(ValueError, match="holds"):
        _backbone(_Recording(span=torch.nn.Linear(2, 2)))


# ---------------------------------------------------------------------------
# The Clebsch-Gordan basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["reduced", "full"])
def test_the_basis_reaches_every_contraction_unchanged(basis):
    config = _config(clebsch_gordan_basis=basis)
    assert ModelConfig.model_validate(config.model_dump(mode="json")) == config
    model = MACEBackbone(ReferenceBackend(), config, ELEMENTS, precision=_precision())
    products = cast(list[EquivariantProductBasisBlock], list(model.products))
    assert [product.contraction.descriptor.basis for product in products] == [
        basis,
        basis,
    ]


def test_a_full_basis_model_builds_only_on_a_backend_that_declares_it():
    reduced_only = _Recording(bases=frozenset({"reduced"}))
    _backbone(reduced_only)
    with pytest.raises(UnsupportedDescriptorError, match="full"):
        _backbone(_Recording(bases=frozenset({"reduced"})), clebsch_gordan_basis="full")


# ---------------------------------------------------------------------------
# Padding and locality
# ---------------------------------------------------------------------------


def test_a_padded_batch_gives_identical_features_on_the_real_slots():
    model = _backbone(**VARIANTS["density"])
    water = ([8, 1, 1], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    methane_fragment = ([6, 1], [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    batch = _graph([water, methane_fragment])
    padded = _padded(batch, n_atoms=4, n_edges=9)
    n_real = batch["positions"].shape[0]
    # The same numbers up to rounding, not bit for bit: a matrix product over
    # more rows may tile its sums differently.
    for ours, theirs in zip(model(padded).layers, model(batch).layers, strict=True):
        assert_close(ours[:n_real], theirs, "real slots")


def _exchange(full_layers: tuple[torch.Tensor, ...]):
    """A between-layer hook that fills the ghosts from a whole-system run,
    as a real exchange would from the domains that own them."""
    calls = []

    def hook(features, locality):
        layer = len(calls)
        calls.append(layer)
        ghosts = full_layers[layer][
            locality.n_local : locality.n_local + locality.n_ghost
        ]
        return torch.cat([features, ghosts], dim=0)

    return hook, calls


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_a_domain_with_ghosts_reproduces_its_local_atoms(variant):
    graph = _cluster(5)
    whole = _backbone(seed=2, **VARIANTS[variant])
    full_layers = whole(graph).layers
    hook, calls = _exchange(full_layers)
    domain = MACEBackbone(
        ReferenceBackend(),
        _config(**VARIANTS[variant]),
        ELEMENTS,
        avg_num_neighbors=3.0,
        precision=_precision(),
        between_layers=hook,
    )
    domain.load_state_dict(whole.state_dict())
    # The domain holds every edge into its own atoms, from local and ghost
    # senders alike, and none into the ghosts.
    local = dict(graph)
    keep = graph["edge_index"][1] < 3
    local["edge_index"] = graph["edge_index"][:, keep]
    local["shifts"] = graph["shifts"][keep]
    layers = domain(local, LocalityInfo(n_local=3, n_ghost=3)).layers
    assert calls == [0]
    for ours, theirs in zip(layers, full_layers, strict=True):
        assert ours.shape[0] == 3
        assert_close(ours, theirs[:3], f"{variant} local atoms")


def test_the_hook_is_inert_without_a_locality():
    graph = _cluster()
    plain = _backbone(seed=3)
    hooked = MACEBackbone(
        ReferenceBackend(),
        _config(),
        ELEMENTS,
        avg_num_neighbors=3.0,
        precision=_precision(),
        between_layers=lambda features, locality: features * 0.0,
    )
    hooked.load_state_dict(plain.state_dict())
    for ours, theirs in zip(hooked(graph).layers, plain(graph).layers, strict=True):
        assert torch.equal(ours, theirs)


@pytest.mark.parametrize(("n_local", "n_ghost"), [(0, 3), (3, -1)])
def test_a_locality_with_impossible_counts_is_refused(n_local, n_ghost):
    with pytest.raises(ValueError, match="LocalityInfo"):
        LocalityInfo(n_local=n_local, n_ghost=n_ghost)


def test_ghosts_without_an_exchange_are_refused_past_one_layer():
    with pytest.raises(ValueError, match="between_layers"):
        _backbone()(_cluster(), LocalityInfo(n_local=3, n_ghost=3))


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


def test_the_invariant_descriptors_are_each_layers_scalars():
    model = _backbone(num_interactions=3)
    features = model(_cluster())
    channels = model.num_features
    every = model.descriptors(features)
    assert every.shape[1] == 3 * channels
    for layer in range(3):
        assert torch.equal(
            every[:, layer * channels : (layer + 1) * channels],
            features.layers[layer][:, :channels],
        )
    assert torch.equal(
        model.descriptors(features, num_layers=2), every[:, : 2 * channels]
    )


def test_the_full_descriptors_are_the_layers_side_by_side():
    model = _backbone()
    features = model(_cluster())
    full = model.descriptors(features, invariants_only=False)
    widths = [layer.shape[1] for layer in features.layers]
    assert full.shape[1] == sum(widths)
    # The last layer carries only its scalars.
    assert widths[-1] == model.num_features
    assert torch.equal(
        model.descriptors(features, num_layers=1, invariants_only=False),
        features.layers[0],
    )


def test_a_layer_count_out_of_range_is_refused():
    model = _backbone()
    features = model(_cluster())
    with pytest.raises(ValueError, match="num_layers"):
        model.descriptors(features, num_layers=3)


def test_the_descriptors_aggregate_per_structure():
    descriptors = torch.arange(10.0).reshape(5, 2)
    batch = torch.tensor([0, 0, 1, 1, 1])
    element = torch.tensor([0, 1, 1, 1, 2])
    means = aggregate_descriptors(descriptors, batch, 2, "mean")
    assert_close(means, [[1.0, 2.0], [6.0, 7.0]], "mean")
    per_element = aggregate_descriptors(
        descriptors,
        batch,
        2,
        "per_element_mean",
        element=element,
        atomic_numbers=ELEMENTS,
    )
    assert sorted(per_element[0]) == [1, 6]
    assert sorted(per_element[1]) == [6, 8]
    assert_close(per_element[1][6], [5.0, 6.0], "per element")
    assert aggregate_descriptors(descriptors, batch, 2, None) is descriptors
    with pytest.raises(ValueError, match="aggregation"):
        aggregate_descriptors(descriptors, batch, 2, cast(Any, "median"))


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize("variant", ["residual", "nonlinear"])
def test_the_backbone_compiles_whole(variant):
    from torch._dynamo import reset
    from torch._dynamo.testing import CompileCounter

    model = _backbone(**VARIANTS[variant])
    graph = _cluster()
    reset()
    counter = CompileCounter()
    compiled = torch.compile(model, backend=counter, fullgraph=True)
    result = compiled(graph)
    assert isinstance(result, NodeFeatures)
    for ours, theirs in zip(result.layers, model(graph).layers, strict=True):
        assert_close(ours, theirs, f"{variant} compiled")
    assert counter.frame_count == 1
