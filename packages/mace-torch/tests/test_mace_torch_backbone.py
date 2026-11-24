"""The backbone: equivariance, a read-only forward, padding, locality,
descriptors, compile, and what it asks of the backend.

Every test builds through the reference backend and runs once per dtype, the
model built in the dtype the suite is running in.
"""

import inspect
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from conftest import (
    ELEMENTS,
    R_MAX,
    assert_close,
    cluster,
    fp64_only,
    graph_of,
    precision,
    rotation_matrix,
    transform_features,
)
from mace_core.config import (
    AgnesiTransformConfig,
    BesselBasisConfig,
    ChebyshevBasisConfig,
    GaussianBasisConfig,
    ModelConfig,
    SoftTransformConfig,
)
from mace_core.kernels import (
    BackendCapabilities,
    ChannelwiseTPConvSpec,
    DistanceTransformSpec,
    RadialBasisSpec,
    RadialMLPSpec,
    SphericalHarmonicsSpec,
    UnsupportedKernelSpecError,
)
from mace_torch.backends.reference import ReferenceBackend, ReferenceLinear
from mace_torch.kernels import (
    canonical_state,
    initialize_model_weights,
    load_canonical_state,
)
from mace_torch.kernels.ops import equivariant_linear
from mace_torch.nn import backbone as backbone_module
from mace_torch.nn.backbone import (
    LocalityInfo,
    MACEBackbone,
    NodeFeatures,
)
from mace_torch.nn.edges import EdgeInputs
from mace_torch.nn.interaction import (
    InteractionBlock,
    NonLinearInteractionBlock,
    ResidualInteractionBlock,
)
from mace_torch.nn.product_basis import EquivariantProductBasisBlock

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


pytestmark = pytest.mark.usefixtures("no_gradients")


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


def _backbone(
    backend: Any = None,
    seed: int = 0,
    avg_num_neighbors: float = 3.0,
    **overrides: Any,
) -> MACEBackbone:
    model = MACEBackbone(
        backend or ReferenceBackend(),
        _config(**overrides),
        ELEMENTS,
        avg_num_neighbors=avg_num_neighbors,
        precision=precision(),
    )
    initialize_model_weights(model, seed)
    return model


def _padded(graph, n_atoms: int, n_edges: int) -> dict[str, torch.Tensor]:
    """``graph`` with inert padding appended, as the backbone's contract has it.

    One padding structure of ``n_atoms`` atoms of atomic number 0, and
    ``n_edges`` self-loops among them whose shift puts them at twice the
    cutoff, where the envelope is exactly zero.
    """
    dtype = graph["positions"].dtype
    n_real = graph["positions"].shape[0]
    padding_nodes = n_real + torch.arange(n_edges) % n_atoms
    shift = torch.zeros(n_edges, 3, dtype=dtype)
    shift[:, 0] = 2.0 * R_MAX
    return {
        "positions": torch.cat(
            [graph["positions"], torch.zeros(n_atoms, 3, dtype=dtype)]
        ),
        "node_atomic_numbers": torch.cat(
            [graph["node_atomic_numbers"], torch.zeros(n_atoms, dtype=torch.int64)]
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
    graph = cluster()
    rotation = rotation_matrix(11)
    dtype = graph["positions"].dtype
    rotated = _moved(
        graph,
        graph["positions"] @ rotation.T.to(dtype),
        graph["shifts"] @ rotation.T.to(dtype),
    )
    before, after = model(graph).layers, model(rotated).layers
    for layer, irreps in enumerate(model.layer_irreps):
        expected = transform_features(before[layer], irreps, rotation, parity=1)
        assert_close(after[layer], expected, f"{variant} layer {layer}")
    # The invariant channels do not move at all.
    assert_close(
        model.atomic_descriptors(model(rotated)),
        model.atomic_descriptors(model(graph)),
        "invariants",
    )


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_inverted_inputs_flip_exactly_the_odd_irreps(variant):
    model = _backbone(**VARIANTS[variant])
    graph = cluster(1)
    inverted = _moved(graph, -graph["positions"], -graph["shifts"])
    before, after = model(graph).layers, model(inverted).layers
    identity = torch.eye(3, dtype=torch.float64)
    for layer, irreps in enumerate(model.layer_irreps):
        expected = transform_features(before[layer], irreps, identity, parity=-1)
        assert_close(after[layer], expected, f"{variant} layer {layer}")


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_translated_inputs_give_the_same_features(variant):
    model = _backbone(**VARIANTS[variant])
    graph = cluster(2)
    translation = torch.tensor([[1.7, -0.4, 3.1]], dtype=graph["positions"].dtype)
    moved = _moved(graph, graph["positions"] + translation)
    for ours, theirs in zip(model(moved).layers, model(graph).layers, strict=True):
        assert_close(ours, theirs, f"{variant} translation")


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_permuted_nodes_give_permuted_features(variant):
    model = _backbone(**VARIANTS[variant])
    graph = cluster(3)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    inverse = torch.argsort(permutation)
    permuted = dict(graph)
    for key in ("positions", "node_atomic_numbers"):
        permuted[key] = graph[key][permutation]
    permuted["edge_index"] = inverse[graph["edge_index"]]
    for ours, theirs in zip(model(permuted).layers, model(graph).layers, strict=True):
        assert_close(ours, theirs[permutation], f"{variant} permutation")


# ---------------------------------------------------------------------------
# The forward reads, resolves nothing, and permutes nothing
# ---------------------------------------------------------------------------


class _ReadKeys(Mapping):
    """A graph that records every key read, through any accessor.

    ``get`` and ``in`` go through ``__getitem__``, and anything that walks the
    keys is recorded as reading all of them.
    """

    def __init__(self, graph):
        self._graph = dict(graph)
        self.read: set[str] = set()

    def __getitem__(self, key):
        self.read.add(key)
        return self._graph[key]

    def __iter__(self):
        self.read.update(self._graph)
        return iter(self._graph)

    def __len__(self):
        return len(self._graph)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_forward_reads_four_keys_whatever_else_the_graph_carries(variant):
    graph = _ReadKeys(
        {**cluster(), "magmom": torch.zeros(6, 3), "total_charge": torch.zeros(1)}
    )
    _backbone(**VARIANTS[variant])(graph)
    assert graph.read == {"positions", "node_atomic_numbers", "edge_index", "shifts"}


def test_the_forward_never_writes_into_the_graph():
    model = _backbone()
    graph = cluster()
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


# ---------------------------------------------------------------------------
# The element one-hot
# ---------------------------------------------------------------------------


def _table_one_hot(atomic_numbers: torch.Tensor) -> torch.Tensor:
    columns = torch.tensor([ELEMENTS.index(int(z)) for z in atomic_numbers])
    return torch.nn.functional.one_hot(columns, len(ELEMENTS)).to(
        torch.get_default_dtype()
    )


def test_the_one_hot_and_the_index_are_the_element_tables():
    graph = cluster()
    features = _backbone()(graph)
    assert torch.equal(
        features.node_attrs, _table_one_hot(graph["node_atomic_numbers"])
    )
    assert torch.equal(features.element_index, features.node_attrs.argmax(dim=-1))


def test_the_one_hot_is_in_the_models_precision():
    model = _backbone()
    assert model(cluster()).node_attrs.dtype == torch.get_default_dtype()


@pytest.mark.parametrize(
    ("atomic_numbers", "message"),
    [
        ((1, 119), r"atomic_numbers: .*holds \[119\]"),
        ((8, 1, 6), r"atomic_numbers: the element table \[8, 1, 6\] is not strictly"),
        (iter((0, 1)), "element table"),
    ],
    ids=["outside", "unsorted", "iterator"],
)
def test_the_element_table_is_checked_at_build(atomic_numbers, message):
    """The check itself is mace_core's, tested there."""
    with pytest.raises(ValueError, match=message):
        MACEBackbone(ReferenceBackend(), _config(), atomic_numbers)


def test_nothing_the_backbone_keeps_beyond_its_saved_state_depends_on_the_table():
    """A lookup derived from the table and kept would go stale when a saved
    state replaces the table."""

    def kept(model):
        saved = set(model.state_dict())
        return {
            name: tensor for name, tensor in model.named_buffers() if name not in saved
        }

    first = MACEBackbone(ReferenceBackend(), _config(), (1, 6, 8))
    second = MACEBackbone(ReferenceBackend(), _config(), (3, 29, 79))
    first_kept, second_kept = kept(first), kept(second)
    assert first_kept.keys() == second_kept.keys()
    for name, tensor in first_kept.items():
        assert torch.equal(tensor, second_kept[name]), name


def test_a_backbone_built_at_one_table_computes_at_the_table_it_loads():
    """Only the table's length is fixed at build: a state for another table
    of that length gives the backbone built at that table, padding included."""
    source = _trained("residual")
    other = (3, 29, 79)
    state = canonical_state(source)
    state["element_embedding"] = {
        **state["element_embedding"],
        "atomic_numbers": torch.tensor(other),
    }
    built_at_other = MACEBackbone(
        ReferenceBackend(),
        _config(**VARIANTS["residual"]),
        other,
        avg_num_neighbors=3.7,
        precision=precision(),
    )
    loaded_into_original = _backbone(**VARIANTS["residual"])
    load_canonical_state(built_at_other, state)
    load_canonical_state(loaded_into_original, state)
    graph = _padded(cluster(), n_atoms=2, n_edges=2)
    graph["node_atomic_numbers"] = torch.tensor([3, 29, 79, 29, 3, 79, 0, 0])
    results = []
    with torch.enable_grad():
        for model in (loaded_into_original, built_at_other):
            positions = graph["positions"].clone().requires_grad_(True)
            layers = model({**graph, "positions": positions}).layers
            (gradient,) = torch.autograd.grad(layers[-1].sum(), positions)
            results.append((layers, gradient))
    (ours, our_gradient), (theirs, their_gradient) = results
    for our_layer, their_layer in zip(ours, theirs, strict=True):
        assert torch.equal(our_layer, their_layer)
    assert torch.equal(our_gradient, their_gradient)


def test_a_backbone_built_from_a_saved_table_loads_its_state():
    """What a checkpoint loader does: the table is read from the state before
    the model is built."""
    source = _trained("residual")
    state = canonical_state(source)
    target = MACEBackbone(
        ReferenceBackend(),
        _config(**VARIANTS["residual"]),
        state["element_embedding"]["atomic_numbers"],
        precision=precision(),
    )
    load_canonical_state(target, state)
    graph = cluster()
    for ours, theirs in zip(target(graph).layers, source(graph).layers, strict=True):
        assert torch.equal(ours, theirs)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "radial_basis": GaussianBasisConfig(),
            "distance_transform": SoftTransformConfig(),
        },
        {"distance_transform": AgnesiTransformConfig()},
        {"radial_basis": ChebyshevBasisConfig()},
        {"interaction_first": "nonlinear", "interaction": "density"},
    ],
    ids=["bessel", "gaussian_soft", "agnesi", "chebyshev", "nonlinear_density"],
)
def test_the_checkpoint_holds_only_learned_weights_and_fitted_statistics(overrides):
    """What the configuration fixes is rebuilt from it and never saved.

    A saved constant would be overwritten by the checkpoint's value while the
    configuration, and so the neighbour list, kept its own: a checkpoint for
    another `r_max` would load without error and compute with two cutoffs.
    """
    model = _backbone(**overrides)
    parameters = {name for name, _ in model.named_parameters()}
    saved = set(model.state_dict()) - parameters
    assert "element_embedding.atomic_numbers" in saved
    assert {key.rsplit(".", 1)[-1] for key in saved} <= {
        "atomic_numbers",
        "neighbours",
    }, saved


#: Every option that changes which tensors the backbone saves.
SAVED_STATE_VARIANTS: dict[str, dict[str, Any]] = {
    **VARIANTS,
    "bessel_trainable": {"radial_basis": BesselBasisConfig(trainable=True)},
    "gaussian_trainable": {"radial_basis": GaussianBasisConfig(trainable=True)},
    "chebyshev": {"radial_basis": ChebyshevBasisConfig()},
    "agnesi_trainable": {"distance_transform": AgnesiTransformConfig(trainable=True)},
    "soft_trainable": {"distance_transform": SoftTransformConfig(trainable=True)},
    "agnostic_product": {"use_agnostic_product": True},
    "envelope_outside": {"apply_cutoff": False},
}


def _trained(variant: str) -> MACEBackbone:
    """A backbone whose every saved number differs from a freshly built one's,
    trainable constants and the neighbour count included."""
    model = _backbone(seed=1, avg_num_neighbors=3.7, **SAVED_STATE_VARIANTS[variant])
    generator = torch.Generator().manual_seed(5)
    for parameter in model.parameters():
        noise = torch.randn(parameter.shape, generator=generator)
        parameter.add_(0.01 * noise.to(parameter.dtype))
    return model


@pytest.mark.parametrize("variant", sorted(SAVED_STATE_VARIANTS))
def test_a_saved_state_rebuilds_the_backbone_bit_for_bit(variant):
    """The second backbone starts from other weights and another neighbour
    count, so a tensor the state leaves out is one the two disagree on."""
    source = _trained(variant)
    target = _backbone(seed=2, avg_num_neighbors=1.0, **SAVED_STATE_VARIANTS[variant])
    load_canonical_state(target, canonical_state(source))
    source_state, target_state = source.state_dict(), target.state_dict()
    assert source_state.keys() == target_state.keys()
    for key, tensor in source_state.items():
        assert torch.equal(tensor, target_state[key]), key
    graph = cluster()
    for ours, theirs in zip(source(graph).layers, target(graph).layers, strict=True):
        assert torch.equal(ours, theirs)


@pytest.mark.parametrize("variant", sorted(SAVED_STATE_VARIANTS))
def test_the_saved_state_holds_the_element_table_once(variant):
    state = canonical_state(_backbone(**SAVED_STATE_VARIANTS[variant]))
    assert [
        (path, name)
        for path, tensors in state.items()
        for name in tensors
        if name == "atomic_numbers"
    ] == [("element_embedding", "atomic_numbers")]
    assert state["element_embedding"]["atomic_numbers"].tolist() == list(ELEMENTS)
    assert state["element_embedding"]["weight"].shape == (len(ELEMENTS), 4)


class _ReversedLinear(ReferenceLinear):
    """The reference linear holding its weight reversed, as a backend with
    its own layout would, and converting at the canonical boundary."""

    def forward(self, features):
        return equivariant_linear(
            features,
            self.weight.flip(0) * self.path_normalization,
            self.row,
            self.column,
            self.source,
            self.bias,
            self.bias_row,
            self.dim_out,
        )

    def to_canonical(self):
        return {"weight": self.weight.detach().flip(0), "bias": self.bias.detach()}

    def load_canonical(self, state):
        super().load_canonical({**state, "weight": state["weight"].flip(0)})


class _DerivedBackend(ReferenceBackend):
    name = "derived"

    def make_linear(self, kernel_spec):
        return _ReversedLinear(kernel_spec)


@fp64_only
def test_one_saved_state_builds_with_any_backend():
    """The backend is chosen by whoever builds the model: it is not part of
    the config, and what is saved is the canonical layout, not the layout a
    backend holds."""
    source = _trained("residual")
    target = MACEBackbone(
        _DerivedBackend(),
        _config(**VARIANTS["residual"]),
        ELEMENTS,
        precision=precision(),
    )
    load_canonical_state(target, canonical_state(source))
    assert any(isinstance(module, _ReversedLinear) for module in target.modules())
    graph = cluster()
    for ours, theirs in zip(target(graph).layers, source(graph).layers, strict=True):
        assert_close(ours, theirs, "derived backend")


def test_the_element_embedding_draws_its_weights_with_the_ops():
    model = MACEBackbone(ReferenceBackend(), _config(), ELEMENTS)
    assert "element_embedding" in initialize_model_weights(model, 0)
    assert float(model.element_embedding.weight.abs().max()) > 0.0


def test_the_neighbour_count_is_saved_where_it_is_read():
    """A block that normalizes by a learned density has no neighbour count."""
    plain = canonical_state(_backbone(avg_num_neighbors=3.7))
    density = canonical_state(_backbone(**VARIANTS["density"]))
    assert float(plain["interactions.1.body"]["neighbours"]) == pytest.approx(3.7)
    assert not [path for path in density if path.endswith(".body")]


def test_untrained_constants_are_not_saved():
    """A basis or transform built without ``trainable`` is rebuilt from the config."""
    state = canonical_state(_backbone(distance_transform=AgnesiTransformConfig()))
    assert not [path for path in state if path.startswith("radial_embedding")]


def test_a_table_its_module_refuses_leaves_the_whole_backbone_as_it_was():
    """The table is checked by the module that loads it, after the modules
    before it in the state have loaded; those are put back."""
    variant = "bessel_trainable"
    target = _backbone(seed=2, avg_num_neighbors=1.0, **SAVED_STATE_VARIANTS[variant])
    state = canonical_state(_trained(variant))
    assert list(state).index("element_embedding") > 0
    state["element_embedding"] = {
        **state["element_embedding"],
        "atomic_numbers": torch.tensor([8, 6, 1]),
    }
    before = {key: tensor.clone() for key, tensor in target.state_dict().items()}
    with pytest.raises(
        ValueError, match=r"^element_embedding: atomic_numbers: .*not strictly"
    ):
        load_canonical_state(target, state)
    for key, tensor in target.state_dict().items():
        assert torch.equal(tensor, before[key]), key


def test_load_state_dict_refuses_a_table_that_is_not_a_models():
    target = _backbone(seed=2, avg_num_neighbors=1.0)
    state = _trained("residual").state_dict()
    state["element_embedding.atomic_numbers"] = torch.tensor([8, 6, 1])
    with pytest.raises(
        RuntimeError, match=r"element_embedding\.atomic_numbers: .*not strictly"
    ):
        target.load_state_dict(state)
    assert target.element_embedding.atomic_numbers.tolist() == list(ELEMENTS)


def _state_without(state, path, name):
    changed = {key: dict(value) for key, value in state.items()}
    del changed[path][name]
    return changed


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda state: _state_without(state, "interactions.0.body", "neighbours"),
            "interactions.0.body: missing 'neighbours'",
        ),
        (
            lambda state: {**state, "extra": {"weight": torch.zeros(1)}},
            "extra: no module here saves state",
        ),
        (
            lambda state: {**state, "radial_embedding": {}},
            "radial_embedding: no module here saves state",
        ),
        (
            lambda state: {**state, "": {}},
            "<root>: no module here saves state",
        ),
        (
            lambda state: {
                **state,
                "interactions.0.body": {
                    **state["interactions.0.body"],
                    "extra": torch.zeros(1),
                },
            },
            "interactions.0.body: unexpected 'extra'",
        ),
        (
            lambda state: {
                **state,
                "element_embedding": {
                    **state["element_embedding"],
                    "weight": torch.zeros(5),
                },
            },
            "'weight' has shape (5,)",
        ),
    ],
)
def test_a_state_that_does_not_fit_is_refused_and_nothing_is_loaded(change, message):
    target = _backbone(seed=2, avg_num_neighbors=1.0)
    before = {key: tensor.clone() for key, tensor in target.state_dict().items()}
    with pytest.raises(ValueError, match=re.escape(message)):
        load_canonical_state(target, change(canonical_state(_trained("residual"))))
    for key, tensor in target.state_dict().items():
        assert torch.equal(tensor, before[key]), key


# ---------------------------------------------------------------------------
# Ops the backend declines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["spherical_harmonics", "radial_basis"])
def test_an_edge_op_the_backend_declined_is_refused_at_build(op):
    backend = ReferenceBackend()
    setattr(backend, f"make_{op}", lambda kernel_spec: None)
    with pytest.raises(UnsupportedKernelSpecError, match=f"declined '{op}'"):
        _backbone(backend)


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

    def make_interaction_layer(self, kernel_specs):
        self.calls.append("make_interaction_layer")
        self.spans.append(kernel_specs)
        return self._span

    def __getattr__(self, name):
        factory = getattr(self._backend, name)

        def recorded(kernel_spec):
            self.calls.append(name)
            self.capabilities().require(kernel_spec, self.name)
            return factory(kernel_spec)

        return recorded


def test_the_forward_builds_and_resolves_nothing():
    backend = _Recording()
    model = _backbone(backend)
    made = len(backend.calls)
    model(cluster())
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
    graph = cluster()
    harmonics = _Counting(model.spherical_harmonics)
    radial = _Counting(model.radial_embedding)
    vectors = (
        graph["positions"][graph["edge_index"][1]]
        - graph["positions"][graph["edge_index"][0]]
    )
    edges = EdgeInputs(
        vectors, graph["node_atomic_numbers"], graph["edge_index"], harmonics, radial
    )
    assert (harmonics.count, radial.count) == (0, 0)
    _ = edges.attributes
    _ = edges.attributes
    assert (harmonics.count, radial.count) == (1, 0)
    _ = edges.radial
    _ = edges.envelope
    assert (harmonics.count, radial.count) == (1, 1)


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
            RadialBasisSpec,
            SphericalHarmonicsSpec,
            DistanceTransformSpec,
            RadialMLPSpec,
            ChannelwiseTPConvSpec,
        ):
            assert required in kinds, required.__name__


@pytest.mark.parametrize(
    "options",
    [
        {"apply_cutoff": False},
        {"radial_basis": ChebyshevBasisConfig(num_basis=8, include_constant=True)},
        {"radial_basis": BesselBasisConfig(num_basis=8, trainable=True)},
    ],
    ids=["envelope-after-the-networks", "chebyshev-constant", "trainable-bessel"],
)
def test_the_span_request_states_every_option_of_the_radial_basis(options):
    """The radial basis a span is asked about is the one the model built: an
    option of the basis section that changed what it computes reaches the
    specification, so a backend that does not handle it can decline."""
    backend = _Recording()
    model = _backbone(backend, **options)
    asked = {
        member
        for members in backend.spans
        for member in members
        if isinstance(member, RadialBasisSpec)
    }
    assert asked == {model.radial_embedding.kernel_spec}
    (spec,) = asked
    assert spec.apply_cutoff == options.get("apply_cutoff", True)
    basis = options.get("radial_basis")
    assert spec.include_constant == getattr(basis, "include_constant", False)
    assert spec.trainable == getattr(basis, "trainable", False)


def test_the_radial_embedding_is_the_backends_op():
    backend = _Recording()
    model = _backbone(backend)
    assert "make_radial_basis" in backend.calls
    assert "make_spherical_harmonics" in backend.calls
    assert model.radial_embedding.kernel_spec.cutoff == model.config.r_max


def test_an_accepted_span_changes_nothing_the_reference_computes():
    graph = cluster()
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
    model = MACEBackbone(ReferenceBackend(), config, ELEMENTS, precision=precision())
    products = cast(list[EquivariantProductBasisBlock], list(model.products))
    assert [product.contraction.kernel_spec.basis for product in products] == [
        basis,
        basis,
    ]


def test_a_full_basis_model_builds_only_on_a_backend_that_declares_it():
    reduced_only = _Recording(bases=frozenset({"reduced"}))
    _backbone(reduced_only)
    with pytest.raises(UnsupportedKernelSpecError, match="full"):
        _backbone(_Recording(bases=frozenset({"reduced"})), clebsch_gordan_basis="full")


# ---------------------------------------------------------------------------
# Padding and locality
# ---------------------------------------------------------------------------


def test_a_padded_batch_gives_identical_features_on_the_real_slots():
    model = _backbone(**VARIANTS["density"])
    water = ([8, 1, 1], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    methane_fragment = ([6, 1], [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    batch = graph_of([water, methane_fragment])
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
    graph = cluster(5)
    whole = _backbone(seed=2, **VARIANTS[variant])
    full_layers = whole(graph).layers
    hook, calls = _exchange(full_layers)
    domain = MACEBackbone(
        ReferenceBackend(),
        _config(**VARIANTS[variant]),
        ELEMENTS,
        avg_num_neighbors=3.0,
        precision=precision(),
        ghost_exchange=hook,
    )
    domain.load_state_dict(whole.state_dict())
    # The domain holds every edge into its own atoms, from local and ghost
    # senders alike, and none into the ghosts.
    local = dict(graph)
    keep = graph["edge_index"][1] < 3
    local["edge_index"] = graph["edge_index"][:, keep]
    local["shifts"] = graph["shifts"][keep]
    features = domain(local, LocalityInfo(n_local=3, n_ghost=3))
    layers = features.layers
    assert calls == [0]
    assert features.node_attrs.shape[0] == 3
    for ours, theirs in zip(layers, full_layers, strict=True):
        assert ours.shape[0] == 3
        assert_close(ours, theirs[:3], f"{variant} local atoms")


def test_the_hook_is_inert_without_a_locality():
    graph = cluster()
    plain = _backbone(seed=3)
    hooked = MACEBackbone(
        ReferenceBackend(),
        _config(),
        ELEMENTS,
        avg_num_neighbors=3.0,
        precision=precision(),
        ghost_exchange=lambda features, locality: features * 0.0,
    )
    hooked.load_state_dict(plain.state_dict())
    for ours, theirs in zip(hooked(graph).layers, plain(graph).layers, strict=True):
        assert torch.equal(ours, theirs)


@pytest.mark.parametrize(("n_local", "n_ghost"), [(0, 3), (3, -1)])
def test_a_locality_with_impossible_counts_is_refused(n_local, n_ghost):
    with pytest.raises(ValueError, match="LocalityInfo"):
        LocalityInfo(n_local=n_local, n_ghost=n_ghost)


def test_ghosts_without_an_exchange_are_refused_past_one_layer():
    with pytest.raises(ValueError, match="ghost_exchange"):
        _backbone()(cluster(), LocalityInfo(n_local=3, n_ghost=3))


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


def test_the_invariant_descriptors_are_each_layers_scalars():
    model = _backbone(num_interactions=3)
    features = model(cluster())
    channels = model.num_features
    every = model.atomic_descriptors(features)
    assert every.shape[1] == 3 * channels
    for layer in range(3):
        assert torch.equal(
            every[:, layer * channels : (layer + 1) * channels],
            features.layers[layer][:, :channels],
        )
    assert torch.equal(
        model.atomic_descriptors(features, num_layers=2), every[:, : 2 * channels]
    )


def test_the_full_descriptors_are_the_layers_side_by_side():
    model = _backbone()
    features = model(cluster())
    full = model.atomic_descriptors(features, invariants_only=False)
    widths = [layer.shape[1] for layer in features.layers]
    assert full.shape[1] == sum(widths)
    # The last layer carries only its scalars.
    assert widths[-1] == model.num_features
    assert torch.equal(
        model.atomic_descriptors(features, num_layers=1, invariants_only=False),
        features.layers[0],
    )


def test_a_layer_count_out_of_range_is_refused():
    model = _backbone()
    features = model(cluster())
    with pytest.raises(ValueError, match="num_layers"):
        model.atomic_descriptors(features, num_layers=3)


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize("variant", ["residual", "nonlinear"])
def test_the_backbone_compiles_whole(variant):
    from torch._dynamo import reset
    from torch._dynamo.testing import CompileCounter

    model = _backbone(**VARIANTS[variant])
    graph = cluster()
    reset()
    counter = CompileCounter()
    compiled = torch.compile(model, backend=counter, fullgraph=True)
    result = compiled(graph)
    assert isinstance(result, NodeFeatures)
    eager = model(graph)
    for ours, theirs in zip(result.layers, eager.layers, strict=True):
        assert_close(ours, theirs, f"{variant} compiled")
    assert torch.equal(result.node_attrs, eager.node_attrs)
    assert torch.equal(result.element_index, eager.element_index)
    assert counter.frame_count == 1
