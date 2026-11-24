"""The element embedding and the radial embedding block.

The radial block's forward runs three steps in an order that is load-bearing
and invisible from outside unless asserted: the cutoff is computed on the RAW
edge lengths, then the distance transform, then the basis. With an Agnesi
transform, an envelope computed from the transformed lengths would differ by
0.87 at r = 0.9 Ang, a different model rather than a rounding difference.
"""

import math

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.config import (
    AgnesiTransformConfig,
    BesselBasisConfig,
    ChebyshevBasisConfig,
    GaussianBasisConfig,
    NoDistanceTransformConfig,
    PolynomialCutoffConfig,
    SoftTransformConfig,
)
from mace_core.data import Configuration
from mace_core.graph import SampleError, graph_layout, graph_sample
from mace_core.kernels.precision import Precision
from mace_torch.backends.radial import (
    BesselBasis,
    ChebyshevBasis,
    GaussianBasis,
    PolynomialCutoff,
)
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.nn.embedding import (
    ElementEmbedding,
    RadialEmbeddingBlock,
    build_distance_transform,
    radial_basis_spec,
)
from mace_torch.nn.radial import (
    AgnesiTransform,
    SoftTransform,
)

EMBEDDING_R_MAX = 3.0
RADIAL_BASES = {
    "bessel": BesselBasisConfig(num_basis=4),
    "gaussian": GaussianBasisConfig(num_basis=4),
    "chebyshev": ChebyshevBasisConfig(num_basis=4),
}
DISTANCE_TRANSFORMS = {
    "none": NoDistanceTransformConfig(),
    "agnesi": AgnesiTransformConfig(),
    "soft": SoftTransformConfig(),
}


def _embedding_inputs():
    lengths = torch.tensor([[0.9], [1.7], [2.5]])
    node_atomic_numbers = torch.tensor([1, 6])
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 1]])
    return lengths, node_atomic_numbers, edge_index


def _precision() -> Precision:
    precisions: dict[torch.dtype, Precision] = {
        torch.float64: "float64",
        torch.float32: "float32",
    }
    return precisions[torch.get_default_dtype()]


def _radial_op(config, cutoff=None, r_max=EMBEDDING_R_MAX, apply_cutoff=True):
    """The reference backend's op for the basis and envelope the sections name."""
    spec = radial_basis_spec(
        config,
        cutoff or PolynomialCutoffConfig(polynomial_order=6),
        r_max,
        apply_cutoff=apply_cutoff,
        precision=_precision(),
    )
    return ReferenceBackend().make_radial_basis(spec)


def _block(
    radial_basis: str = "bessel",
    distance_transform: str = "none",
    apply_cutoff: bool = True,
) -> RadialEmbeddingBlock:
    return RadialEmbeddingBlock(
        radial_basis=_radial_op(RADIAL_BASES[radial_basis], apply_cutoff=apply_cutoff),
        distance_transform=build_distance_transform(
            DISTANCE_TRANSFORMS[distance_transform]
        ),
    )


# ---------------------------------------------------------------------------
# ElementEmbedding
# ---------------------------------------------------------------------------


def _element_embedding(atomic_numbers=(1, 6, 8), num_channels=5, seed=0):
    embedding = ElementEmbedding(atomic_numbers, num_channels, _precision())
    embedding.initialize_weights(seed)
    return embedding


def test_each_node_takes_its_elements_column_and_row():
    """The output of node i is the row of its element, scaled by the
    `1/sqrt(n_elements)` fan-in; the one-hot and the index say which row."""
    embedding = _element_embedding()
    encoding = embedding(torch.tensor([8, 1, 8, 6]))
    assert encoding.index.tolist() == [2, 0, 2, 1]
    assert torch.equal(encoding.one_hot, torch.eye(3)[[2, 0, 2, 1]])
    torch.testing.assert_close(
        encoding.features, embedding.weight[[2, 0, 2, 1]] / math.sqrt(3)
    )


def test_the_one_hot_keeps_the_tables_width_without_its_last_element():
    encoding = _element_embedding()(torch.tensor([1, 6, 1]))
    assert encoding.one_hot.shape == (3, 3)


def test_a_padding_atom_takes_the_first_column():
    assert _element_embedding()(torch.tensor([0, 8])).index.tolist() == [0, 2]


def test_the_initial_embedding_is_a_unit_normal_drawn_from_its_seed():
    embedding = _element_embedding((1, 6, 7, 8), num_channels=4096, seed=3)
    assert embedding.weight.std().item() == pytest.approx(1.0, rel=0.05)
    assert embedding.weight.mean().item() == pytest.approx(0.0, abs=0.02)
    again = _element_embedding((1, 6, 7, 8), num_channels=4096, seed=3)
    assert torch.equal(embedding.weight, again.weight)


def test_the_embedding_saves_its_table_and_its_weight():
    embedding = _element_embedding()
    state = embedding.to_canonical()
    assert state.keys() == {"atomic_numbers", "weight"}
    assert state["atomic_numbers"].tolist() == [1, 6, 8]
    assert state["atomic_numbers"].dtype == torch.int64
    assert state["weight"].shape == (3, 5)


# Each way a saved state reaches the embedding, and the error it raises on a
# table that is not a model's. torch's loader reports through its error list,
# which raises even when not strict.
LOADERS = {
    "canonical": (lambda module, state: module.load_canonical(state), ValueError),
    "state_dict": (lambda module, state: module.load_state_dict(state), RuntimeError),
    "state_dict_not_strict": (
        lambda module, state: module.load_state_dict(state, strict=False),
        RuntimeError,
    ),
}


@pytest.mark.parametrize("loader", LOADERS)
def test_a_loaded_table_is_the_one_the_forward_searches(loader):
    load, _ = LOADERS[loader]
    embedding = _element_embedding()
    load(
        embedding,
        {"atomic_numbers": torch.tensor([3, 29, 79]), "weight": embedding.weight},
    )
    assert embedding(torch.tensor([79, 3])).index.tolist() == [2, 0]


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize(
    ("atomic_numbers", "message"),
    [
        ([6, 1, 8], "not strictly ascending"),
        ([1, 1, 6], "not strictly ascending"),
        ([0, 1, 6], r"holds \[0\], outside"),
        ([1.0, 6.7, 8.2], "not atomic numbers"),
    ],
    ids=["unsorted", "repeated", "padding", "float"],
)
def test_a_saved_table_that_is_not_a_models_is_refused_and_not_loaded(
    atomic_numbers, message, loader
):
    load, error = LOADERS[loader]
    embedding = _element_embedding()
    before = {name: tensor.clone() for name, tensor in embedding.state_dict().items()}
    with pytest.raises(error, match=f"atomic_numbers: .*{message}"):
        load(
            embedding,
            {
                "atomic_numbers": torch.tensor(atomic_numbers),
                "weight": torch.zeros_like(embedding.weight),
            },
        )
    for name, tensor in embedding.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_load_state_dict_leaves_a_table_that_is_not_a_tensor_to_torch():
    embedding = _element_embedding()
    with pytest.raises(
        RuntimeError, match=r'named "atomic_numbers", expected torch\.Tensor'
    ):
        embedding.load_state_dict(
            {"atomic_numbers": [1, 6, 8], "weight": embedding.weight}
        )


@fp64_only
def test_the_embedding_compiles_whole_at_any_node_count():
    from torch._dynamo import reset
    from torch._dynamo.testing import CompileCounter

    embedding = _element_embedding()
    reset()
    counter = CompileCounter()
    compiled = torch.compile(embedding, backend=counter, fullgraph=True, dynamic=True)
    for numbers in ([8, 1, 1], [6, 0, 8, 1, 1]):
        ours = compiled(torch.tensor(numbers))
        theirs = embedding(torch.tensor(numbers))
        assert torch.equal(ours.index, theirs.index)
        assert torch.equal(ours.features, theirs.features)
    assert counter.frame_count == 1


def test_a_layout_from_the_embeddings_table_refuses_an_element_outside_it():
    """The structure's elements are checked once, when its sample is built,
    against the table the model holds."""
    embedding = _element_embedding()
    layout = graph_layout(embedding.atomic_numbers.tolist(), cutoff=3.0)
    assert layout.atomic_numbers == (1, 6, 8)
    gold = Configuration(atomic_numbers=np.array([8, 79]), positions=np.zeros((2, 3)))
    with pytest.raises(
        SampleError, match=r"elements \[79\], .*element table \[1, 6, 8\]"
    ):
        graph_sample(gold, None, layout)


# ---------------------------------------------------------------------------
# RadialEmbeddingBlock: the two-tensor contract, --apply_cutoff, the three steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radial_basis", RADIAL_BASES)
@pytest.mark.parametrize("distance_transform", DISTANCE_TRANSFORMS)
def test_apply_cutoff_decides_whether_the_features_carry_the_envelope(
    radial_basis, distance_transform, dtype
):
    """Features `[n_edges, num_basis]` and envelope `[n_edges, 1]` in both
    modes, the envelope being the cutoff module on the raw lengths. The default
    (the CLI default) pre-multiplies; the other mode hands the bare basis over
    and the consumer's product is bit-for-bit the default."""
    lengths, node_atomic_numbers, edge_index = _embedding_inputs()
    assert _block().apply_cutoff is True
    applying = _block(radial_basis, distance_transform)
    deferring = _block(radial_basis, distance_transform, apply_cutoff=False)
    applied, applied_cutoff = applying(lengths, node_atomic_numbers, edge_index)
    bare, cutoff = deferring(lengths, node_atomic_numbers, edge_index)

    for block, features, envelope in (
        (applying, applied, applied_cutoff),
        (deferring, bare, cutoff),
    ):
        assert features.shape == (3, 4)
        assert envelope.shape == (3, 1)
        assert features.dtype == envelope.dtype == dtype
        assert block.num_basis == 4
        assert torch.equal(envelope, block.cutoff(lengths))

    assert torch.equal(applied, bare * cutoff)
    # the envelope is not all ones inside the cutoff
    assert not torch.equal(applied, bare)


@pytest.mark.parametrize("distance_transform", ["agnesi", "soft"])
def test_the_cutoff_is_computed_before_the_distance_transform(distance_transform):
    lengths, node_atomic_numbers, edge_index = _embedding_inputs()
    block = _block(distance_transform=distance_transform)
    _, cutoff = block(lengths, node_atomic_numbers, edge_index)

    reference = PolynomialCutoff(r_max=EMBEDDING_R_MAX, polynomial_order=6)
    transform = {"agnesi": AgnesiTransform, "soft": SoftTransform}[distance_transform]()
    transformed = transform(lengths, node_atomic_numbers, edge_index)

    assert torch.equal(cutoff, reference(lengths))
    assert not torch.allclose(cutoff, reference(transformed))


def test_the_basis_sees_the_transformed_lengths():
    lengths, node_atomic_numbers, edge_index = _embedding_inputs()
    block = _block(distance_transform="agnesi", apply_cutoff=False)
    features, _ = block(lengths, node_atomic_numbers, edge_index)
    transformed = AgnesiTransform()(lengths, node_atomic_numbers, edge_index)
    reference = BesselBasis(r_max=EMBEDDING_R_MAX, num_basis=4)
    assert torch.equal(features, reference(transformed))
    assert not torch.allclose(features, reference(lengths))


def test_a_padding_edge_contributes_exactly_nothing():
    """A self-loop edge shifted by 2*r_max has length 2*r_max and embeds to
    exactly zero: in the features in the default mode, through an exactly zero
    envelope in the other."""
    padded = torch.tensor([[2 * EMBEDDING_R_MAX]])
    node_atomic_numbers = torch.tensor([1])
    edge_index = torch.tensor([[0], [0]])
    features, cutoff = _block()(padded, node_atomic_numbers, edge_index)
    assert torch.equal(features, torch.zeros_like(features))
    assert torch.equal(cutoff, torch.zeros_like(cutoff))
    bare, cutoff = _block(apply_cutoff=False)(padded, node_atomic_numbers, edge_index)
    assert torch.equal(cutoff, torch.zeros_like(cutoff))
    assert (bare != 0).any()  # the bare basis itself does not vanish


# ---------------------------------------------------------------------------
# The builders: config section -> specification or module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "expected_module"),
    [
        (
            BesselBasisConfig(num_basis=5, trainable=True),
            lambda: BesselBasis(r_max=EMBEDDING_R_MAX, num_basis=5, trainable=True),
        ),
        (
            GaussianBasisConfig(num_basis=5, trainable=True),
            lambda: GaussianBasis(r_max=EMBEDDING_R_MAX, num_basis=5, trainable=True),
        ),
        (
            ChebyshevBasisConfig(num_basis=5, include_constant=True),
            lambda: ChebyshevBasis(num_basis=5, include_constant=True),
        ),
    ],
    ids=["bessel", "gaussian", "chebyshev"],
)
def test_a_basis_config_builds_the_module_it_names(config, expected_module):
    """Every field reaches the module through the specification: same class,
    same buffers and parameters, same numbers. The expected module is a factory
    so that it is built under the test's default dtype, not at collection."""
    built = _radial_op(config).basis
    _assert_same_module(built, expected_module())


@pytest.mark.parametrize(
    ("config", "expected_module"),
    [
        (
            AgnesiTransformConfig(
                exponent_q=1.1, exponent_p=4.0, amplitude=0.9, trainable=True
            ),
            lambda: AgnesiTransform(
                exponent_q=1.1, exponent_p=4.0, amplitude=0.9, trainable=True
            ),
        ),
        (
            SoftTransformConfig(steepness=3.0, trainable=True),
            lambda: SoftTransform(steepness=3.0, trainable=True),
        ),
    ],
    ids=["agnesi", "soft"],
)
def test_a_transform_config_builds_the_module_it_names(config, expected_module):
    _assert_same_module(build_distance_transform(config), expected_module())


def test_the_kind_none_builds_no_transform():
    """The transform slot is an explicit None, never an absent attribute
    probed with hasattr, and the block skips the step for it."""
    assert build_distance_transform(NoDistanceTransformConfig()) is None


def test_the_cutoff_config_builds_the_envelope_at_the_given_radius():
    built = _radial_op(
        BesselBasisConfig(), PolynomialCutoffConfig(polynomial_order=3), r_max=4.0
    ).cutoff
    _assert_same_module(built, PolynomialCutoff(r_max=4.0, polynomial_order=3))


def _assert_same_module(built, expected):
    assert type(built) is type(expected)
    assert built.extra_repr() == expected.extra_repr()
    built_state, expected_state = built.state_dict(), expected.state_dict()
    assert built_state.keys() == expected_state.keys()
    for name, tensor in expected_state.items():
        assert torch.equal(built_state[name], tensor), name
    assert [name for name, _ in built.named_parameters()] == [
        name for name, _ in expected.named_parameters()
    ]


@fp64_only
@pytest.mark.parametrize("radial_basis", RADIAL_BASES)
@pytest.mark.parametrize("distance_transform", DISTANCE_TRANSFORMS)
def test_radial_embedding_passes_gradgradcheck(radial_basis, distance_transform):
    """Force training differentiates twice through the whole block, in both
    of its outputs."""
    block = _block(radial_basis=radial_basis, distance_transform=distance_transform)
    _, node_atomic_numbers, edge_index = _embedding_inputs()
    lengths = torch.tensor([[0.9], [1.7], [2.5]], requires_grad=True)

    def function(lengths_):
        return block(lengths_, node_atomic_numbers, edge_index)

    assert torch.autograd.gradcheck(function, (lengths,))
    assert torch.autograd.gradgradcheck(function, (lengths,))
