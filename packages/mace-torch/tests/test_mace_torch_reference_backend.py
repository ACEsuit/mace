"""The reference backend: every factory, the force path, and the checkpoint.

The force path test is the one that matters. It differentiates an energy
through spherical harmonics, a tensor product and a scatter, with respect to
positions, and then differentiates that again. Every op in the chain has to
carry a differentiable backward or the second derivative is silently wrong, and
training on forces is exactly that second derivative.
"""

import importlib.metadata
import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("the reference backend needs torch", allow_module_level=True)

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels import (
    ChannelwiseTPConvDescriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    RadialBasisDescriptor,
    SegmentReduceDescriptor,
    SphericalHarmonicsDescriptor,
    SymmetricContractionDescriptor,
    UnsupportedDescriptorError,
)
from mace_torch.backends.reference import ReferenceBackend


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture
def backend():
    return ReferenceBackend()


# ---------------------------------------------------------------------------
# What it declares
# ---------------------------------------------------------------------------


def test_it_declares_every_op_and_a_second_derivative(backend):
    capabilities = backend.capabilities()
    assert "symmetric_contraction" in capabilities.ops
    assert "spherical_harmonics" in capabilities.ops
    assert capabilities.supports_double_backward
    assert "mul_ir" in capabilities.layouts


def test_it_refuses_a_precision_it_did_not_declare(backend):
    with pytest.raises(UnsupportedDescriptorError, match="does not support"):
        backend.make_linear(LinearDescriptor(precision="bfloat16"))


def test_it_fuses_nothing_and_says_so_rather_than_failing(backend):
    """`None` from the span factory is the normal answer: build the layer op by
    op. A backend that could fuse a whole layer returns one here."""
    assert backend.make_interaction_layer(()) is None


# ---------------------------------------------------------------------------
# The linear map
# ---------------------------------------------------------------------------


def test_the_linear_weight_count_matches_what_the_descriptor_promised(backend):
    descriptor = LinearDescriptor(
        irreps_in="4x0e+2x1o", irreps_out="3x0e+2x1o", has_bias=True
    )
    built = backend.make_linear(descriptor)
    assert built.weight.numel() + built.bias.numel() == descriptor.weight_numel


def test_the_linear_map_does_not_mix_one_irrep_into_another(backend):
    """The equivariance, stated as the thing it forbids: however the weights are
    set, a 1o input cannot move a 0e output."""
    built = backend.make_linear(
        LinearDescriptor(irreps_in="2x0e+2x1o", irreps_out="2x0e+2x1o")
    )
    with torch.no_grad():
        built.weight.uniform_(-1, 1)
    scalars_only = torch.zeros(1, 8)
    scalars_only[0, :2] = torch.randn(2)
    vectors_only = torch.zeros(1, 8)
    vectors_only[0, 2:] = torch.randn(6)
    assert torch.allclose(built(vectors_only)[0, :2], torch.zeros(2))
    assert torch.allclose(built(scalars_only)[0, 2:], torch.zeros(6))


def test_a_bias_lands_only_on_the_scalars(backend):
    built = backend.make_linear(
        LinearDescriptor(irreps_in="2x0e", irreps_out="2x0e+1x1o", has_bias=True)
    )
    with torch.no_grad():
        built.bias.fill_(3.0)
    out = built(torch.zeros(1, 2))
    assert torch.allclose(out[0, :2], torch.full((2,), 3.0))
    assert torch.allclose(out[0, 2:], torch.zeros(3))


# ---------------------------------------------------------------------------
# The force path, differentiated twice
# ---------------------------------------------------------------------------


@pytest.fixture
def chain(backend):
    """A miniature of the real pipeline, built once.

    Built once and reused on purpose: gradcheck calls the function many times
    and compares the results, so a chain that re-randomized its weights on
    every call would be a different function each time. The first version of
    this test did exactly that and failed for that reason rather than for a
    wrong derivative.
    """
    harmonics = backend.make_spherical_harmonics(SphericalHarmonicsDescriptor(lmax=1))
    radial = backend.make_radial_basis(
        RadialBasisDescriptor(kind="bessel", num_basis=8, cutoff=5.0)
    )
    convolution = backend.make_channelwise_tp_conv(
        ChannelwiseTPConvDescriptor(
            irreps_node="0e+1o", irreps_edge="0e+1o", irreps_out="0e+1o"
        )
    )
    contraction = backend.make_symmetric_contraction(
        SymmetricContractionDescriptor(
            irreps_in="0e+1o",
            irreps_out="0e",
            correlation=2,
            num_elements=1,
            num_features=2,
        )
    )
    reduce = backend.make_segment_reduce(SegmentReduceDescriptor())
    with torch.no_grad():
        for parameter in contraction.weights:
            parameter.uniform_(-1, 1)

    def energy(positions, sender, receiver):
        vectors = positions[receiver] - positions[sender]
        lengths = vectors.norm(dim=-1, keepdim=True)
        attributes = harmonics(vectors)
        # The radial MLP is external to the op, so its width has to match the
        # number of coupling paths the convolution declares. Reading it off the
        # built op is the point of exposing `num_paths`.
        weights = (
            radial(lengths)[:, : convolution.num_paths].unsqueeze(-1).expand(-1, -1, 2)
        )
        features = torch.zeros(positions.shape[0], 2, 4, dtype=positions.dtype)
        features[:, :, 0] = 1.0
        messages = convolution(
            features, attributes, weights, sender, receiver, positions.shape[0]
        )
        nodes = torch.zeros(positions.shape[0], dtype=torch.long)
        site = contraction(messages, nodes)
        return reduce(site.flatten(1), nodes, 1).sum()

    return energy


def test_forces_come_out_of_the_chain_and_differentiate_twice(chain):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.3, 0.0]], requires_grad=True
    )
    sender = torch.tensor([0, 1, 0, 2, 1, 2])
    receiver = torch.tensor([1, 0, 2, 0, 2, 1])

    energy = chain(positions, sender, receiver)
    (forces,) = torch.autograd.grad(energy, positions, create_graph=True)
    assert forces.shape == positions.shape
    assert torch.isfinite(forces).all()

    # The second derivative, which is what training on forces needs. A backward
    # that is not itself differentiable would fail here rather than give a
    # wrong number, which is the whole reason for the check.
    (second,) = torch.autograd.grad(forces.pow(2).sum(), positions)
    assert torch.isfinite(second).all()
    assert second.abs().max() > 0

    def total(p):
        return chain(p, sender, receiver)

    assert torch.autograd.gradcheck(total, (positions,))
    assert torch.autograd.gradgradcheck(total, (positions,))


def test_the_forces_sum_to_zero(chain):
    """Newton's third law, which the chain has to satisfy for free: the energy
    depends on the positions only through differences, so the gradient sums to
    zero. A broken scatter or a mixed-up sender and receiver breaks it."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.3, 0.0]], requires_grad=True
    )
    sender = torch.tensor([0, 1, 0, 2, 1, 2])
    receiver = torch.tensor([1, 0, 2, 0, 2, 1])
    energy = chain(positions, sender, receiver)
    (forces,) = torch.autograd.grad(energy, positions)
    assert forces.sum(0).abs().max() < 1e-10


# ---------------------------------------------------------------------------
# The checkpoint boundary
# ---------------------------------------------------------------------------


def test_the_canonical_weights_round_trip_through_a_fresh_instance(backend):
    descriptor = SymmetricContractionDescriptor(
        irreps_in="0e+1o",
        irreps_out="0e",
        correlation=2,
        num_elements=2,
        num_features=3,
    )
    written = backend.make_symmetric_contraction(descriptor)
    with torch.no_grad():
        for parameter in written.weights:
            parameter.uniform_(-1, 1)
    features = torch.randn(4, 3, 4)
    element = torch.tensor([0, 1, 0, 1])
    expected = written(features, element)

    state = written.to_canonical()
    assert state["weight"].shape[1] == descriptor.path_count

    read = backend.make_symmetric_contraction(descriptor)
    assert not torch.allclose(read(features, element), expected)
    read.load_canonical(state)
    assert torch.allclose(read(features, element), expected)


def test_the_linear_weights_round_trip_too(backend):
    descriptor = LinearDescriptor(
        irreps_in="4x0e", irreps_out="2x0e+1x1o", has_bias=True
    )
    written = backend.make_linear(descriptor)
    with torch.no_grad():
        written.weight.uniform_(-1, 1)
        written.bias.uniform_(-1, 1)
    features = torch.randn(3, 4)
    read = backend.make_linear(descriptor)
    read.load_canonical(written.to_canonical())
    assert torch.allclose(read(features), written(features))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_the_reference_is_declared_as_an_entry_point_that_resolves():
    """Discovery must work for the mandatory backend exactly as it does for a
    third-party one: `mace_core` names no backend anywhere.

    Read from the installed metadata rather than from `pyproject.toml`, because
    what the registry searches is what pip wrote. A declaration this package
    ships and the install does not carry would pass a reading of the file and
    fail every discovery.
    """
    entries = importlib.metadata.entry_points(group="mace.kernel_backends.torch")
    declared = {entry.name: entry for entry in entries}
    assert "reference" in declared, sorted(declared)
    assert declared["reference"].value == (
        "mace_torch.backends.reference:ReferenceBackend"
    )
    assert declared["reference"].load() is ReferenceBackend


def test_the_skip_connection_refuses_a_non_scalar_second_input(backend):
    """A wrong skip connection is a wrong model that still trains, so the
    unbuilt case raises instead of returning something plausible."""
    with pytest.raises(NotImplementedError, match="only against scalars"):
        backend.make_fully_connected_tp(
            FullyConnectedTPDescriptor(
                irreps_in1="4x0e", irreps_in2="1x1o", irreps_out="4x0e"
            )
        )


# ---------------------------------------------------------------------------
# Output irreps the contraction has to cover
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("irreps_out", ["0e", "1o", "0e+1o", "0e+1o+2e", "1o+1e"])
def test_the_contraction_builds_every_output_the_layers_ask_for(backend, irreps_out):
    """A mixed output is the normal case, not an exotic one.

    Each output irrep has its own component count, so one stacked basis over
    all of them is a join of arrays whose second axis differs: `0e+1o` did not
    build at all. Two same-sized irreps are worse than that, because they join
    without complaint and then write into one shared slice instead of adjacent
    ones, which is a model that trains and is wrong.
    """
    descriptor = SymmetricContractionDescriptor(
        irreps_in="0e+1o",
        irreps_out=irreps_out,
        correlation=3,
        num_elements=2,
        num_features=4,
    )
    operation = backend.make_symmetric_contraction(descriptor)
    operation.initialize_weights(3)
    features = torch.randn(5, 4, Irreps.parse("0e+1o").dimension)
    elements = torch.randint(0, 2, (5,))
    assert operation(features, elements).shape == (
        5,
        4,
        Irreps.parse(irreps_out).dimension,
    )


def test_an_output_irrep_no_body_order_reaches_is_still_built(backend):
    """`2e` is unreachable from `0e+1o` at body order one, so that order's
    basis is empty. Inferring its trailing extent with `-1` cannot work on an
    empty array, and the whole contraction failed to build over it."""
    descriptor = SymmetricContractionDescriptor(
        irreps_in="0e+1o",
        irreps_out="2e",
        correlation=2,
        num_elements=1,
        num_features=2,
    )
    operation = backend.make_symmetric_contraction(descriptor)
    assert operation.weights[0].shape[1] == 0
    assert operation.weights[1].shape[1] > 0


def test_the_skip_connection_advertises_the_weights_it_holds(backend):
    """The descriptor's count is what a capability filter reads and what sizes
    a checkpoint, and it used to be whatever the caller passed."""
    descriptor = FullyConnectedTPDescriptor(
        irreps_in1="4x0e+4x1o", irreps_in2="2x0e", irreps_out="4x0e+4x1o"
    )
    operation = backend.make_fully_connected_tp(descriptor)
    assert descriptor.weight_numel == operation.weight.numel()
    assert descriptor.weight_numel == operation.to_canonical()["weight"].numel()
