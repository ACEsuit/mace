"""The reference backend: every factory, the force path, and the checkpoint.

The force path test is the one that matters. It differentiates an energy
through spherical harmonics, a tensor product and a scatter, with respect to
positions, and then differentiates that again. Every op in the chain has to
carry a differentiable backward or the second derivative is silently wrong, and
training on forces is exactly that second derivative.
"""

import importlib.util
from pathlib import Path

import pytest

# `tomllib` is 3.11+, and this file runs on the 3.10 leg of the matrix too.
# pytest declares `tomli` there, and this file is collected by pytest or not at
# all, so the fallback always resolves; a bare import breaks collection.
try:  # pragma: no cover - one branch per interpreter
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("the reference backend needs torch", allow_module_level=True)

import torch
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

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


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
    third-party one: `mace_core` names no backend anywhere."""
    metadata = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())
    group = metadata["project"]["entry-points"]["mace.kernel_backends.torch"]
    assert group["reference"] == "mace_torch.backends.reference:ReferenceBackend"

    module_name, _, attribute = group["reference"].partition(":")
    module = importlib.import_module(module_name)
    assert getattr(module, attribute) is ReferenceBackend


def test_the_skip_connection_refuses_a_non_scalar_second_input(backend):
    """A wrong skip connection is a wrong model that still trains, so the
    unbuilt case raises instead of returning something plausible."""
    with pytest.raises(NotImplementedError, match="only against scalars"):
        backend.make_fully_connected_tp(
            FullyConnectedTPDescriptor(
                irreps_in1="4x0e", irreps_in2="1x1o", irreps_out="4x0e"
            )
        )
