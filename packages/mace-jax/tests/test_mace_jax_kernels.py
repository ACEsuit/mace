"""Independent small algebra checks for the framework-generic kernel binding."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mace_core.clebsch_gordan.reduced_basis import (
    reduced_symmetric_tensor_product_basis,
)
from mace_core.kernels.capabilities import UnsupportedKernelSpecError
from mace_core.kernels.protocol import (
    DISPATCHED_OPS,
    REFERENCE_ONLY_OPS,
    InternalWeights,
    KernelBackend,
)
from mace_core.kernels.specs import (
    ChannelwiseTPConvSpec,
    FullyConnectedTPSpec,
    LinearSpec,
    RadialBasisSpec,
    RadialMLPSpec,
    SegmentReduceSpec,
    SphericalHarmonicsSpec,
    SymmetricContractionSpec,
)
from mace_jax.kernels.reference import ReferenceBackend


@pytest.fixture
def backend():
    jax.config.update("jax_enable_x64", True)
    return ReferenceBackend()


def test_protocol_and_raw_linear_skip_weights(backend):
    assert isinstance(backend, KernelBackend)
    assert backend.capabilities().ops == DISPATCHED_OPS | REFERENCE_ONLY_OPS
    linear = backend.make_linear(LinearSpec(irreps_in="2x0e", irreps_out="0e"))
    assert isinstance(linear, InternalWeights)
    linear.load_canonical({"weight": np.asarray([2.0, 3.0]), "bias": np.empty(0)})
    np.testing.assert_allclose(linear(jnp.asarray([[4.0, 5.0]])), [[23 / math.sqrt(2)]])
    skip = backend.make_fully_connected_tp(
        FullyConnectedTPSpec(irreps_in1="2x0e", irreps_in2="3x0e", irreps_out="0e")
    )
    skip.load_canonical({"weight": np.asarray([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])})
    np.testing.assert_allclose(
        skip(jnp.asarray([[4.0, 5.0]]), jnp.asarray([[0.0, 1.0, 0.0]])),
        [[41 / math.sqrt(6)]],
    )
    np.testing.assert_array_equal(linear.to_canonical()["weight"], [2, 3])


def test_empty_edge_reductions_keep_explicit_node_count(backend):
    reduction = backend.make_segment_reduce(SegmentReduceSpec())
    np.testing.assert_array_equal(
        reduction(jnp.zeros((0, 2)), jnp.zeros(0, dtype=int), 4), np.zeros((4, 2))
    )
    kernel_spec = ChannelwiseTPConvSpec(
        irreps_node="0e", irreps_edge="0e+1o", irreps_out="0e+1o", num_features=2
    )
    convolution = backend.make_channelwise_tp_conv(kernel_spec)
    result = convolution(
        jnp.ones((4, 2)),
        jnp.empty((0, 4)),
        jnp.empty((0, len(kernel_spec.paths), 2)),
        jnp.empty(0, dtype=int),
        jnp.empty(0, dtype=int),
        4,
    )
    np.testing.assert_array_equal(result, np.zeros((4, 8)))
    np.testing.assert_array_equal(
        reduction(jnp.asarray([[2.0], [3.0]]), jnp.asarray([0, 0]), 4),
        [[5], [0], [0], [0]],
    )


def test_compressed_contraction_matches_dense_couplings_and_gradient(backend):
    kernel_spec = SymmetricContractionSpec(
        irreps_in="0e+1o",
        irreps_out="0e",
        correlation=3,
        num_elements=1,
        num_features=1,
    )
    contraction = backend.make_symmetric_contraction(kernel_spec)
    contraction.initialize_weights(42)
    weights = np.asarray(contraction.to_canonical()["weight"])[0, :, 0]
    features = np.asarray([0.7, -0.2, 0.5, 1.1])

    def dense(x):
        tensor, offset, total = np.array(1.0), 0, 0.0
        for order in range(1, 4):
            tensor = np.multiply.outer(tensor, x)
            basis = reduced_symmetric_tensor_product_basis("0e+1o", order, "0e")["0e"]
            projected = basis.reshape(len(basis), -1) @ tensor.ravel()
            total += weights[offset : offset + len(basis)] @ projected
            offset += len(basis)
        return total

    def function(x):
        return contraction(x[None, :], jnp.asarray([0]))[0, 0]

    np.testing.assert_allclose(
        function(jnp.asarray(features)), dense(features), atol=1e-12
    )
    finite = [
        (dense(features + 1e-5 * row) - dense(features - 1e-5 * row)) / 2e-5
        for row in np.eye(4)
    ]
    np.testing.assert_allclose(
        jax.grad(function)(jnp.asarray(features)), finite, atol=1e-9
    )


@pytest.mark.parametrize("normalize", [False, True])
def test_spherical_harmonic_normalization_and_parity(backend, normalize):
    harmonic = backend.make_spherical_harmonics(
        SphericalHarmonicsSpec(lmax=3, normalize=normalize)
    )
    x = jnp.asarray([[0.3, 0.8, -0.4]])
    positive, negative = harmonic(x), harmonic(-x)
    for degree in range(4):
        block = slice(degree**2, (degree + 1) ** 2)
        norm = 1 if normalize else float(jnp.sum(x * x)) ** degree
        np.testing.assert_allclose(
            jnp.sum(positive[:, block] ** 2), (2 * degree + 1) * norm, atol=1e-12
        )
        np.testing.assert_allclose(
            negative[:, block], (-1) ** degree * positive[:, block], atol=1e-12
        )


@pytest.mark.parametrize(
    "kernel_spec,factory",
    [
        (LinearSpec(precision="float32", accumulate="float64"), "make_linear"),
        (SegmentReduceSpec(reduction="mean"), "make_segment_reduce"),
        (SegmentReduceSpec(reduction="max"), "make_segment_reduce"),
        (RadialBasisSpec(kind="gaussian", num_basis=4), "make_radial_basis"),
        (
            RadialMLPSpec(kind="layer_norm_silu", channels=(2, 3)),
            "make_radial_mlp",
        ),
        (SymmetricContractionSpec(basis="full"), "make_symmetric_contraction"),
        (FullyConnectedTPSpec(irreps_in2="1o"), "make_fully_connected_tp"),
    ],
)
def test_capability_and_factory_refuse_the_same_unsupported_descriptor(
    backend, kernel_spec, factory
):
    assert not backend.capabilities().supports(kernel_spec)
    with pytest.raises(UnsupportedKernelSpecError):
        getattr(backend, factory)(kernel_spec)


def test_radial_origin_and_cutoff_have_finite_derivatives(backend):
    radial = backend.make_radial_basis(RadialBasisSpec(num_basis=4, cutoff=5.0))

    def function(radius):
        return jnp.sum(radial(jnp.asarray([[radius]])))

    assert np.isfinite(function(0.0)) and np.isfinite(jax.grad(function)(0.0))
    np.testing.assert_array_equal(function(5.0), 0)
    np.testing.assert_allclose(jax.grad(function)(5.0), 0, atol=1e-12)


@pytest.mark.parametrize("apply_cutoff", [True, False])
def test_radial_accepts_the_raw_distance_envelope(backend, apply_cutoff):
    from dataclasses import replace

    kernel_spec = RadialBasisSpec(num_basis=4, cutoff=5.0, apply_cutoff=False)
    lengths = jnp.asarray([[0.7], [1.2]])
    envelope = jnp.asarray([[0.3], [0.8]])
    bare = backend.make_radial_basis(kernel_spec)(lengths)
    radial = backend.make_radial_basis(replace(kernel_spec, apply_cutoff=apply_cutoff))
    actual = radial(lengths, envelope=envelope)
    np.testing.assert_array_equal(actual, bare * envelope if apply_cutoff else bare)
