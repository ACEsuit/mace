"""The span shape is a closed, framework-free tuple contract."""

import itertools
import subprocess
import sys

import pytest
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    DistanceTransformDescriptor,
    RadialBasisDescriptor,
    RadialMLPDescriptor,
    SphericalHarmonicsDescriptor,
)
from mace_core.kernels.spans import EdgeConvolutionSpan, supports_span_contract


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("with_density", [False, True])
def test_extended_shapes_keep_network_order_and_embedding_dimensions(
    conditioned, with_density
):
    from dataclasses import replace

    network = RadialMLPDescriptor(
        channels=(12 if conditioned else 8, 4, 16),
        num_elements=3 if conditioned else 0,
        element_embedding_width=2 if conditioned else 0,
        layer_norm_epsilon=1e-4,
    )
    density = (
        replace(network, channels=(network.channels[0], 3, 1)) if with_density else None
    )
    shape = EdgeConvolutionSpan(
        RadialBasisDescriptor(),
        SphericalHarmonicsDescriptor(),
        network,
        ChannelwiseTPConvDescriptor(),
        density=density,
    )
    assert EdgeConvolutionSpan.from_descriptors(shape.descriptors) == shape
    assert hash(shape)
    if density is not None:
        assert shape.descriptors[-3:] == (network, density, shape.convolution)
        with pytest.raises(ValueError, match="one scalar"):
            replace(shape, density=network)
        reordered = (*shape.descriptors[:2], density, network, shape.convolution)
        assert EdgeConvolutionSpan.from_descriptors(reordered) is None


@pytest.mark.parametrize(
    "settings",
    [
        {"num_elements": 2},
        {"element_embedding_width": 3},
        {"num_elements": -1},
        {"num_elements": True},
        {"num_elements": 2, "element_embedding_width": 4},
        {"layer_norm_epsilon": 0},
        {"layer_norm_epsilon": float("nan")},
    ],
)
def test_invalid_conditioning_is_rejected_before_build(settings):
    with pytest.raises(ValueError):
        RadialMLPDescriptor(**settings)


@pytest.mark.parametrize("transform", [None, DistanceTransformDescriptor()])
def test_only_the_exact_shape_round_trips(transform):
    shape = EdgeConvolutionSpan(
        RadialBasisDescriptor(),
        SphericalHarmonicsDescriptor(),
        RadialMLPDescriptor(),
        ChannelwiseTPConvDescriptor(),
        transform,
    )
    assert EdgeConvolutionSpan.from_descriptors(shape.descriptors) == shape
    for members in itertools.permutations(shape.descriptors):
        if members != shape.descriptors:
            assert EdgeConvolutionSpan.from_descriptors(members) is None
    assert EdgeConvolutionSpan.from_descriptors(()) is None
    assert (
        EdgeConvolutionSpan.from_descriptors(
            (*shape.descriptors, RadialMLPDescriptor()),
        )
        is None
    )


def test_the_core_shape_imports_no_framework():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mace_core.kernels.spans; "
            "assert not {'torch', 'jax', 'e3nn'} & sys.modules.keys()",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "version, supported",
    [
        ("1.1", False),
        ("1.0", False),
        ("1.2", True),
        ("1.3", True),
        ("2.0", False),
        ("typo", False),
    ],
)
def test_span_version_negotiation(version, supported):
    assert supports_span_contract(version) is supported
