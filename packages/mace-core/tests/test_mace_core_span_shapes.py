"""The span shape is a closed, framework-free tuple contract."""

import itertools
import subprocess
import sys

import pytest
from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.spans import EdgeConvolutionSpan
from mace_core.kernels.specs import (
    ChannelwiseTPConvSpec,
    DistanceTransformSpec,
    RadialBasisSpec,
    RadialMLPSpec,
    SphericalHarmonicsSpec,
)


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("with_density", [False, True])
def test_extended_shapes_keep_network_order_and_embedding_dimensions(
    conditioned, with_density
):
    from dataclasses import replace

    network = RadialMLPSpec(
        channels=(12 if conditioned else 8, 4, 16),
        num_elements=3 if conditioned else 0,
        element_embedding_width=2 if conditioned else 0,
        layer_norm_epsilon=1e-4,
    )
    density = (
        replace(network, channels=(network.channels[0], 3, 1)) if with_density else None
    )
    shape = EdgeConvolutionSpan(
        RadialBasisSpec(),
        SphericalHarmonicsSpec(),
        network,
        ChannelwiseTPConvSpec(),
        density=density,
    )
    assert EdgeConvolutionSpan.from_specs(shape.kernel_specs) == shape
    assert hash(shape)
    if density is not None:
        assert shape.kernel_specs[-3:] == (network, density, shape.convolution)
        with pytest.raises(ValueError, match="one scalar"):
            replace(shape, density=network)
        reordered = (*shape.kernel_specs[:2], density, network, shape.convolution)
        assert EdgeConvolutionSpan.from_specs(reordered) is None


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
        RadialMLPSpec(**settings)


@pytest.mark.parametrize("transform", [None, DistanceTransformSpec()])
def test_only_the_exact_shape_round_trips(transform):
    shape = EdgeConvolutionSpan(
        RadialBasisSpec(),
        SphericalHarmonicsSpec(),
        RadialMLPSpec(),
        ChannelwiseTPConvSpec(),
        transform,
    )
    assert EdgeConvolutionSpan.from_specs(shape.kernel_specs) == shape
    for members in itertools.permutations(shape.kernel_specs):
        if members != shape.kernel_specs:
            assert EdgeConvolutionSpan.from_specs(members) is None
    assert EdgeConvolutionSpan.from_specs(()) is None
    assert (
        EdgeConvolutionSpan.from_specs(
            (*shape.kernel_specs, RadialMLPSpec()),
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


@pytest.mark.parametrize("version", ["1.1", "1.2", "1.3", "2.0", "typo", 1, None])
def test_other_kernel_contract_versions_are_refused(version):
    with pytest.raises(ValueError, match="kernel spec_version"):
        BackendCapabilities(spec_version=version)


def test_initial_contract_includes_stateless_spans():
    assert BackendCapabilities().spec_version == "1.0"
