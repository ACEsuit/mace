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
