"""Inference fusion boundaries are explicit, immutable and framework-free."""

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest
from mace_core.kernels.inference import InteractionFusionSpec, ProductReadoutSpec
from mace_core.kernels.spans import EdgeConvolutionSpan
from mace_core.kernels.specs import (
    ChannelwiseTPConvSpec,
    FullyConnectedTPSpec,
    LinearSpec,
    RadialBasisSpec,
    RadialMLPSpec,
    SphericalHarmonicsSpec,
    SymmetricContractionSpec,
)


def shape():
    product = ProductReadoutSpec(
        SymmetricContractionSpec(
            irreps_in="0e", irreps_out="0e", num_features=2, num_elements=2
        ),
        LinearSpec(irreps_in="2x0e", irreps_out="2x0e"),
        residual=True,
        readout=LinearSpec(irreps_in="2x0e", irreps_out="0e"),
    )
    edge = EdgeConvolutionSpan(
        RadialBasisSpec(num_basis=3),
        SphericalHarmonicsSpec(lmax=0),
        RadialMLPSpec(channels=(3, 4, 2)),
        ChannelwiseTPConvSpec(
            irreps_node="0e", irreps_edge="0e", irreps_out="0e", num_features=2
        ),
    )
    return InteractionFusionSpec(
        edge,
        product.linear,
        product.linear,
        FullyConnectedTPSpec(irreps_in1="2x0e", irreps_in2="2x0e", irreps_out="2x0e"),
        product,
    )


def test_closed_shapes_declare_boundaries_and_canonical_roles():
    spec = shape()
    assert hash(spec)
    with pytest.raises(FrozenInstanceError):
        spec.product = None
    assert spec.product is not None
    assert spec.product.input_names == ("message", "element", "residual")
    assert spec.output_names == ("features", "readout")
    assert "normalization" in spec.weight_roles
    assert "product.readout" in spec.weight_roles
    shared = replace(
        spec.product, contraction=replace(spec.product.contraction, num_elements=1)
    )
    assert replace(spec, product=shared).product.contraction.num_elements == 1
    density = replace(spec.edge, density=RadialMLPSpec(channels=(3, 1)))
    assert "density" in replace(spec, edge=density).weight_roles
    assert "normalization" not in replace(spec, edge=density).weight_roles


@pytest.mark.parametrize(
    "change",
    [
        lambda s: replace(s, linear_up=replace(s.linear_up, irreps_out="3x0e")),
        lambda s: replace(s, linear_down=replace(s.linear_down, irreps_in="3x0e")),
        lambda s: replace(s, skip=replace(s.skip, irreps_out="3x0e")),
        lambda s: replace(s, skip=replace(s.skip, irreps_in2="1o")),
        lambda s: replace(
            s,
            product=replace(
                s.product, contraction=replace(s.product.contraction, num_elements=3)
            ),
        ),
        lambda s: replace(
            s,
            edge=replace(
                s.edge,
                radial_network=replace(s.edge.radial_network, channels=(3, 4, 4)),
            ),
        ),
        lambda s: replace(
            s,
            edge=replace(
                s.edge,
                radial_network=replace(s.edge.radial_network, channels=(4, 4, 2)),
            ),
        ),
        lambda s: replace(
            s, linear_up=replace(s.linear_up, activation_layout="ir_mul")
        ),
        lambda s: replace(
            s.product, readout=replace(s.product.readout, irreps_in="3x0e")
        ),
        lambda s: replace(
            s.product, contraction=replace(s.product.contraction, basis="full")
        ),
        lambda s: replace(s.product, residual="true"),
    ],
)
def test_incompatible_shapes_are_rejected(change):
    with pytest.raises((TypeError, ValueError)):
        change(shape())


def test_imports_no_framework_and_does_not_extend_required_backend_methods():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mace_core.kernels.inference; "
            "from mace_core.kernels.protocol import KernelBackend; "
            "assert not hasattr(KernelBackend, 'prepare_inference'); "
            "assert not {'torch', 'jax', 'e3nn'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_interface_precision_does_not_overwrite_member_compute_policies():
    spec = shape()
    mixed = replace(spec.product, interface_dtype="float32")
    assert mixed.interface_dtype == "float32"
    assert all(member.precision == "float64" for member in mixed.members)
    assert replace(spec, product=mixed).interface_dtype == "float32"
    with pytest.raises(ValueError, match="interface_dtype"):
        replace(mixed, interface_dtype="typo")
