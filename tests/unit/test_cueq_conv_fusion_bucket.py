"""Node-bucket padding for cuEq conv fusion.

With convolution fusion enabled, the fused segmented polynomial specializes
on the node count it sees. Node counts vary batch to batch, so unpadded
calls would specialize per batch shape. The wrapper pads the node axis to a
multiple of ``node_bucket`` (default 256, ``None`` disables) so compiled
graphs specialize per bucket instead. These tests pin the contract the
padding must uphold: identical numerics with and without padding, output
sliced back to the true node count, and rejection of invalid buckets.
"""

import pytest
import torch

cue = pytest.importorskip("cuequivariance")
cuet = pytest.importorskip("cuequivariance_torch")

from mace.tools.cg import O3_e3nn  # noqa: E402
from mace.modules.wrapper_ops import (  # noqa: E402
    CuEquivarianceConfig,
    CueqConvFusionWrapper,
    with_cueq_conv_fusion,
)

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="uniform_1d kernels need CUDA"
)


def _make_fused(node_bucket):
    from e3nn import o3

    irreps_in1 = o3.Irreps("16x0e")
    irreps_in2 = o3.Irreps("1x0e + 1x1o + 1x2e")
    irreps_out = o3.Irreps("16x0e")
    polynomial = (
        cue.descriptors.channelwise_tensor_product(
            cue.Irreps(O3_e3nn, irreps_in1),
            cue.Irreps(O3_e3nn, irreps_in2),
            cue.Irreps(O3_e3nn, irreps_out),
        )
        .flatten_coefficient_modes()
        .squeeze_modes()
        .polynomial
    )
    conv_tp = cuet.SegmentedPolynomial(
        polynomial,
        math_dtype=torch.get_default_dtype(),
        method="uniform_1d",
    )
    return with_cueq_conv_fusion(conv_tp, polynomial, node_bucket=node_bucket)


def _run(wrapper, num_nodes, num_edges, seed=1702):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    node_feats = torch.randn(
        num_nodes, 16, device=device, dtype=torch.get_default_dtype()
    )
    edge_attrs = torch.randn(
        num_edges, 9, device=device, dtype=torch.get_default_dtype()
    )
    tp_weights = torch.randn(
        num_edges, wrapper.weight_numel, device=device, dtype=torch.get_default_dtype()
    ) * 0.05
    edge_index = torch.stack(
        [
            torch.randint(0, num_nodes, (num_edges,), device=device),
            torch.randint(0, num_nodes, (num_edges,), device=device),
        ]
    )
    out = wrapper(node_feats, edge_attrs, tp_weights, edge_index)
    torch.cuda.synchronize()
    return out


@CUDA
@pytest.mark.parametrize("num_nodes", [10, 64, 100])
def test_bucketed_fusion_matches_unpadded(num_nodes):
    """Padding to the bucket must not change a single number."""
    padded = _make_fused(node_bucket=64).to(torch.float64)
    plain = _make_fused(node_bucket=None).to(torch.float64)

    out_padded = _run(padded, num_nodes, num_edges=128)
    out_plain = _run(plain, num_nodes, num_edges=128)

    assert out_padded.shape == (num_nodes, 16)
    torch.testing.assert_close(out_padded, out_plain)


@CUDA
def test_bucket_rounds_up_to_multiple():
    """The wrapper must call the polynomial with a bucket-aligned node count."""
    wrapper = _make_fused(node_bucket=64)
    seen = []
    original = wrapper.conv_tp.forward

    def spy(*args, **kwargs):
        seen.append(args[0][1].shape[0])
        return original(*args, **kwargs)

    wrapper.conv_tp.forward = spy
    _run(wrapper, num_nodes=10, num_edges=32)
    assert seen == [64]


def test_invalid_buckets_are_rejected():
    with pytest.raises(ValueError, match="node_bucket must be positive"):
        CueqConvFusionWrapper.__init__(
            object.__new__(CueqConvFusionWrapper), conv_tp=None,
            polynomial=None, node_bucket=0,
        )
    with pytest.raises(ValueError, match="conv_fusion_node_bucket"):
        CuEquivarianceConfig(enabled=True, conv_fusion_node_bucket=-4)


def test_none_bucket_disables_padding():
    """node_bucket=None stores zero, which skips the pad branch entirely."""
    wrapper = _make_fused(node_bucket=None)
    assert wrapper.node_bucket == 0

def test_old_pickles_gain_default_bucket():
    wrapper = CueqConvFusionWrapper.__new__(CueqConvFusionWrapper)
    state = {"conv_tp": torch.nn.Identity()}
    wrapper.__setstate__(state)
    assert wrapper.node_bucket == 256
