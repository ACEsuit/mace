"""The convolution's paths: which couplings, in which order."""

import pytest
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels import (
    ChannelwiseTPConvDescriptor,
    channelwise_paths,
    path_irreps,
)


def test_couplings_onto_one_irrep_stay_apart_in_generation_order():
    paths = channelwise_paths("0e+1o", "0e+1o", "0e+1o")
    assert [(p.node_term, p.edge_term, str(p.irrep)) for p in paths] == [
        (0, 0, "0e"),
        (1, 1, "0e"),
        (0, 1, "1o"),
        (1, 0, "1o"),
    ]
    assert path_irreps(paths) == "1x0e+1x0e+1x1o+1x1o"


def test_a_coupling_outside_the_target_is_dropped():
    paths = channelwise_paths("1o", "1o", "0e+1e")
    assert [str(p.irrep) for p in paths] == ["0e", "1e"]


def test_odd_comes_before_even_at_one_degree():
    paths = channelwise_paths("0e+0o", "1o", "1o+1e")
    assert [str(p.irrep) for p in paths] == ["1o", "1e"]


def test_the_descriptor_declares_its_message_one_term_per_path():
    descriptor = ChannelwiseTPConvDescriptor(
        irreps_node="0e+1o", irreps_edge="0e+1o+2e", irreps_out="0e+1o+2e"
    )
    assert len(Irreps.parse(descriptor.irreps_paths).terms) == len(descriptor.paths)


def test_a_node_multiplicity_is_refused_in_favour_of_the_channel_count():
    with pytest.raises(ValueError, match="num_features"):
        ChannelwiseTPConvDescriptor(irreps_node="2x0e", irreps_edge="0e")


def test_a_channel_count_below_one_is_refused():
    with pytest.raises(ValueError, match="num_features"):
        ChannelwiseTPConvDescriptor(num_features=0)
