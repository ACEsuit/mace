"""mace-torch builds graphs with vesin unless the config names a backend."""

from mace_core.neighbors import NeighborListConfig
from mace_core.neighbors.matscipy_backend import MatscipyBackend
from mace_core.neighbors.vesin_backend import VesinBackend
from mace_torch.neighbors import DEFAULT_NEIGHBOR_BACKEND, resolve_neighbor_backend


def test_mace_torch_defaults_to_vesin():
    assert DEFAULT_NEIGHBOR_BACKEND == "vesin"
    assert isinstance(resolve_neighbor_backend(), VesinBackend)


def test_a_backend_named_in_the_config_wins():
    config = NeighborListConfig(kind="matscipy")
    assert isinstance(resolve_neighbor_backend(config), MatscipyBackend)
