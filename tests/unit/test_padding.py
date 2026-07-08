"""Tests for graph padding utilities and padded calculator inference."""

import numpy as np
import pytest
import torch
from ase import build

from mace.data import AtomicData
from mace.data.padding_tools import build_fake_padding_graph
from mace.tools import torch_geometric, utils


@pytest.fixture(scope="module")
def water_graph():
    water = build.molecule("H2O")
    water.cell = [6.0] * 3
    water.pbc = True
    z_table = utils.AtomicNumberTable([1, 8])
    from mace import data as mace_data

    keyspec = mace_data.KeySpecification(info_keys={}, arrays_keys={})
    config = mace_data.config_from_atoms(water, key_specification=keyspec)
    graph = AtomicData.from_config(config, z_table=z_table, cutoff=3.5)
    return graph


class TestBuildFakePaddingGraph:
    def test_basic_shape(self, water_graph):
        pad_atoms = 10
        pad_edges = 32
        fake = build_fake_padding_graph(water_graph, pad_atoms, pad_edges, r_max=3.5)
        assert fake["node_attrs"].shape[0] == pad_atoms
        assert fake["edge_index"].shape[1] == pad_edges
        assert fake["positions"].shape == (pad_atoms, 3)

    def test_edge_index_within_bounds(self, water_graph):
        pad_atoms = 5
        pad_edges = 20
        fake = build_fake_padding_graph(water_graph, pad_atoms, pad_edges, r_max=3.5)
        assert fake["edge_index"].max() < pad_atoms
        assert fake["edge_index"].min() >= 0

    def test_zero_edges(self, water_graph):
        fake = build_fake_padding_graph(
            water_graph, num_atoms=3, num_edges=0, r_max=3.5
        )
        assert fake["edge_index"].shape[1] == 0
        assert fake["node_attrs"].shape[0] == 3

    def test_raises_on_zero_atoms(self, water_graph):
        with pytest.raises(ValueError, match="at least one fake atom"):
            build_fake_padding_graph(water_graph, num_atoms=0, num_edges=10, r_max=3.5)

    def test_cell_scale(self, water_graph):
        r_max = 5.0
        fake = build_fake_padding_graph(
            water_graph, num_atoms=2, num_edges=4, r_max=r_max
        )
        expected_scale = max(r_max * 2.0, 1.0)
        assert torch.allclose(
            fake["cell"],
            torch.eye(3, dtype=fake["cell"].dtype) * expected_scale,
        )

    def test_pbc_disabled(self, water_graph):
        fake = build_fake_padding_graph(
            water_graph, num_atoms=2, num_edges=4, r_max=3.5
        )
        assert not fake["pbc"].any()

    def test_batch_collation(self, water_graph):
        """Padding graph can be batched with the real graph."""
        fake = build_fake_padding_graph(
            water_graph, num_atoms=5, num_edges=16, r_max=3.5
        )
        batch = torch_geometric.Batch.from_data_list([water_graph, fake])
        total_atoms = water_graph["node_attrs"].shape[0] + 5
        total_edges = water_graph["edge_index"].shape[1] + 16
        assert batch["node_attrs"].shape[0] == total_atoms
        assert batch["edge_index"].shape[1] == total_edges
        assert batch["batch"].shape[0] == total_atoms

    def test_node_attrs_first_species_set(self, water_graph):
        fake = build_fake_padding_graph(
            water_graph, num_atoms=4, num_edges=8, r_max=3.5
        )
        assert (fake["node_attrs"][:, 0] == 1.0).all()
