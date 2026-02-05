import types
import torch
from e3nn import o3

import mace.modules.blocks as blocks
from mace.modules.blocks import (
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticResidualNonLinearInteractionBlock,
)


class DummyMP(torch.autograd.Function):
    calls = 0
    last_shape = None

    @staticmethod
    def forward(ctx, x, lammps_class):
        DummyMP.calls += 1
        DummyMP.last_shape = x.shape
        expected_total = lammps_class.expected_total
        assert x.shape[0] == expected_total
        n_real = lammps_class.n_real
        out = x.clone()
        if expected_total > n_real:
            out[n_real:] = 7.0
        return out


def _make_block_inputs(n_real, n_ghost, node_feat_dim, node_attr_dim, num_edges):
    node_feats = torch.randn(n_real, node_feat_dim)
    node_attrs = torch.randn(n_real, node_attr_dim)
    edge_attrs = torch.randn(num_edges, 1)
    edge_feats = torch.randn(num_edges, 1)
    senders = torch.tensor([0, n_real + 0, 1], dtype=torch.int64)
    receivers = torch.tensor([1, 2, 3], dtype=torch.int64)
    edge_index = torch.stack([senders, receivers], dim=0)
    return node_attrs, node_feats, edge_attrs, edge_feats, edge_index


def _make_irreps(node_feat_dim, node_attr_dim):
    node_feats_irreps = o3.Irreps(f"{node_feat_dim}x0e")
    node_attrs_irreps = o3.Irreps(f"{node_attr_dim}x0e")
    edge_attrs_irreps = o3.Irreps("1x0e")
    edge_feats_irreps = o3.Irreps("1x0e")
    target_irreps = o3.Irreps(f"{node_feat_dim}x0e")
    hidden_irreps = o3.Irreps(f"{node_feat_dim}x0e")
    return (
        node_attrs_irreps,
        node_feats_irreps,
        edge_attrs_irreps,
        edge_feats_irreps,
        target_irreps,
        hidden_irreps,
    )


def test_mliap_exchange_residual_nonlinear(monkeypatch):
    monkeypatch.setattr(blocks, "LAMMPS_MP", DummyMP)
    DummyMP.calls = 0
    DummyMP.last_shape = None

    n_real, n_ghost = 4, 2
    node_feat_dim, node_attr_dim = 2, 2
    node_attrs, node_feats, edge_attrs, edge_feats, edge_index = _make_block_inputs(
        n_real, n_ghost, node_feat_dim, node_attr_dim, num_edges=3
    )

    irreps = _make_irreps(node_feat_dim, node_attr_dim)
    block = RealAgnosticResidualNonLinearInteractionBlock(
        *irreps,
        avg_num_neighbors=1.0,
    )

    lammps_class = types.SimpleNamespace(n_real=n_real, expected_total=n_real + n_ghost)
    out, sc = block(
        node_attrs=node_attrs,
        node_feats=node_feats,
        edge_attrs=edge_attrs,
        edge_feats=edge_feats,
        edge_index=edge_index,
        lammps_class=lammps_class,
        lammps_natoms=(n_real, n_ghost),
        first_layer=False,
    )

    assert DummyMP.calls == 1
    assert DummyMP.last_shape == (n_real + n_ghost, node_feat_dim + node_attr_dim)
    assert out.shape[0] == n_real
    assert sc.shape[0] == n_real


def test_mliap_exchange_density_residual(monkeypatch):
    monkeypatch.setattr(blocks, "LAMMPS_MP", DummyMP)
    DummyMP.calls = 0
    DummyMP.last_shape = None

    n_real, n_ghost = 4, 2
    node_feat_dim, node_attr_dim = 2, 2
    node_attrs, node_feats, edge_attrs, edge_feats, edge_index = _make_block_inputs(
        n_real, n_ghost, node_feat_dim, node_attr_dim, num_edges=3
    )

    irreps = _make_irreps(node_feat_dim, node_attr_dim)
    block = RealAgnosticDensityResidualInteractionBlock(
        *irreps,
        avg_num_neighbors=1.0,
    )

    lammps_class = types.SimpleNamespace(n_real=n_real, expected_total=n_real + n_ghost)
    out, sc = block(
        node_attrs=node_attrs,
        node_feats=node_feats,
        edge_attrs=edge_attrs,
        edge_feats=edge_feats,
        edge_index=edge_index,
        lammps_class=lammps_class,
        lammps_natoms=(n_real, n_ghost),
        first_layer=False,
    )

    assert DummyMP.calls == 1
    assert DummyMP.last_shape == (n_real + n_ghost, node_feat_dim)
    assert out.shape[0] == n_real
    assert sc.shape[0] == n_real
