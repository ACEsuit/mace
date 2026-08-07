import types

import pytest
import torch
from e3nn import o3

from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE
from mace.modules import blocks
from mace.modules.blocks import (
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticResidualNonLinearInteractionBlock,
)


class DummyMP(
    torch.autograd.Function
):  # pylint: disable=abstract-method,arguments-differ
    calls = 0
    last_shape = None

    @staticmethod
    def forward(_ctx, x, lammps_class):
        DummyMP.calls += 1
        DummyMP.last_shape = x.shape
        expected_total = lammps_class.expected_total
        assert x.shape[0] == expected_total
        n_real = lammps_class.n_real
        out = x.clone()
        if expected_total > n_real:
            out[n_real:] = 7.0
        return out


def _make_block_inputs(n_real, _n_ghost, node_feat_dim, node_attr_dim, num_edges):
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


class _StubMACE(torch.nn.Module):
    """The metadata `MACEEdgeForcesWrapper` reads off a real MACE model."""

    def __init__(self, num_interactions):
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor([1, 8]))
        self.register_buffer("r_max", torch.tensor(3.5))
        self.register_buffer("num_interactions", torch.tensor(num_interactions))
        self.lin = torch.nn.Linear(1, 1)


def _mliap_data(**attrs):
    """A stand-in for LAMMPS's MLIAPDataPy, with only the attributes named."""
    return types.SimpleNamespace(elems=torch.zeros(3, dtype=torch.int64), **attrs)


@pytest.mark.parametrize("num_interactions", [2, 3])
def test_multilayer_without_forward_exchange_is_actionable(num_interactions):
    # conda-forge's CPU LAMMPS builds with PKG_KOKKOS=OFF, so its MLIAPDataPy
    # has no forward_exchange and a >1-layer model used to die on a bare
    # AttributeError three frames deep in the second interaction block.
    unified = LAMMPS_MLIAP_MACE(_StubMACE(num_interactions))
    data = _mliap_data()

    with pytest.raises(RuntimeError, match="PKG_KOKKOS"):
        unified._check_ghost_exchange_support(data)  # pylint: disable=protected-access


def test_single_layer_needs_no_forward_exchange():
    # One layer never leaves the local atoms, which is what makes a stock
    # non-KOKKOS build usable at all -- and what the real tier relies on.
    unified = LAMMPS_MLIAP_MACE(_StubMACE(1))
    unified._check_ghost_exchange_support(_mliap_data())  # pylint: disable=protected-access


def test_multilayer_with_forward_exchange_is_accepted():
    unified = LAMMPS_MLIAP_MACE(_StubMACE(2))
    data = _mliap_data(forward_exchange=lambda *_: None)
    unified._check_ghost_exchange_support(data)  # pylint: disable=protected-access
