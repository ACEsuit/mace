import pytest
import torch

from mace.modules.radial import AgnesiTransform, ZBLBasis


@pytest.fixture
def zbl_basis():
    return ZBLBasis(p=6, trainable=False)


def test_zbl_basis_initialization(zbl_basis):
    assert zbl_basis.p == torch.tensor(6.0)
    assert torch.allclose(zbl_basis.c, torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]))

    assert zbl_basis.a_exp == torch.tensor(0.300)
    assert zbl_basis.a_prefactor == torch.tensor(0.4543)
    assert not zbl_basis.a_exp.requires_grad
    assert not zbl_basis.a_prefactor.requires_grad


def test_trainable_zbl_basis_initialization(zbl_basis):
    zbl_basis = ZBLBasis(p=6, trainable=True)
    assert zbl_basis.p == torch.tensor(6.0)
    assert torch.allclose(zbl_basis.c, torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]))

    assert zbl_basis.a_exp == torch.tensor(0.300)
    assert zbl_basis.a_prefactor == torch.tensor(0.4543)
    assert zbl_basis.a_exp.requires_grad
    assert zbl_basis.a_prefactor.requires_grad


def test_forward(zbl_basis):
    x = torch.tensor([1.0, 1.0, 2.0]).unsqueeze(-1)  # [n_edges]
    node_attrs = torch.tensor(
        [[1, 0], [0, 1]]
    )  # [n_nodes, n_node_features] - one_hot encoding of atomic numbers
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 1]])  # [2, n_edges]
    atomic_numbers = torch.tensor([1, 6])  # [n_nodes]
    output = zbl_basis(x, node_attrs, edge_index, atomic_numbers)

    assert output.shape == torch.Size([node_attrs.shape[0]])
    assert torch.is_tensor(output)
    assert torch.allclose(
        output,
        torch.tensor([0.0031, 0.0031], dtype=torch.get_default_dtype()),
        rtol=1e-2,
    )


@pytest.fixture
def agnesi():
    return AgnesiTransform(trainable=False)


def test_agnesi_transform_initialization(agnesi: AgnesiTransform):
    assert agnesi.q.item() == pytest.approx(0.9183, rel=1e-4)
    assert agnesi.p.item() == pytest.approx(4.5791, rel=1e-4)
    assert agnesi.a.item() == pytest.approx(1.0805, rel=1e-4)
    assert not agnesi.a.requires_grad
    assert not agnesi.q.requires_grad
    assert not agnesi.p.requires_grad


def test_trainable_agnesi_transform_initialization():
    agnesi = AgnesiTransform(trainable=True)

    assert agnesi.q.item() == pytest.approx(0.9183, rel=1e-4)
    assert agnesi.p.item() == pytest.approx(4.5791, rel=1e-4)
    assert agnesi.a.item() == pytest.approx(1.0805, rel=1e-4)
    assert agnesi.a.requires_grad
    assert agnesi.q.requires_grad
    assert agnesi.p.requires_grad


def test_agnesi_transform_forward():
    agnesi = AgnesiTransform()
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.get_default_dtype()).unsqueeze(-1)
    node_attrs = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.get_default_dtype())
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    atomic_numbers = torch.tensor([1, 6, 8])
    output = agnesi(x, node_attrs, edge_index, atomic_numbers)
    assert output.shape == x.shape
    assert torch.is_tensor(output)
    assert torch.allclose(
        output,
        torch.tensor(
            [0.3646, 0.2175, 0.2089], dtype=torch.get_default_dtype()
        ).unsqueeze(-1),
        rtol=1e-2,
    )


if __name__ == "__main__":
    pytest.main([__file__])
