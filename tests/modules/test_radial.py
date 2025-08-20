import pytest
import torch
import numpy as np

from mace.modules.radial import (
    AgnesiTransform,
    ZBLBasis,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    SoftTransform,
)


@pytest.fixture
def zbl_basis():
    return ZBLBasis(p=6, trainable=False).to(torch.float32)


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


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_zbl_basis_forward(zbl_basis, dtype):
    zbl_basis = zbl_basis.to(dtype)
    x = torch.tensor([1.0, 1.0, 2.0], dtype=dtype).unsqueeze(-1)
    node_attrs = torch.tensor([[1, 0], [0, 1]], dtype=dtype)
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 1]])
    atomic_numbers = torch.tensor([1, 6])
    output = zbl_basis(x, node_attrs, edge_index, atomic_numbers)
    assert output.shape == torch.Size([node_attrs.shape[0]])
    assert output.dtype == dtype
    assert torch.is_tensor(output)
    assert torch.allclose(
        output,
        torch.tensor([0.0031, 0.0031], dtype=dtype),
        rtol=1e-2,
    )


@pytest.fixture
def agnesi():
    return AgnesiTransform(trainable=False).to(torch.float32)


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


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_agnesi_transform_forward(agnesi, dtype):
    agnesi = agnesi.to(dtype)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype).unsqueeze(-1)
    node_attrs = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=dtype)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    atomic_numbers = torch.tensor([1, 6, 8])
    output = agnesi(x, node_attrs, edge_index, atomic_numbers)
    assert output.shape == x.shape
    assert output.dtype == dtype
    assert torch.is_tensor(output)
    assert torch.allclose(
        output,
        torch.tensor([0.3646, 0.2175, 0.2089], dtype=dtype).unsqueeze(-1),
        rtol=1e-2,
    )


@pytest.fixture
def bessel_basis():
    return BesselBasis(r_max=5.0, num_basis=8, trainable=False).to(torch.float32)


def test_bessel_basis_initialization(bessel_basis):
    assert bessel_basis.r_max == torch.tensor(5.0)
    assert len(bessel_basis.bessel_weights) == 8
    assert bessel_basis.prefactor.item() == pytest.approx(np.sqrt(2.0 / 5.0), rel=1e-4)
    assert not bessel_basis.bessel_weights.requires_grad


def test_trainable_bessel_basis_initialization():
    basis = BesselBasis(r_max=5.0, num_basis=8, trainable=True)
    assert basis.bessel_weights.requires_grad


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_bessel_basis_forward(bessel_basis, dtype):
    bessel_basis = bessel_basis.to(dtype)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype).unsqueeze(-1)
    output = bessel_basis(x)
    assert output.shape == (3, 8)
    assert output.dtype == dtype
    assert torch.is_tensor(output)


@pytest.fixture
def chebychev_basis():
    return ChebychevBasis(r_max=5.0, num_basis=8).to(torch.float32)


def test_chebychev_basis_initialization(chebychev_basis):
    assert chebychev_basis.r_max == 5.0
    assert chebychev_basis.num_basis == 8
    assert chebychev_basis.n.shape == (1, 8)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_chebychev_basis_forward(chebychev_basis, dtype):
    chebychev_basis = chebychev_basis.to(dtype)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype).unsqueeze(-1)
    output = chebychev_basis(x)
    assert output.shape == (3, 8)
    assert output.dtype == dtype
    assert torch.is_tensor(output)


@pytest.fixture
def gaussian_basis():
    return GaussianBasis(r_max=5.0, num_basis=8, trainable=False).to(torch.float32)


def test_gaussian_basis_initialization(gaussian_basis):
    assert len(gaussian_basis.gaussian_weights) == 8
    assert not gaussian_basis.gaussian_weights.requires_grad


def test_trainable_gaussian_basis_initialization():
    basis = GaussianBasis(r_max=5.0, num_basis=8, trainable=True)
    assert basis.gaussian_weights.requires_grad


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_gaussian_basis_forward(gaussian_basis, dtype):
    gaussian_basis = gaussian_basis.to(dtype)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype).unsqueeze(-1)
    output = gaussian_basis(x)
    assert output.shape == (3, 8)
    assert output.dtype == dtype
    assert torch.is_tensor(output)


@pytest.fixture
def polynomial_cutoff():
    return PolynomialCutoff(r_max=5.0, p=6).to(torch.float32)


def test_polynomial_cutoff_initialization(polynomial_cutoff):
    assert polynomial_cutoff.r_max == torch.tensor(5.0)
    assert polynomial_cutoff.p == torch.tensor(6)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_polynomial_cutoff_forward(polynomial_cutoff, dtype):
    polynomial_cutoff = polynomial_cutoff.to(dtype)
    x = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=dtype)
    output = polynomial_cutoff(x)
    assert output.shape == (4,)
    assert output.dtype == dtype
    assert torch.is_tensor(output)
    assert torch.allclose(output[3], torch.tensor(0.0, dtype=dtype))


@pytest.fixture
def soft_transform():
    return SoftTransform(alpha=4.0, trainable=False).to(torch.float32)


def test_soft_transform_initialization(soft_transform):
    assert soft_transform.alpha == torch.tensor(4.0)
    assert not soft_transform.alpha.requires_grad


def test_trainable_soft_transform_initialization():
    transform = SoftTransform(alpha=4.0, trainable=True)
    assert transform.alpha.requires_grad


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_soft_transform_forward(soft_transform, dtype):
    soft_transform = soft_transform.to(dtype)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype).unsqueeze(-1)
    node_attrs = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=dtype)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    atomic_numbers = torch.tensor([1, 6, 8])
    output = soft_transform(x, node_attrs, edge_index, atomic_numbers)
    assert output.shape == x.shape
    assert output.dtype == dtype
    assert torch.is_tensor(output)
