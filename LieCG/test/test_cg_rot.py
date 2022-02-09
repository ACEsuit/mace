from random import randrange

import numpy as np

import torch



from sympy.physics.quantum import spin
from sympy import Ynm
from LieACE.modules.spherical_harmonics import SphericalHarmonics

from LieCG.CG_coefficients.CG_rot import ClebschGordan, Rot3DCoeffs, clebschgordan, re_basis


def test_cg():
    """
    checking the CG coefficients
    """
    cg_eval = ClebschGordan()
    ntest = 0
    while ntest < 200:
        j1 = randrange(11)
        j2 = randrange(11)
        J = randrange(abs(j1 - j2), min([11, j1 + j2 + 1]))
        m1 = randrange(-1 * j1, j1 + 1)
        for m2 in range(-1 * j2, j2 + 2):
            M = m1 + m2
            if abs(M) <= J:
                ntest += 1
                inds = (j1, m1, j2, m2, J, M)
                assert np.isclose(cg_eval(inds), spin.CG(j1, m1, j2, m2, J, M).doit().evalf().__float__(), atol=1e-7)


def xyz_to_angles(xyz):
    r"""convert a point :math:`\vec r = (x, y, z)` on the sphere into angles :math:`(\alpha, \beta)`
    .. math::
        \vec r = R(\alpha, \beta, 0) \vec e_z
    Parameters
    ----------
    xyz : `torch.Tensor`
        tensor of shape :math:`(..., 3)`
    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    xyz = torch.nn.functional.normalize(xyz, p=2, dim=-1)  # forward 0's instead of nan for zero-radius
    xyz = xyz.clamp(-1, 1)

    beta = torch.acos(xyz[..., 1])
    alpha = torch.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta

def test_shp_sympy(lmax = 20):
    Ylm_func = SphericalHarmonics(lmax=lmax)
    p_r = 0.0
    p_i = 0.0
    p2_r = 0.0
    p2_i = 0.0
    for ntest in range(200):
        xyz = torch.randn(3)
        xyz = xyz / torch.linalg.norm(xyz)
        theta = xyz_to_angles(torch.tensor((xyz[1],xyz[2],xyz[0])))[0].item()
        phi = xyz_to_angles(torch.tensor((xyz[1],xyz[2],xyz[0])))[1].item()
        for l in range(lmax):
            for m in range(-l,l+1):
                p_r += Ylm_func.compute_ylm(l,m,xyz)[0]
                p_i += Ylm_func.compute_ylm(l,m,xyz)[1]
                p2_r += complex(Ynm(l,m,phi,theta)).real
                p2_i += complex(Ynm(l,m,phi,theta)).imag
        p = p_r + p_i
        p2 = p2_r + p2_i
        assert np.isclose(p, p2, atol=1e-7)


def test_sph_exp():
    """
    checking the Spherical Harmonic expansion of products of Ylm-s
    in terms of a single Ylm
    """
    Ylm_func = SphericalHarmonics(lmax=20)
    for ntest in range(200):
        # two random Ylm-s
        l1 = randrange(1, 11)
        l2 = randrange(1, 11)
        m1 = randrange(-1 * l1, l1 + 1)
        m2 = randrange(-1 * l2, l2 + 1)
        theta = np.random.uniform() * np.pi
        phi = np.random.uniform(-0.5, 0.5) * 2 * np.pi
        R = torch.tensor([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
        p = Ylm_func.compute_ylm(l1, m1, R)[0] * Ylm_func.compute_ylm(l2, m2, R)[0] - \
            Ylm_func.compute_ylm(l1, m1, R)[1] * Ylm_func.compute_ylm(l2, m2, R)[1]
        p2 = 0.0
        M = m1 + m2
        for L in range(abs(M), l1 + l2 + 1):
            p2 += np.sqrt( (2*l1 + 1)*(2*l2+1) / (4*np.pi*(2*L+1))) * \
            clebschgordan(l1, 0, l2, 0, L, 0) * \
            clebschgordan(l1, m1, l2, m2, L, M) * Ylm_func.compute_ylm(L, M, R)[0]
        assert np.isclose(p, p2, atol=1e-7)


def test_Rot3Dcoeffs():
    A = Rot3DCoeffs(6)
    ll = torch.tensor([0, 2, 1])
    mm = torch.tensor([0, -1, 1])
    kk = torch.tensor([0, 1, -1])
    assert A(ll, mm, kk) == 0
    ll = torch.tensor([0, 1, 3, 2])
    mm = torch.tensor([0, 1, 1, -2])
    kk = torch.tensor([0, 1, -2, 1])
    assert np.isclose(A(ll, mm, kk), -0.030116930096841705, atol=1e-7)  # number from ACE.jl


def test_re_basis():
    A = Rot3DCoeffs(6)
    ll = torch.tensor([0, 2, 2])
    Ure, Mll = re_basis(A, ll)
    # check against the Julia implementation
    assert Ure.size() == torch.Size([1, 5])
    assert all(
        np.isclose(Ure.numpy()[0],
                   np.array([
                       -0.4472135954999578, 0.4472135954999578, -0.4472135954999578, 0.4472135954999578,
                       -0.4472135954999578
                   ]),
                   atol=1e-7))