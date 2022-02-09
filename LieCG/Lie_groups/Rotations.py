import torch
import numpy as np
from LieCG.CG_coefficients.CG_lorentz import CGDict



#Copied from the Lorentz Equivariant Network A. Bogatskiy, B. Anderson, J. T. Offermann, M. Roussi, D. W. Miller, R. Kondor, 
# Lorentz Group Equivariant Neural Network for Particle Physics, ICML 2020 (accepted).
# https://github.com/fizisist/LorentzGroupNetwork

def create_J(j):
    mrange = -torch.arange(-j, j)
    jp_diag = torch.sqrt((j + mrange) * (j - mrange + 1))
    Jp = torch.diag(jp_diag, diagonal=1)
    Jm = torch.diag(jp_diag, diagonal=-1)
    # Jx = (Jp + Jm) / complex(2, 0)
    # Jy = -(Jp - Jm) / complex(0, 2)
    Jz = torch.diag(-torch.arange(-j, j + 1))
    Id = torch.eye(2 * j + 1)
    return Jp, Jm, Jz, Id


def create_Jy(j):
    mrange = -torch.arange(-j, j)
    jp_diag = torch.sqrt((j + mrange) * (j - mrange + 1))
    Jp = torch.diag(jp_diag, diagonal=1)
    Jm = torch.diag(jp_diag, diagonal=-1)
    Jy = -(Jp - Jm) / complex(0, 2)
    return Jy


def create_Jx(j):
    mrange = -torch.arange(-j, j)
    jp_diag = torch.sqrt((j + mrange) * (j - mrange + 1))
    Jp = torch.diag(jp_diag, diagonal=1)
    Jm = torch.diag(jp_diag, diagonal=-1)
    Jx = (Jp + Jm) / complex(2, 0)
    return Jx


def littled(j, beta):
    Jy = create_Jy(j)
    evals, evecs = torch.linalg.eigh(Jy)
    evecsh = evecs.conj().T
    evals_exp = torch.diag(torch.exp(1j * beta * evals))
    d = torch.matmul(torch.matmul(evecs, evals_exp), evecsh)
    return d


def WignerD(j, alpha, beta, gamma, numpy_test=False, dtype=torch.float64, device=torch.device('cpu')):
    d = littled(j, beta)

    Jz = torch.arange(-j, j + 1)
    Jzl = torch.unsqueeze(Jz,1)

    # np.multiply() broadcasts, so this isn't actually matrix multiplication, and 'left'/'right' are lies
    left = torch.exp(1j * alpha * Jzl)
    right = torch.exp(1j * gamma * Jz)

    D = left * d * right


    return D


def LorentzD(key, alpha, beta, gamma, numpy_test=False, dtype=torch.float64, device=torch.device('cpu'), cg_dict=None):

    (k, n) = key
    if cg_dict is None:
        cg_dict = CGDict(maxdim=max(k, n) + 1, transpose=True, dtype=dtype, device=device)._cg_dict

    D = complex_tensor_prod(WignerD(k / 2, alpha, beta, gamma, numpy_test=numpy_test, dtype=dtype, device=device),
                            conj(WignerD(n / 2, -alpha, beta, -gamma, numpy_test=numpy_test, dtype=dtype, device=device)))
    cg_mat = cg_dict[((k, 0), (0, n))][(k, n)]
    D_re = torch.matmul(torch.matmul(cg_mat, D.unbind(0)[0]), cg_mat.t())
    D_im = torch.matmul(torch.matmul(cg_mat, D.unbind(0)[1]), cg_mat.t())
    D = torch.stack((D_re, D_im), 0)
    return D


def dagger(D):
    conj = torch.tensor([1, -1], dtype=D.dtype, device=D.device).view(2, 1, 1)
    D = (D * conj).permute((0, 2, 1))
    return D


def conj(D):
    conj = torch.tensor([1, -1], dtype=D.dtype, device=D.device).view(2, 1, 1)
    D = D * conj
    return D


def complex_from_numpy(z, dtype=torch.float64, device=torch.device('cpu')):
    """ Take a numpy array and output a complex array of the same size. """
    zr = torch.from_numpy(z.real).to(dtype=dtype, device=device)
    zi = torch.from_numpy(z.imag).to(dtype=dtype, device=device)

    return torch.stack((zr, zi), 0)


def complex_tensor_prod(d1, d2):
    d1_re, d1_im = d1.unbind(0)
    d2_re, d2_im = d2.unbind(0)
    s1 = d1.shape[1:]
    s2 = d2.shape[1:]
    assert len(s1) == 2 and len(s2) == 2, "Both tensors must be of rank 2 (and complex)!"
    d_re = d1_re.view(s1[0], 1, s1[1], 1) * d2_re.view(1, s2[0], 1, s2[1]) - \
        d1_im.view(s1[0], 1, s1[1], 1) * d2_im.view(1, s2[0], 1, s2[1])
    d_im = d1_re.view(s1[0], 1, s1[1], 1) * d2_im.view(1, s2[0], 1, s2[1]) + \
        d1_im.view(s1[0], 1, s1[1], 1) * d2_re.view(1, s2[0], 1, s2[1])
    return torch.stack((d_re, d_im), 0).view(2, s1[0] * s2[0], s1[1] * s2[1])


#From e3nn https://github.com/e3nn by Mario Geiger and al.

def matrix_x(angle: torch.Tensor) -> torch.Tensor:
    """matrix of rotation around X axis
    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack([
        torch.stack([o, z, z], dim=-1),
        torch.stack([z, c, -s], dim=-1),
        torch.stack([z, s, c], dim=-1),
    ], dim=-2)


def matrix_y(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around Y axis
    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack([
        torch.stack([c, z, s], dim=-1),
        torch.stack([z, o, z], dim=-1),
        torch.stack([-s, z, c], dim=-1),
    ], dim=-2)


def matrix_z(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around Z axis
    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack([
        torch.stack([c, -s, z], dim=-1),
        torch.stack([s, c, z], dim=-1),
        torch.stack([z, z, o], dim=-1)
    ], dim=-2)


def angles_to_matrix(alpha, beta, gamma):
    r"""conversion from angles to matrix
    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)

def matrix_to_angles(R):
    r"""conversion from matrix to angles
    Parameters
    ----------
    R : `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    assert torch.allclose(torch.det(R), R.new_tensor(1))
    x = R @ R.new_tensor([0.0, 1.0, 0.0])
    a, b = xyz_to_angles(x)
    R = angles_to_matrix(a, b, torch.zeros_like(a)).transpose(-1, -2) @ R
    c = torch.atan2(R[..., 0, 2], R[..., 0, 0])
    return a, b, c

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

def matrix_to_angles_SU2(R):
    alpha = 2 * torch.acos(torch.sqrt(R[0,0] * R[0,0].conj()))
    beta = torch.atan2(R[0,0].imag,R[0,0].real) + torch.atan2(R[0,1].imag,R[0,1].real) - torch.tensor(np.pi)/2
    gamma = torch.atan2(R[0,0].imag,R[0,0].real) - torch.atan2(R[0,1].imag,R[0,1].real) + torch.tensor(np.pi)/2


    return alpha,beta,gamma

class WignerD_SU2():
  def __init__(self,l):
    self.l = l
  def WignerD(self,input, numpy_test=False, dtype=torch.float64, device=torch.device('cpu')):
    angles = matrix_to_angles_SU2(input)
    d = littled(self.l, angles[1])

    Jz = torch.arange(-self.l, self.l + 1)
    Jzl = torch.unsqueeze(Jz,1)

    # np.multiply() broadcasts, so this isn't actually matrix multiplication, and 'left'/'right' are lies
    left = torch.exp(1j * angles[0] * Jzl)
    right = torch.exp(1j * angles[2] * Jz)

    D = left * d * right
    return D
  def matrix_to_angles_SU2(R):
 
            
    alpha = 2 * torch.acos(torch.sqrt(R[0,0] * R[0,0].conj()))
    beta = torch.atan2(R[0,0].imag,R[0,0].real) + torch.atan2(R[0,1].imag,R[0,1].real) - torch.tensor(np.pi)/2
    gamma = torch.atan2(R[0,0].imag,R[0,0].real) - torch.atan2(R[0,1].imag,R[0,1].real) + torch.tensor(np.pi)/2

    return alpha,beta,gamma 

class su2_algebra_irres():
    def __init__(self,j,group):
        self.j = j
        self.lie_algebra = group.lie_algebra

    def mul_plus(self,j,m):
        mul_p = torch.sqrt(torch.tensor(j*(j+1) - m*(m+1)))
        return mul_p

    def mul_min(self,j,m) : 
        mul_min = torch.sqrt(torch.tensor(j*(j+1) - m*(m-1)))
        return mul_min
    
    def drho(self,M):
        if torch.equal(M , self.lie_algebra[0]) : 
         return (1/(2*1j))*(self.drho_plus() + self.drho_min())
        if torch.equal(M , self.lie_algebra[1])  :
            return -(1/(2))*(self.drho_plus() - self.drho_min())
        if torch.equal(M , self.lie_algebra[2])  :
            return (1/1j)*self.drho_J3()
    def drho_plus(self) :
        drho = torch.zeros((int(2*self.j + 1),int(2*self.j + 1)),dtype=torch.cfloat)
        for i in range(int(2*self.j + 1)) :
            for k in range(int(2*self.j + 1)): 
                if i+1==k :
                    drho[i,k] = self.mul_plus(self.j,i- self.j) 
                else :
                    drho[0,0] = 0
        return drho
    def drho_min(self) :
        drho = torch.zeros((int(2*self.j + 1),int(2*self.j + 1)),dtype=torch.cfloat)
        for i in range(int(2*self.j + 1)) :
            for k in range(int(2*self.j + 1)) :
                if i-1==k :
                    drho[i,k] = self.mul_min(self.j,i - self.j) 
                else :
                    drho[0,0] = 0
        return drho
    def drho_J3(self) :
        drho = self.drho_min()@ self.drho_plus()  -  self.drho_plus() @ self.drho_min()
        return (-1/2)*drho








    


