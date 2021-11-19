import torch
import numpy as np
from scipy.special import factorial
import itertools

import logging
logger = logging.getLogger(__name__)


#From Lorentz group equivariant network Bogatskiy

class CGDict():
    """
    A dictionary of Clebsch-Gordan (CG) coefficients to be used in CG operations.
    The CG coefficients
    .. math::
        \langle \ell_1, m_1, \ell_2, m_2 | l, m \rangle
    are used to decompose the tensor product of two
    irreps of maximum weights :math:`\ell_1` and :math:`\ell_2` into a direct
    sum of irreps with :math:`\ell = |\ell_1 -\ell_2|, \ldots, (\ell_1 + \ell_2)`.
    The coefficients for each :math:`\ell_1` and :math:`\ell_2`
    are stored as a :math:`D \times D` matrix :math:`C_{\ell_1,\ell_2}` ,
    where :math:`D = (2\ell_1+1)\times(2\ell_2+1)`.
    The module has a dict-like interface with keys :math:`(l_1, l_2)` for
    :math:`\ell_1, l_2 \leq l_{\rm max}`. Each value is a matrix of shape
    :math:`D \times D`, where :math:`D = (2l_1+1)\times(2l_2+1)`.
    The matrix has elements.
    Parameters
    ----------
    maxdim: int
        Maximum weight for which to calculate the Clebsch-Gordan coefficients.
        This refers to the maximum weight for the ``input tensors``, not the
        output tensors.
    transpose: bool, optional
        Transpose the CG coefficient matrix for each :math:`(\ell_1, \ell_2)`.
        This cannot be modified after instantiation.
    device: `torch.torch.device`, optional
        Device of CG dictionary.
    dtype: `torch.torch.dtype`, optional
        Data type of CG dictionary.
    """

    def __init__(self, maxdim=None, transpose=True, dtype=torch.float64, device=None):

        self.dtype = dtype
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self._transpose = transpose
        self._maxdim = None
        self._cg_dict = {}

        if maxdim is not None:
            self.update_maxdim(maxdim)

    @property
    def transpose(self):
        """
        Use "transposed" version of CG coefficients.
        """
        return self._transpose

    @property
    def maxdim(self):
        """
        Maximum weight for CG coefficients.
        """
        return self._maxdim

    def update_maxdim(self, new_maxdim):
        """
        Update maxdim to a new (possibly larger) value. If the new_maxdim is
        larger than the current maxdim, new CG coefficients should be calculated
        and the cg_dict will be updated.
        Otherwise, do nothing.
        Parameters
        ----------
        new_maxdim: int
            New maximum weight.
        Return
        ------
        self: `CGDict`
            Returns self with a possibly updated self.cg_dict.
        """
        # If self is already initialized, and maxdim is sufficiently large, do nothing
        if self and (self.maxdim >= new_maxdim):
            return self

        # If self is false, old_maxdim = 0 (uninitialized).
        # old_maxdim = self.maxdim if self else 0

        # Otherwise, update the CG coefficients.
        cg_dict_new = _gen_cg_dict(new_maxdim, existing_keys=self._cg_dict.keys())
        cg_dict_new = {key: {irrep: cg_tens.reshape(-1, cg_tens.shape[-1])
                             for irrep, cg_tens in val.items()}
                       for key, val in cg_dict_new.items()}
        if self.transpose:
            cg_dict_new = {key: {irrep: cg_mat.permute(1, 0)
                                 for irrep, cg_mat in val.items()}
                           for key, val in cg_dict_new.items()}

        # Ensure elements of new CG dict are on correct device.
        cg_dict_new = {key: {irrep: cg_mat.to(dtype=self.dtype, device=self.device)
                             for irrep, cg_mat in val.items()}
                       for key, val in cg_dict_new.items()}

        # Now update the CG dict, and also update maxdim
        self._cg_dict.update(cg_dict_new)

        self._maxdim = new_maxdim

        return self

    def to(self, dtype=None, device=None):
        """
        Convert CGDict() to a new device/dtype.
        Parameters
        ----------
        device : `torch.torch.device`, optional
            Device to move the cg_dict to.
        dtype : `torch.torch.dtype`, optional
            Data type to convert the cg_dict to.
        """
        if dtype is None and device is None:
            pass
        elif dtype is None and device is not None:
            self._cg_dict = {key: {irrep: cg_mat.to(
                device=device) for irrep, cg_mat in val.items()} for key, val in self._cg_dict.items()}
            self.device = device
        elif dtype is not None and device is None:
            self._cg_dict = {key: val.to(dtype=dtype) for key, val in self._cg_dict.items()}
            self.dtype = dtype
        elif dtype is not None and device is not None:
            self._cg_dict = {key: {irrep: cg_mat.to(device=device, dtype=dtype)
                                   for irrep, cg_mat in val.items()}
                             for key, val in self._cg_dict.items()}
            self.device, self.dtype = device, dtype
        return self

    def keys(self):
        return self._cg_dict.keys()

    def values(self):
        return self._cg_dict.values()

    def items(self):
        return self._cg_dict.items()

    def __getitem__(self, idx):
        if not self:
            raise ValueError('CGDict() not initialized. Either set maxdim, or use update_maxdim()')
        return self._cg_dict[idx]

    def __bool__(self):
        """
        Check to see if CGDict has been properly initialized, since :maxdim=-1: initially.
        """
        return self.maxdim is not None


def _gen_cg_dict(maxdim, existing_keys=None):
    '''
    Outputs a dictionary of tables of CG coefficients for the Lorentz group
    up to the given dimension maxdim of the G irrep subcomponents
    (every irrep of Lorentz group is V1 x V2, where V1 and V2 are irreps of G).
    Keys are tuples of labels (irrep T1, irrep T2) for irreps (which are themselves tuples of integers),
    and the values are again dictionaries of the form (irrep T): matrix,
    where the matrix is rectangular and maps irrep T into T1 x T2.
    This matrix is in fact more naturally stored as a torch.tensor of rank 3.
    Elements of an irrep of Lorentz group whose label is (k,n) are stored as vectors of size (k+1)*(n+1)
    which are concatenations of a set of vectors, exactly one for each l going from abs(k-n)/2 to (k+n)/2,
    and the size of the vector corresponding to l is 2*l+1. These sub-vectors belong to irreps of G.
    Therefore the dictionary values are tensors of shape ( (k1+1)*(n1+1), (k2+1)*(n2+1), (k+1)*(n+1) ).
    If we concatenate all such tensors for given (k1,n1,k2,n2), we get an orthogonal
    transformation from sum(T) to T1xT2, which is the CG operation done in cg_product().
    '''
    cg_dict = {}
    # print("gen_cg_dict called with maxdim =", maxdim)

    fastcgmat = memoize(clebschSU2mat)

    for k1, n1, k2, n2 in itertools.product(range(maxdim), repeat=4):
        if ((k1, n1), (k2, n2)) in existing_keys:
            continue
        cg_dict.setdefault(((k1, n1), (k2, n2)), {})
        kmin, kmax = abs(k1 - k2), k1 + k2
        nmin, nmax = abs(n1 - n2), n1 + n2
        # dim1, dim2 = (k1 + 1) * (n1 + 1), (k2 + 1) * (n2 + 1)
        for k, n in itertools.product(range(kmin, kmax + 1, 2), range(nmin, nmax + 1, 2)):
            cg_dict[((k1, n1), (k2, n2))][(k, n)] = torch.tensor(clebschmat((k1, n1), (k2, n2), (k, n), fastcgmat=fastcgmat))

    return cg_dict


def memoize(func):  # create a cached version of any function for fast repeated use
    cache = dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func


def clebschSU2mat(j1, j2, j3):

    mat = np.zeros((int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)))
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)] = clebschSU2((j1, m1), (j2, m2), (j3, m1 + m2))
    return np.array(mat)


def clebschmat(rep1, rep2, rep, fastcgmat=memoize(clebschSU2mat)):
    """
    Compute the whole rank 3 tensor of CG coefficients over (l1,m1),(l2,m2),(l,m) (implemented via fast matrix multiplication)
    """
    k1, n1 = rep1
    k2, n2 = rep2
    k, n = rep
    B1 = np.concatenate([fastcgmat(k / 2, n / 2, i / 2)
                         for i in range(abs(k - n), k + n + 1, 2)], axis=-1)
    B2a = fastcgmat(k1 / 2, k2 / 2, k / 2)
    B2b = fastcgmat(n1 / 2, n2 / 2, n / 2)
    B3a = np.concatenate([fastcgmat(k1 / 2, n1 / 2, i1 / 2)
                          for i1 in range(abs(k1 - n1), k1 + n1 + 1, 2)], axis=-1)
    B3b = np.concatenate([fastcgmat(k2 / 2, n2 / 2, i2 / 2)
                          for i2 in range(abs(k2 - n2), k2 + n2 + 1, 2)], axis=-1)
    H = np.einsum('cab', np.einsum('abc,dea,ghb,dgk,ehn', B1, B2a, B2b, B3a, B3b))
    return H


def clebsch(idx1, idx2, idx):
    """
    Calculate a single Clebsch-Gordan coefficient
    for SL(2,C) coupling (k1,n1,j1,m1) and (k2,n2,j2,m2) to give (k,n,j,m).
    We will never use this in the network.
    """
    fastcg = clebschSU2

    k1, n1, j1, m1 = idx1
    k2, n2, j2, m2 = idx2
    k, n, j, m = idx

    if int(2 * j1) not in range(abs(k1 - n1), k1 + n1 + 1, 2):
        print(idx1, idx2, idx)
        raise ValueError('Invalid value of l1')
    if int(2 * j2) not in range(abs(k2 - n2), k2 + n2 + 1, 2):
        print(idx1, idx2, idx)
        raise ValueError('Invalid value of l2')
    if int(2 * j) not in range(abs(k - n), k + n + 1, 2):
        print(idx1, idx2, idx)
        raise ValueError('Invalid value of l')
    if m != m1 + m2:
        return 0

    H = sum(fastcg((k / 2, mm1 + mm2), (n / 2, m - mm1 - mm2), (j, m)) *
            fastcg((k1 / 2, mm1), (k2 / 2, mm2), (k / 2, mm1 + mm2)) *
            fastcg((n1 / 2, m1 - mm1), (n2 / 2, m2 - mm2), (n / 2, m - mm1 - mm2)) *
            fastcg((k1 / 2, mm1), (n1 / 2, m1 - mm1), (j1, m1)) *
            fastcg((k2 / 2, mm2), (n2 / 2, m2 - mm2), (j2, m2))
            for mm1 in (x / 2 for x in set(range(-k1, k1 + 1, 2)).intersection(set(range(int(2 * m1 - n1), int(2 * m1 + n1 + 1), 2))))
            for mm2 in (x / 2 for x in set(range(-k2, k2 + 1, 2)).intersection(
                set(range(int(2 * m2 - n2), int(2 * m2 + n2 + 1), 2))).intersection(
                    set(range(int(2 * m - n - 2 * mm1), int(2 * m + n - 2 * mm1 + 1), 2))).intersection(
                        set(range(int(- k - 2 * mm1), int(k - 2 * mm1 + 1), 2))))
            )
    return H


# clebschSU2
# Taken from http://qutip.org/docs/3.1.0/modules/qutip/utilities.html

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

def clebschSU2(idx1, idx2, idx3):
    """Calculates the Clebsch-Gordon coefficient
    for SU(2) coupling (j1,m1) and (j2,m2) to give (j3,m3).
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    m1 : float
        z-component of angular momentum 1.
    m2 : float
        z-component of angular momentum 2.
    m3 : float
        z-component of angular momentum 3.
    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.
    """
    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3

    if m3 != m1 + m2:
        return 0
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    C = np.sqrt((2.0 * j3 + 1.0) * factorial(j3 + j1 - j2) * factorial(j3 - j1 + j2) * factorial(j1 + j2 - j3) * factorial(j3 + m3) * factorial(j3 - m3) /
                (factorial(j1 + j2 + j3 + 1) * factorial(j1 - m1) * factorial(j1 + m1) * factorial(j2 - m2) * factorial(j2 + m2)))
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) / factorial(v) * factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v) / \
            factorial(j3 - j1 + j2 - v) / \
            factorial(j3 + m3 - v) / \
            factorial(v + j1 - j2 - m3)
    C = C * S
    return C