import torch
import numpy as np
from torch.nn import Module
from math import sqrt



class ZonalFunctions(torch.nn.Module):
    def __init__(self, maxdim, normalize=False, basis='cartesian',
                 cg_dict=None, dtype=torch.float, device=torch.device('cpu')):
        super(ZonalFunctions, self).__init__()

        self.normalize = normalize
        self.basis = basis
        self.cg_dict = cg_dict
        self.maxdim = maxdim
        #super().__init__(cg_dict=self.cg_dict, maxdim=maxdim, device=device, dtype=dtype)

    def forward(self, pos, *ignore):
        return zonal_functions4(self.cg_dict, pos, self.maxdim, normalize=self.normalize)


class ZonalFunctionsRel(torch.nn.Module):
    def __init__(self, maxdim, normalize=False, basis='cartesian',
                 cg_dict=None, dtype=torch.float, device=torch.device('cpu')):

        super(ZonalFunctionsRel, self).__init__()

        self.normalize = normalize
        self.basis = basis

        super().__init__(cg_dict=cg_dict, maxdim=maxdim, device=device, dtype=dtype)

    def forward(self, pos1, pos2):
        return zonal_functions_rel(self.cg_dict, pos1, pos2, self.maxdim, normalize=self.normalize)


def zonal_functions(cg_dict, p, max_zf, normalize=False, basis='cartesian'):
    """Produce the list of zonal spherical functions evaluated at a given 4-vector p.
    Input p: a real (1,1) rep torch tensor (any number of batches/atoms/channels),
        presented as a dictionary with a single item.
        All momenta must have nonnegative norm, although we don't check for that.
    Output: a rep vector consisting of (l,l) irrep tensors, from l=1 to l=max_zf
     (same number of batches/atoms/channels)
     the l=1 tensor equals the input (in canonical basis).
     The higher l's are the values of the 'interesting' zonal functions."""

    if type(p) == dict:
        assert list(p.keys()) == [(1, 1)], 'p must contain only the (1,1) irrep!'
        p = p[(1, 1)]
    elif type(p) == torch.Tensor:
        assert p.shape[-1] == 4, 'p must be a tensor consisting of 4-vectors!'

    if basis == 'cartesian':
        p = p_to_rep(p)[(1, 1)]
    elif basis == 'canonical':
        pass
    else:
        raise ValueError('parameter "basis" must be either "cartesian" or "canonical"!')

    # Normalize the inputs to make the non-null 4-vectors have unit norm.
    # If the 4-vector is complex in Cartesian coordinates, this normalizes only the REAL part of the norm-squared
    if normalize:
        norm = torch.sqrt(torch.abs(normsq(p)[0]))
        mask = (norm != 0)
        p = torch.where(mask, p / norm, p)

    p = {(1, 1): p}
    zf = {(0, 0): torch.ones(p[(1, 1)].shape[:-1] + (1,), device = p[(1, 1)].device, dtype=p[(1, 1)].dtype)}
    zf.update(p)
    new_zf = zf

    # Iteratively construct zonal functions and store them as entires in the outpit dict
    for l in range(2, max_zf + 1):
        new_zf = cg_product(cg_dict, {(l - 1, l - 1): zf[(l - 1, l - 1)]}, p, maxdim=l + 1)[(l, l)]
        # This ensures that projecting onto the (l,l) component doesn't change the norm
        new_zf *= sqrt(2 * l / (l + 1))
        zf[(l, l)] = new_zf
    return GVec(zf)

def zonal_functions4(cg_dict, p, max_zf, normalize=False):
    """Produce the list of zonal spherical functions evaluated at a given 4-vector p.
    Input p: a real torch tensor of shape ((batch),4)
    Output: a rep vector consisting of (l,l) irrep tensors, from l=1 to l=max_zf
     (same number of batches/atoms/channels) The l=1 tensor equals the input (in canonical basis).
     The higher ones are the values of the zonal functions."""

    assert type(p) == torch.Tensor and p.shape[-1] == 4, 'p must be a tensor consisting of 4-vectors!'
    # Normalize the inputs to make the non-null 4-vectors have unit normself.
    # If the 4-vector is complex in Cartesian coordinates, this normalizes only the REAL part of the norm-squared
    norm_sq = normsq4(p).unsqueeze(-1)
    mask = (norm_sq != 0)
    norm = torch.where(mask, norm_sq / norm_sq.abs().sqrt(), norm_sq)
    if normalize:
        p = torch.where(mask, p / norm, p)

    p_rep = p_to_rep(p)
    zf = {(0, 0): torch.ones(p_rep[(1, 1)].shape[:-1] + (1,), device=p_rep.device, dtype=p_rep.dtype)}
    zf.update(p_rep)
    new_zf = zf

    # Iteratively construct zonal functions and store them as entires in the outpit dict
    for l in range(2, max_zf + 1):
        new_zf = cg_product(cg_dict, {(l - 1, l - 1): zf[(l - 1, l - 1)]}, p_rep, maxdim=l + 1)[(l, l)]
        # This ensures that projecting onto the (l,l) component doesn't change the norm
        new_zf *= sqrt(2 * l / (l + 1))
        zf[(l, l)] = new_zf
    return zf, norm.squeeze(-1), norm_sq.squeeze(-1)

def normsq4(p):
    # Quick hack to calculate the norms of the four-vectors
    # The last dimension of the input gets eaten up
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def zonal_functions_rel(cg_dict, p1, p2, maxdim, normalize=False):
    """Computes the zonal functions applied to all pairwise differences
     of 4-momenta in different channels (and broadcasts over higher batch dimensions).
     The output has TWO channel dimensions corresponding to the
     two channels whose difference was computed.
     Input: batch of REAL 4-vectors given as a tensor of shape ((batch),4) in Cartesuab coords
     Output: dictionary {(l,l):tensor} with l from 1 to maxdim-1"""

    # Pairwise differences of four-momenta
    rel_p = p1.unsqueeze(-2) - p2.unsqueeze(-3)

    zf_rel, rel_norms, rel_norms_sq = zonal_functions4(cg_dict, rel_p, maxdim, normalize=normalize)

    return zf_rel, rel_norms, rel_norms_sq

def p_to_rep(p):
    """Takes a batch of real 4-vectors (no complex dimension) as a tensor
    in Cartesian coordinates (t,x,y,z) and converts to the canonical basis, adding a channel dimension of size 1.
    Input: tensor of shape ((batch),4)
    Output: a complex (1,1) irrep {(1,1):tensor} with tensor of shape (2,(batch),1,4) """
    device = p.device
    dtype = p.dtype
    cartesian4 = torch.tensor([[[1, 0, 0, 0], [0, 1 / sqrt(2.), 0, 0], [0, 0, 0, 1], [0, -1 / sqrt(2.), 0, 0]],
                               [[0, 0, 0, 0], [0, 0, -1 / sqrt(2.), 0], [0, 0, 0, 0], [0, 0, -1 / sqrt(2.), 0]]], device=device, dtype=dtype)
    p = torch.unsqueeze(p, -1)
    rep = torch.stack([torch.matmul(cartesian4[0], p), torch.matmul(cartesian4[1], p)], 0)
    rep = {(1, 1): torch.squeeze(rep, -1).unsqueeze(-2)}
    return GVec(rep)

def p_cplx_to_rep(p):
    """Takes a batch of 4-vectors either as a tensor or a
    (1,1) irrep in Cartesian coordinates (t,x,y,z) and converts to the canonical basis.
    Output: a (1,1) irrep"""
    if type(p) == dict:
        p = p[(1, 1)]
    assert p.shape[0] == 2, "the first dimension of p must be the complex dimension of size 2"
    device = p.device
    cartesian4 = torch.tensor([[[1, 0, 0, 0], [0, 1 / sqrt(2.), 0, 0], [0, 0, 0, 1], [0, -1 / sqrt(2.), 0, 0]],
                               [[0, 0, 0, 0], [0, 0, -1 / sqrt(2.), 0], [0, 0, 0, 0], [0, 0, -1 / sqrt(2.), 0]]], device=device)
    p = torch.unsqueeze(p, -1)
    rep = torch.stack((torch.matmul(cartesian4[0], p[0]) - torch.matmul(cartesian4[1], p[1]),
                       torch.matmul(cartesian4[0], p[1]) + torch.matmul(cartesian4[1], p[0])), 0)
    rep = {(1, 1): torch.squeeze(rep, -1)}
    return GVec(rep)

def rep_to_p(rep):
    """Same unitary transformation in the opposite direction
    Output: torch tensor with a complex dimension at position 0"""
    if isinstance(rep, GTensor) or type(rep) == dict:
        rep = rep[(1, 1)]
    device = rep.device
    assert rep.shape[0] == 2, "the first dimension of rep must be the complex dimension"
    cartesian4H = torch.tensor([[[1, 0, 0, 0], [0, 1 / sqrt(2.), 0, 0], [0, 0, 0, 1], [0, -1 / sqrt(2.), 0, 0]],
                                [[0, 0, 0, 0], [0, 0, 1 / sqrt(2.), 0], [0, 0, 0, 0], [0, 0, 1 / sqrt(2.), 0]]], device=device).permute(0, 2, 1)
    rep = torch.unsqueeze(rep, -1)
    p = torch.stack((torch.matmul(cartesian4H[0], rep[0]) - torch.matmul(cartesian4H[1], rep[1]), torch.matmul(cartesian4H[0], rep[1]) + torch.matmul(cartesian4H[1], rep[0])), 0)
    return torch.squeeze(p, -1)

def normsq(p):
    """Computes the Lorentzian norm squared of reps presented in the canonical basis
    If the rep contains multiple irreps, it outputs a dictionary of tensors, otherwise a tensor."""
    if type(p) is dict or isinstance(p, GVec):
        pass
    else:
        p = {(1, 1): p}
    return repdot(p, p)

def metric(key):
    k, n = key
    met = [(1 if l == ll and m + mm == 0 else 0) * (-1)**(int(l + m))
           for l in np.arange(abs(k - n) / 2, (k + n) / 2 + 1, 1)
           for m in np.arange(-l, l + 1, 1)
           for ll in np.arange(abs(k - n) / 2, (k + n) / 2 + 1, 1)
           for mm in np.arange(-ll, ll + 1, 1)]
    return torch.tensor(met, dtype=torch.float).view([(k + 1) * (n + 1), (k + 1) * (n + 1)])

def repdot(rep1, rep2):
    """Lorentzian dot product of any two GVec's of the same representation type
    written in the canonical basis. This dot product is invariant under LorentzD matrices."""
    device = list(rep1.values())[0].device
    dtype = list(rep1.values())[0].dtype
    n = {}
    assert {key: part.shape for key, part in rep1.items()} == {key: part.shape for key, part in rep2.items()}, 'rep1 and rep2 must have all the same irreps of the same shapes!'
    for key in rep1.keys():
        met = metric(key).to(device=device, dtype=dtype)
        n.setdefault(key, torch.tensor([], device=device, dtype=dtype))
        n[key] = torch.stack((
            torch.einsum('...a,ab,...b->...', rep1[key][0], met, rep2[key][0]) - torch.einsum('...a,ab,...b->...', rep1[key][1], met, rep2[key][1]),
            torch.einsum('...a,ab,...b->...', rep1[key][0], met, rep2[key][1]) + torch.einsum('...a,ab,...b->...', rep1[key][1], met, rep2[key][0])), 0).unsqueeze(-1)
    #
    # if len(rep1.keys())==1:
    #     key=list(rep1.keys())[0]
    #     n=n[key]

    return n