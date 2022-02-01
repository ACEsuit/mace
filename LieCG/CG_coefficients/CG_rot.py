import math
import itertools
import torch
import numpy as np
from sympy.physics.wigner import clebsch_gordan
from sympy import S


def coco_filter(ll, mm, invariant=True):
    # for 2 invariant coefficients
    if invariant:
        return sum(ll) % 2 == 0 and sum(mm) == 0


def coco_filter2(ll, mm, kk, invariant=True):
    # for 3 invariant coefficients
    if invariant:
        return sum(ll) % 2 == 0 and sum(mm) == 0 \
            and sum(kk) == 0


# versions operating on numbers


def cg_l_condition(j1, j2, J):
    return abs(j1 - j2) <= J and abs(j1 + j2) >= J


def cg_m_condition(m1, m2, M):
    return M == m1 + m2


def cg_conditions(j1, m1, j2, m2, J, M):
    return cg_l_condition(j1, j2, J) and cg_m_condition(m1, m2, M) and \
            abs(m1) <= j1 and abs(m2) <= j2 and abs(M) <= J


def clebschgordan(j1, m1, j2, m2, J, M):
    """
    Implementatino of Clebsch-Gordan coefficients based on
    https://hal.inria.fr/hal-01851097/document  Equation (4-6)
    and ACE.jl
    The ordering of parameters corresponds to the following convention:
    ```
    clebschgordan(j1, m1, j2, m2, J, M) = C_{j1m1j2m2}^{JM}
    ```
    where
    ```
    D_{m1k1}^{l1} D_{m2k2}^{l2}}
        =
        ∑_j  C_{l1m1l2m2}^{j(m1+m2)} C_{l1k1l2k2}^{j2(k1+k2)} D_{(m1+m2)(k1+k2)}^{j}
    ```
    """
    if not cg_conditions(j1, m1, j2, m2, J, M):
        return 0
    else:
        return float(clebsch_gordan(S(j1), S(j2), S(J), S(m1), S(m2), S(M)))

def clebschgordan_from_scratch(j1, m1, j2, m2, J, M):
    """
    Implementation wihtout any dependencies
    Implementatino of Clebsch-Gordan coefficients based on
    https://hal.inria.fr/hal-01851097/document  Equation (4-6)
    and ACE.jl
    The ordering of parameters corresponds to the following convention:
    ```
    clebschgordan(j1, m1, j2, m2, J, M) = C_{j1m1j2m2}^{JM}
    ```
    where
    ```
    D_{m1k1}^{l1} D_{m2k2}^{l2}}
        =
        ∑_j  C_{l1m1l2m2}^{j(m1+m2)} C_{l1k1l2k2}^{j2(k1+k2)} D_{(m1+m2)(k1+k2)}^{j}
    ```
    """
    if not cg_conditions(j1, m1, j2, m2, J, M):
        return 0

    N = (2 * J + 1) * math.factorial(j1 + m1) * math.factorial(j1 - m1) * \
        math.factorial(j2 + m2) * math.factorial(j2 - m2) * math.factorial(J + M) * \
        math.factorial(J - M) / math.factorial(j1 + j2 - J) / math.factorial(j1 - j2 + J) / \
        math.factorial(-1*j1 + j2 + J) / math.factorial(j1 + j2 + J + 1)

    G = 0
    # 0 ≦ k ≦ j1+j2-J
    # 0 ≤ j1-m1-k ≤ j1-j2+J   <=>   j2-J-m1 ≤ k ≤ j1-m1
    # 0 ≤ j2+m2-k ≤ -j1+j2+J  <=>   j1-J+m2 ≤ k ≤ j2+m2
    lb = [0, j2 - J - m1, j1 - J + m2]
    ub = [j1 + j2 - J, j1 - m1, j2 + m2]
    for k in range(max(lb), min(ub) + 1):
        G += (-1)**k * math.comb(j1 + j2 - J, k) * \
            math.comb(j1 - j2 + J, j1 - m1 - k) * \
            math.comb(-1*j1 + j2 + J, j2 + m2 - k)
    return math.sqrt(N) * G


class ClebschGordan:
    """
    Class to store precomputed Clebsch-Gordan coefficients
    """
    def __init__(self):
        self.cg_val = {}

    def __call__(self, inds: tuple):
        """
        :param inds: Tuple containing indicies of CG coefficients in the order
                     (j1, m1, j2, m2, J, M)
        :return: float CG coefficient
        """
        assert len(inds) == 6, "CG coefficient should have 6 indicies"
        if not cg_conditions(inds[0], inds[1], inds[2], inds[3], inds[4], inds[5]):
            self.cg_val[inds] = 0.0
            return self.cg_val[inds]
        elif inds in self.cg_val.keys():
            return self.cg_val[inds]
        else:
            self.cg_val[inds] = clebschgordan(inds[0], inds[1], inds[2], inds[3], inds[4], inds[5])
            return self.cg_val[inds]


class Rot3DCoeffs:
    """
    Class to store the computed rotationally averaged coefficints of spherical harmonics
    """
    def __init__(self, max_N=20):
        self.vals = [dict() for i in range(max_N)]
        self.cg = ClebschGordan()

    def get_vals(self, N):
        return self.vals[N]

    @staticmethod
    def _key(ll, mm, kk):
        return (ll, mm, kk)

    def __call__(self, ll, mm, kk):
        """
        ll, mm, kk are vectors of length N for corr. order N
        """
        N = len(ll)
        vals = self.get_vals(N)
        key = self._key(ll, mm, kk)
        if key in vals.keys():
            val = vals[key]
        else:
            val = self._compute_val(ll, mm, kk)  # implement below
            self.vals[N][key] = val
        return val

    @staticmethod
    def coco_init(l, m, k):
        return 1.0 if (l == m == k == 0) else 0.0

    def _compute_val(self, ll, mm, kk):
        val = 0.0
        N = len(ll)
        jmin = max([abs(ll[N - 2] - ll[N - 1]), abs(kk[N - 2] + kk[N - 1]), abs(mm[N - 2] + mm[N - 1])])
        jmax = ll[N - 2] + ll[N - 1]
        for j in range(jmin, jmax + 1):
            cgk = self.cg((ll[N - 2], kk[N - 2], ll[N - 1], kk[N - 1], j, kk[N - 2] + kk[N - 1]))
            cgm = self.cg((ll[N - 2], mm[N - 2], ll[N - 1], mm[N - 1], j, mm[N - 2] + mm[N - 1]))
            if cgk * cgm != 0:
                llpp = torch.cat((ll[:-2], torch.tensor([j])))  # (llp..., j)
                mmpp = torch.cat(
                    (torch.tensor(mm[:-2]), torch.tensor([mm[N - 2] + mm[N - 1]])))  # (mmp..., mm[N-1]+mm[N])
                kkpp = torch.cat(
                    (torch.tensor(kk[:-2]), torch.tensor([kk[N - 2] + kk[N - 1]])))  # (kkp..., kk[N-1]+kk[N])
                if len(llpp) == 1:
                    a = self.coco_init(llpp[0], mmpp[0], kkpp[0])
                else:
                    a = self.__call__(llpp, mmpp, kkpp)  # this is okay because it is
                    # a recursion, so they should be in self.vals
                val += cgk * cgm * a
        return val


# Construction of CG coefficients numerically via SVD

def re_basis(A, ll):
    CC, Mll = compute_Al(A, ll)
    CC = CC.numpy()
    # CC = torch.squeeze(CC).numpy()
    G = np.matmul(CC, CC.T)  # I think it works for invariants, if not below with for loops
    # G = torch.zeros((len(CC), len(CC)))
    # for i in range(len(CC)):
    #     for j in range(len(CC)):
    #         for k in range(len(Mll)):
    #             G[i,j] += torch.dot(CC[i, k], CC[j, k])
    svdC = np.linalg.svd(G)  # , hermitian = True)
    rk = np.linalg.matrix_rank(np.diag(svdC[1]), tol=1e-7)
    # construct the new basis
    # This is obviously not the way and should be fixed ASAP
    # it deals with the rank=1 one case where it is just a multiplication
    try:
        Ured = np.diag(np.sqrt(svdC[1][0:rk])) * np.conjugate(svdC[0][:, 0:rk]).T
    except:
        Ured = np.matmul(np.diag(np.sqrt(svdC[1][0:rk])), np.conjugate(svdC[0][:, 0:rk]).T)
    # Ured = torch.mm(torch.diag(torch.sqrt(svdC.S[0:rk])), torch.transpose(svdC.U[:, 0:rk]))  # Didn't take complex conjugate here,
    # Would be necessary if we were to do complex
    Ure = np.zeros((rk, len(Mll)))  # First dimension in Juila code undefined not sure why, I think it is rk
    for i in range(rk):
        tmp = np.zeros_like(Ure[i, :])
        for j in range(len(CC)):
            tmp += Ured[i, j] * CC[j]
        Ure[i, :] = tmp
    Ure = torch.tensor(Ure)
    return Ure, Mll


# Construction of the sparse U tensor for a given correlation order (invariant)
def l_filter(ll):
    return sum(ll) % 2 == 0


def Mll_to_inds(ns, ll, Mll, lmax):
    nu = len(ll)
    inds = []
    for mll in Mll:
        ind = []
        for j in range(nu):
            ind.append(ns[j] * (lmax + 1)**2 + ll[j]*(ll[j] + 1) + mll[j])
        inds.append(ind)
    return inds


def create_U(A, nu: int, degree_func):
    """
    Function to create the sparse U matrix for a given correlation order
    :param A: Rot3DCoeffs
    :param nu: int, correlation order
    :param degree_func: Degree function, either SparseDeg or NaiveMaxDeg
    :return U: torch.sparse_coo_tensor shape (len(B basis), ((nmax+1)*(lmax+1)**2)**nu),
            index_mm: indices of the mm basis that share weight.
    """

    # first obtain nmax and lmax determining the shape of U
    nmax = degree_func.max_n()
    lmax = degree_func.max_l()
    # Now generate all possible combinations of l
    lls = itertools.combinations_with_replacement(range(lmax+1), nu)
    # generate all possibel combinations of ns
    ns_unique = list(itertools.combinations_with_replacement(range(nmax + 1), nu))
    all_ns = []
    for n in ns_unique:
        all_ns += list(itertools.permutations(n))
    all_ns = sorted(set(all_ns))  # remove the duplicates
    inds = []
    coeffs = torch.tensor([])
    index_mm = []  # indices of mm basis that should share weights
    i_mm = 0
    for ll in lls:  # iterate over allowed l-tuples
        if l_filter(ll):  # check sum(ls)=even
            all_ll = set(itertools.permutations(ll))  # #generate all permutations of allowed l-s
            for ls in all_ll:
                print(ls)
                Ure, Mll = re_basis(A, torch.tensor(ls))  # compute coupling coefficients and corresponding m-s
                for ns in all_ns:  # iterate over all n-tuples
                    if degree_func(ns, ls, nu): # check if the combination of n-s and l-s is allowed
                        if Ure.numpy().size != 0:
                            for u in Ure:
                                ind = Mll_to_inds(ns, ls, Mll, lmax)  # convert them to sparse tensor inidcies
                                inds += ind
                                coeffs = torch.cat((coeffs, u))
                                index_mm += [i_mm]*len(torch.flatten(u))  # will return the same index for elements that should share mm
                                i_mm += 1
    index_mm = torch.tensor(index_mm)
    inds = torch.cat((index_mm[None, :], torch.transpose(torch.tensor(inds), 0, 1)))
    size = (len(index_mm.unique()),) + tuple(((nmax + 1) * (lmax + 1)**2 for i in range(nu)))
    return torch.sparse_coo_tensor(inds, coeffs, size=size).to_dense().moveaxis(0,-1)


# -----------------------------------
# iterating over an m collection
# -----------------------------------


class MRange:
    def __init__(self, ll, cartrg):
        self.ll = ll
        self.cartrg = cartrg
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.idx >= len(self.cartrg):
                raise StopIteration
            mm = self.cartrg[self.idx]
            if coco_filter(self.ll, mm):  # assuming invariant here
                self.idx += 1
                return mm
            self.idx += 1
        print("Code should never get here!")

    def __len__(self):
        l = 0
        for mt in self.cartrg:
            if coco_filter(self.ll, mt):  # Assuming invariant
                l += 1
        return l


def compute_Al(A, ll):
    ms = tuple([tuple([m for m in range(-1 * ll[i], ll[i] + 1)]) for i in range(len(ll))])
    mvecs = list(itertools.product(*ms))
    Mll = list(MRange(ll, mvecs))  # m-s corresponding to l with symmetry obeyed
    if len(Mll) == 0:
        return torch.tensor([]), Mll
    return __compute_Al(A, ll, Mll)


def __compute_Al(A, ll, Mll):
    # each element of CC will be one row of the coupling coefficients
    # CC = torch.tensor([])
    for i_Mll, kk in enumerate(Mll):  # loop over possible basis functions
        # NEEDS REVISION FOR EUCLIDEAN VECTORS
        # do a dummy calculation to determine how many coefficients we will get
        # cc0 = A(ll, Mll[0], kk)
        # numcc = 0
        # for cc in cc0:
        #     if cc0 is not float('nan'):
        #         numcc += 1
        # cc = torch.zeros((numcc, len(Mll))) # NEED THIS FOR EQUIVARIANT
        cc = torch.zeros((1, len(Mll)))  # NEED THIS FOR EQUIVARIANT
        for (im, mm) in enumerate(Mll):  # loop over possible indices
            if not coco_filter2(ll, mm, kk):
                cc[0][im] = 0.0
                # cc00 = torch.zeros(len(cc))
                # for p in range(numcc):
                #    cc[p][im] = cc00[p]
            else:
                # get all possible coupling coefficients
                cc0 = A(ll, mm, kk)
                cc[0][im] = cc0
                # for p in range(numcc):
                #     cc[p][im] = cc0
        # and now push them onto the big stack.
        if i_Mll == 0:
            CC = cc
        else:
            CC = torch.cat((CC, cc))
    return CC, Mll


##########################################################################################
### torch tensor version of CG coefficients, probably not necessary
def torch_cg_l_condition(j1, j2, J):
    return torch.abs(j1 - j2) <= J and torch.abs(j1 + j2) >= J


def torch_cg_m_condition(m1, m2, M):
    return M == m1 + m2


def torch_cg_conditions(j1, m1, j2, m2, J, M):
    return torch_cg_l_condition(j1, j2, J) and torch_cg_m_condition(m1, m2, M) and \
            torch.abs(m1) <= j1 and torch.abs(m2) <= j2 and torch.abs(M) <= J


def torch_binom(n, k):
    # WE MIGHT NEED TO WORRY ABOUT GRADIENTS HERE
    # mask = n.detach() >= k.detach()
    # n = mask * n
    # k = mask * k
    a = torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
    return torch.exp(a)  # * mask


def torch_factorial(n):
    return torch.exp(torch.lgamma(n + 1))


def torch_clebschgordan(j1, m1, j2, m2, J, M):
    """
    Implementatino of Clebsch-Gordan coefficients based on
    https://hal.inria.fr/hal-01851097/document  Equation (4-6)
    and ACE.jl
    The ordering of parameters corresponds to the following convention:
    ```
    clebschgordan(j1, m1, j2, m2, J, M) = C_{j1m1j2m2}^{JM}
    ```
    where
    ```
    D_{m1k1}^{l1} D_{m2k2}^{l2}}
        =
        ∑_j  C_{l1m1l2m2}^{j(m1+m2)} C_{l1k1l2k2}^{j2(k1+k2)} D_{(m1+m2)(k1+k2)}^{j}
    ```
    """
    if not torch_cg_conditions(j1, m1, j2, m2, J, M):
        return 0

    N = (2 * J + 1) * torch_factorial(j1 + m1) * torch_factorial(j1 - m1) * \
        torch_factorial(j2 + m2) * torch_factorial(j2 - m2) * torch_factorial(J + M) * \
        torch_factorial(J - M) / torch_factorial(j1 + j2 - J) / torch_factorial(j1 - j2 + J) / \
        torch_factorial(-1*j1 + j2 + J) / torch_factorial(j1 + j2 + J + 1)

    G = 0
    # 0 ≦ k ≦ j1+j2-J
    # 0 ≤ j1-m1-k ≤ j1-j2+J   <=>   j2-J-m1 ≤ k ≤ j1-m1
    # 0 ≤ j2+m2-k ≤ -j1+j2+J  <=>   j1-J+m2 ≤ k ≤ j2+m2
    lb = torch.tensor([0, j2 - J - m1, j1 - J + m2])
    ub = torch.tensor([j1 + j2 - J, j1 - m1, j2 + m2])
    for k in range(torch.max(lb), torch.min(ub) + 1):
        G += (-1)**k * torch_binom(j1 + j2 - J, k) * \
            torch_binom(j1 - j2 + J, j1 - m1 - k) * \
            torch_binom(-1*j1 + j2 + J, j2 + m2 - k)
    return torch.sqrt(N) * G
