import torch
from os import sys

sys.path.append(r'C:\Users\Lilyes\Documents\GitHub\LieCG\LieCG')

from tqdm.auto import tqdm
import logging
import itertools
from Lie_groups.Reps import SU2Irreps
from Lie_groups.Linear_ops import ConcatLazy, LazyKron, LazyKronsum
from Lie_groups.Groups import SU

class ConvergenceError(Exception): pass




class Clebsch_Gordan():
    def __init__(self,Group,lmax : int, irreps) :
        self.Group = Group
        self.lmax = lmax
        self.irreps = irreps
        self.lie_generators = Group.lie_algebra
        self.discrete_generators = Group.discrete_generators

    def constraint_matrix(self,l1,l2,l3) : 
        #Create the constraint matrix for CG(l1,l2,l3). The Nullspace of this constraint matrix corresponds to the 
        # vectorization of the CG coefficients. 

        self.irreps_l1 = self.irreps(l1)
        self.irreps_l2 = self.irreps(l2)
        self.irreps_l3 = self.irreps(l3)
        
        A =[]
        print(self.irreps_l1.drho)
        A.extend([LazyKronsum([self.irreps_l1.drho(Ti), self.irreps_l2.drho(Ti)]) for Ti in self.lie_generators]) #replace with lazy for saving cost
        A.extend([LazyKron(self.irreps_l1.rho(hi), self.irreps_l2.rho(hi)) for hi in self.discrete_generators]) #replace with lazy for cost
        A = ConcatLazy(A)#replace with lazy for cost

        B = []
        B.extend([self.irreps_l3.drho(Ti) for Ti in self.lie_generators])
        B.extend([self.irreps_l3.rho(hi) for hi in self.discrete_generators])
        B = ConcatLazy(B)

        const_m = LazyKronsum([A.T,-B])    

        return const_m
    
    def dict(self) : 
        len_weights = len(self.lmax)
        l_tuples = list(i for i in itertools.product() if i[0] >= i[1])

        CG = dict()

        for lls in l_tuples : 
            C = self.constraint_matrix(self,lls[0],lls[1],lls[2])
            CG_lls = krylov_constraint_solve(C)
            CG[lls] = CG_lls
    
        return CG
        
        

 



def krylov_constraint_solve(C,tol=1e-5):
    """ Computes the solution basis Q for the linear constraint CQ=0  and QᵀQ=I
        up to specified tolerance with C expressed as a LinearOperator. """
    r = 5
    if C.shape[0]*r*2>2e9: raise Exception(f"Solns for contraints {C.shape} too large to fit in memory")
    found_rank=5
    while found_rank==r:
        r *= 2 # Iterative doubling of rank until large enough to include the full solution space
        if C.shape[0]*r>2e9:
            logging.error(f"Hit memory limits, switching to sample equivariant subspace of size {found_rank}")
            break
        Q = krylov_constraint_solve_upto_r(C,r,tol)
        found_rank = Q.shape[-1]
    return Q

def krylov_constraint_solve_upto_r(C,r,tol=1e-5,lr=1e-2):#,W0=None):
    """ Iterative routine to compute the solution basis to the constraint CQ=0 and QᵀQ=I
        up to the rank r, with given tolerance. Uses gradient descent (+ momentum) on the
        objective |CQ|^2, which provably converges at an exponential rate."""
    W0 = torch.randn(C.shape[-1],r)/torch.sqrt(C.shape[-1])# if W0 is None else W0
    W = torch.nn.parameter(W0)
    optimizer = torch.optim.SGD(W, lr=lr,momentum=0.9)
    #opt_init,opt_update = optax.sgd(lr,.9)
    #opt_state = opt_init(W)  # init stats

    def loss(W):
        return (torch.abs(C@W)**2).sum()/2 # added absolute for complex support

    # setup progress bar
    pbar = tqdm(total=100,desc=f'Krylov Solving for Equivariant Subspace r<={r}',
    bar_format="{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    prog_val = 0
    lstart = loss(W) 
    
    for i in range(20000):
        lossval = loss(W)
        optimizer.zero_grad()
        lossval.backward()
        optimizer.step()
        # update progress bar
        progress = max(100*torch.log(lossval/lstart)/torch.log(tol**2/lstart)-prog_val,0)
        progress = min(100-prog_val,progress)
        if progress>0:
            prog_val += progress
            pbar.update(progress)

        if torch.sqrt(lossval) <tol: # check convergence condition
            pbar.close()
            break # has converged
        if lossval>2e3 and i>100: # Solve diverged due to too high learning rate
            logging.warning(f"Constraint solving diverged, trying lower learning rate {lr/3:.2e}")
            if lr < 1e-4: raise ConvergenceError(f"Failed to converge even with smaller learning rate {lr:.2e}")
            return krylov_constraint_solve_upto_r(C,r,tol,lr=lr/3)
    else: raise ConvergenceError("Failed to converge.")
    # Orthogonalize solution at the end
    U,S,VT = torch.linalg.svd(W,full_matrices=False) 
    # Would like to do economy SVD here (to not have the unecessary O(n^2) memory cost) 
    # but this is not supported in numpy (or Jax) unfortunately.
    rank = (S>10*tol).sum()
    Q = U[:,:rank]
    # final_L
    final_L = loss(Q)
    if final_L >tol: logging.warning(f"Normalized basis has too high error {final_L:.2e} for tol {tol:.2e}")
    scutoff = (S[rank] if r>rank else 0)
    assert rank==0 or scutoff < S[rank-1]/100, f"Singular value gap too small: {S[rank-1]:.2e} \
        above cutoff {scutoff:.2e} below cutoff. Final L {final_L:.2e}, earlier {S[rank-5:rank]}"
    #logging.debug(f"found Rank {r}, above cutoff {S[rank-1]:.3e} after {S[rank] if r>rank else np.inf:.3e}. Loss {final_L:.1e}")
    return Q

#print(Clebsch_Gordan(Group=SU(2),lmax=3,irreps=SU2Irreps).constraint_matrix(1/2,1/2,1))