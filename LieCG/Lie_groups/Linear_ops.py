import torch
from functools import reduce 
from Lie_groups.Linear_ops_base import LinearOperator,Lazy


product = lambda c: reduce(lambda a,b:a*b,c)

#Taken on https://github.com/mfinzi/equivariant-MLP from Marc Finzi


def lazify(x):
    if isinstance(x,LinearOperator): return x
    elif isinstance(x,(torch.tensor)): return Lazy(x)
    else: raise NotImplementedError

def densify(x):
    if isinstance(x,LinearOperator): return x.to_dense()
    elif isinstance(x,(torch.tensor)): return x


class I(LinearOperator):
    def __init__(self,d):
        shape = (d,d)
        super().__init__(None, shape)
    def _matmat(self,V): #(c,k)
        return V
    def _matvec(self,V):
        return V
    def _adjoint(self):
        return self
    def invT(self):
        return self

class LazyKron(LinearOperator):

    def __init__(self,Ms):
        self.Ms = Ms
        shape = product([Mi.shape[0] for Mi in Ms]), product([Mi.shape[1] for Mi in Ms])
        super().__init__(None,shape)

    def _matvec(self,v):
        return self._matmat(v).reshape(-1)
    def _matmat(self,v):
        ev = v.reshape(*[Mi.shape[-1] for Mi in self.Ms],-1)
        for i,M in enumerate(self.Ms):
            ev_front = torch.moveaxis(ev,i,0)
            Mev_front = (M@ev_front.reshape(M.shape[-1],-1)).reshape(M.shape[0],*ev_front.shape[1:])
            ev =torch.moveaxis(Mev_front,0,i)
        return ev.reshape(self.shape[0],ev.shape[-1])
    def _adjoint(self):
        return LazyKron([Mi.T for Mi in self.Ms])
    def invT(self):
        return LazyKron([M.invT() for M in self.Ms])
    def to_dense(self):
        Ms = [M.to_dense() if isinstance(M,LinearOperator) else M for M in self.Ms]
        return reduce(jnp.kron,Ms)
    def __new__(cls,Ms):
        if len(Ms)==1: return Ms[0]
        return super().__new__(cls)


def kronsum(A,B):
    return torch.kron(A,torch.eye(B.shape[-1])) + torch.kron(torch.eye(A.shape[-1]),B)


class LazyKronsum(LinearOperator):
    
    def __init__(self,Ms):
        self.Ms = Ms
        shape = product([Mi.shape[0] for Mi in Ms]), product([Mi.shape[1] for Mi in Ms])
        dtype=torch.cfloat
        super().__init__(dtype,shape)

    def _matvec(self,v):
        return self._matmat(v).reshape(-1)

    def _matmat(self,v):
        ev = v.reshape(*[Mi.shape[-1] for Mi in self.Ms],-1)
        out = 0*ev
        for i,M in enumerate(self.Ms):
            ev_front = torch.moveaxis(ev,i,0)
            Mev_front = (M@ev_front.reshape(M.shape[-1],-1)).reshape(M.shape[0],*ev_front.shape[1:])
            out += torch.moveaxis(Mev_front,0,i)
        return out.reshape(self.shape[0],ev.shape[-1])
        
    def _adjoint(self):
        return LazyKronsum([Mi.T for Mi in self.Ms])
    def to_dense(self):
        Ms = [M.to_dense() if isinstance(M,LinearOperator) else M for M in self.Ms]
        return reduce(kronsum,Ms)
    def __new__(cls,Ms):
        if len(Ms)==1: return Ms[0]
        return super().__new__(cls)

class JVP(LinearOperator):
    def __init__(self,operator_fn,X,TX):
        self.shape = operator_fn(X).shape
        self.vjp = lambda v: torch.autograd.functional.jvp(lambda x: operator_fn(x)@v,[X],[TX])[1]
        self.vjp_T = lambda v: torch.autograd.functional.jvp(lambda x: operator_fn(x).T@v,[X],[TX])[1]
        self.dtype= torch.cfloat
    def _matmat(self,v):
        return self.vjp(v)
    def _matvec(self,v):
        return self.vjp(v)
    def _rmatmat(self,v):
        return self.vjp_T(v)


class ConcatLazy(LinearOperator):
    """ Produces a linear operator equivalent to concatenating
        a collection of matrices Ms along axis=0 """
    def __init__(self,Ms):
        self.Ms = Ms
        assert all(M.shape[0]==Ms[0].shape[0] for M in Ms),\
             f"Trying to concatenate matrices of different sizes {[M.shape for M in Ms]}"
        shape = (sum(M.shape[0] for M in Ms),Ms[0].shape[1])
        super().__init__(None,shape)

    def _matmat(self,V):
        return torch.concatenate([M@V for M in self.Ms],axis=0)
    def _rmatmat(self,V):
        Vs = torch.split(V,len(self.Ms))
        return sum([self.Ms[i].T@Vs[i] for i in range(len(self.Ms))])
    def to_dense(self):
        dense_Ms = [M.to_dense() if isinstance(M,LinearOperator) else M for M in self.Ms]
        return torch.concatenate(dense_Ms,axis=0)
    
class LazyDirectSum(LinearOperator):
    def __init__(self,Ms,multiplicities=None):
        self.Ms = [ M for M in Ms]
        self.multiplicities = [1 for M in Ms] if multiplicities is None else multiplicities
        shape = (sum(Mi.shape[0]*c for Mi,c in zip(Ms,multiplicities)),
                      sum(Mi.shape[0]*c for Mi,c in zip(Ms,multiplicities)))
        super().__init__(None,shape)
        #self.dtype=Ms[0].dtype
        #self.dtype=jnp.dtype('float32')

    def _matvec(self,v):
        return lazy_direct_matmat(v,self.Ms,self.multiplicities)

    def _matmat(self,v): # (n,k)
        return lazy_direct_matmat(v,self.Ms,self.multiplicities)
    def _adjoint(self):
        return LazyDirectSum([Mi.T for Mi in self.Ms])
    def invT(self):
        return LazyDirectSum([M.invT() for M in self.Ms])
    def to_dense(self):
        Ms_all = [M for M,c in zip(self.Ms,self.multiplicities) for _ in range(c)]
        Ms_all = [Mi.to_dense() if isinstance(Mi,LinearOperator) else Mi for Mi in Ms_all]
        return jax.scipy.linalg.block_diag(*Ms_all)
    # def __new__(cls,Ms,multiplicities=None):
    #     if len(Ms)==1 and multiplicities is None: return Ms[0]
    #     return super().__new__(cls)
        
def lazy_direct_matmat(v,Ms,mults):
    n = v.shape[0]
    k = v.shape[1] if len(v.shape)>1 else 1
    i=0
    y = []
    for M, multiplicity in zip(Ms,mults):
        i_end = i+multiplicity*M.shape[-1]
        elems = M@v[i:i_end].T.reshape(k*multiplicity,M.shape[-1]).T
        y.append(elems.T.reshape(k,multiplicity*M.shape[0]).T)
        i = i_end
    y = torch.concatenate(y,axis=0) #concatenate over rep axis
    return  y
