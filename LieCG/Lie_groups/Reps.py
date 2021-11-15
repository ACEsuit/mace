import torch
from Rotations import LorentzD, WignerD, matrix_to_angles
from Groups import Group,SO,SU,SO13
class Rep(object):
    r""" The base Representation class. Representation objects formalize the vector space V
        on which the group acts, the group representation matrix ρ(g), and the Lie Algebra
        representation dρ(A) in a single object. Representations act as types for vectors coming
        from V. These types can be manipulated and transformed with the built in operators
        ⊕,⊗,dual, as well as incorporating custom representations. Rep objects should
        be immutable.

        At minimum, new representations need to implement ``rho``, ``__str__``."""
        
    is_permutation=False

    def rho(self,M):
        """ Group representation of the matrix M of shape (d,d)"""
        raise NotImplementedError

        
    def drho(self,A): 
        """ Lie Algebra representation of the matrix A of shape (d,d)"""
        In = torch.eye(A.shape[0])
        return LazyJVP(self.rho,In,A)


    def __call__(self,G):
        """ Instantiate (non concrete) representation with a given symmetry group"""
        raise NotImplementedError

    def __str__(self): raise NotImplementedError 
    #TODO: separate __repr__ and __str__?
    def __repr__(self): return str(self)
    
    
    def __eq__(self,other):
        if type(self)!=type(other): return False
        d1 = tuple([(k,v) for k,v in self.__dict__.items() if (k not in ['_size','is_permutation','is_orthogonal'])])
        d2 = tuple([(k,v) for k,v in other.__dict__.items() if (k not in ['_size','is_permutation','is_orthogonal'])])
        return d1==d2
    def __hash__(self):
        d1 = tuple([(k,v) for k,v in self.__dict__.items() if (k not in ['_size','is_permutation','is_orthogonal'])])
        return hash((type(self),d1))

    def size(self): 
        """ Dimension dim(V) of the representation """
        if hasattr(self,'_size'):
            return self._size
        elif self.concrete and hasattr(self,"G"):
            self._size = self.rho(self.G.sample()).shape[-1]
            return self._size
        else: raise NotImplementedError





    def __add__(self, other):
        """ Direct sum (⊕) of representations. """
        if isinstance(other,int):
            if other==0: return self
            else: return self+other*Scalar
        elif emlp.reps.product_sum_reps.both_concrete(self,other):
            return emlp.reps.product_sum_reps.SumRep(self,other)
        else:
            return emlp.reps.product_sum_reps.DeferredSumRep(self,other)

    def __radd__(self,other):
        if isinstance(other,int): 
            if other==0: return self
            else: return other*Scalar+self
        else: return NotImplemented
        
    def __mul__(self,other):
        """ Tensor sum (⊗) of representations. """
        return mul_reps(self,other)
            
    def __rmul__(self,other):
        return mul_reps(other,self)

    def __pow__(self,other):
        """ Iterated tensor product. """
        assert isinstance(other,int), f"Power only supported for integers, not {type(other)}"
        assert other>=0, f"Negative powers {other} not supported"
        return reduce(lambda a,b:a*b,other*[self],Scalar)
    def __rshift__(self,other):
        """ Linear maps from self -> other """
        return other*self.T
    def __lshift__(self,other):
        """ Linear maps from other -> self """
        return self*other.T
    def __lt__(self, other):
        """ less than defined to disambiguate ordering multiple different representations.
            Canonical ordering is determined first by Group, then by size, then by hash"""
        if other==Scalar: return False
        try: 
            if self.G<other.G: return True
            if self.G>other.G: return False
        except (AttributeError,TypeError): pass
        if self.size()<other.size(): return True
        if self.size()>other.size(): return False
        return hash(self) < hash(other) #For sorting purposes only
    def __mod__(self,other): # Wreath product
        """ Wreath product of representations (Not yet implemented)"""
        raise NotImplementedError
    @property
    def T(self):
        """ Dual representation V*, rho*, drho*."""
        if hasattr(self,"G") and (self.G is not None) and self.G.is_orthogonal: return self
        return Dual(self)




def mul_reps(ra,rb:int):
    if rb==1: return ra
    if rb==0: return 0
    if (not hasattr(ra,'concrete')) or ra.concrete:
        return emlp.reps.product_sum_reps.SumRep(*(rb*[ra]))
    else:
        return emlp.reps.product_sum_reps.DeferredSumRep(*(rb*[ra]))


def mul_reps(ra:int,rb):
    return mul_reps(rb,ra)

# Continued with non int cases in product_sum_reps.py

# A possible
class ScalarRep(Rep):
    def __init__(self,G=None):
        self.G=G
        self.is_permutation = True
    def __call__(self,G):
        self.G=G
        return self
    def size(self):
        return 1
    def __repr__(self): return str(self)#f"T{self.rank+(self.G,)}"
    def __str__(self):
        return "V⁰"
    @property
    def T(self):
        return self
    def rho(self,M):
        return torch.eye(1)
    def drho(self,M):
        return 0*torch.eye(1)
    def __hash__(self):
        return 0
    def __eq__(self,other):
        return isinstance(other,ScalarRep)
    def __mul__(self,other):
        if isinstance(other,int): return super().__mul__(other)
        return other
    def __rmul__(self,other):
        if isinstance(other,int): return super().__rmul__(other)
        return other
    @property
    def concrete(self):
        return True

class Base(Rep):
    """ Base representation V of a group."""
    def __init__(self,G=None):
        self.G=G
        if G is not None: self.is_permutation = G.is_permutation
    def __call__(self,G):
        return self.__class__(G)
    def rho(self,M):
        if hasattr(self,'G') and isinstance(M,dict): M=M[self.G]
        return M
    def drho(self,A):
        if hasattr(self,'G') and isinstance(A,dict): A=A[self.G]
        return A
    def size(self):
        assert self.G is not None, f"must know G to find size for rep={self}"
        return self.G.d
    def __repr__(self): return str(self)#f"T{self.rank+(self.G,)}"
    def __str__(self):
        return "V"# +(f"_{self.G}" if self.G is not None else "")
    
    def __hash__(self):
        return hash((type(self),self.G))
    def __eq__(self,other):
        return type(other)==type(self) and self.G==other.G
    def __lt__(self,other):
        if isinstance(other,Dual): return True
        return super().__lt__(other)
    # @property
    # def T(self):
    #     return Dual(self.G)

class Dual(Rep):
    def __init__(self,rep):
        self.rep = rep
        self.G=rep.G
        if hasattr(rep,"is_permutation"): self.is_permutation = rep.is_permutation
    def __call__(self,G):
        return self.rep(G).T
    def rho(self,M):
        rho = self.rep.rho(M)
        rhoinvT = rho.invT() if isinstance(rho,LinearOperator) else torch.linalg.inv(rho).T
        return rhoinvT
    def drho(self,A):
        return -self.rep.drho(A).T
    def __str__(self):
        return str(self.rep)+"*"
    def __repr__(self): return str(self)
    @property
    def T(self):
        return self.rep
    def __eq__(self,other):
        return type(other)==type(self) and self.rep==other.rep
    def __hash__(self):
        return hash((type(self),self.rep))
    def __lt__(self,other):
        if other==self.rep: return False
        return super().__lt__(other)
    def size(self):
        return self.rep.size()
        
V=Vector= Base()  #: Alias V or Vector for an instance of the Base representation of a group

Scalar = ScalarRep()#: An instance of the Scalar representation, equivalent to V**0

class SO3Irreps(Rep):
    """ (Real) Irreducible representations of SO3 """
    is_regular=False
    def __init__(self,order):
        assert order>0, "Use Scalar for 𝜓₀"
        self.G=SO(3)
        self.order = order
    def size(self):
        return 2*self.order + 1
    def rho(self,M):
        alpha,beta,gamma = matrix_to_angles(M)
        return WignerD(self.order,alpha,beta,gamma)[0] #take the real part
    def __str__(self):
        number2sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return f"𝜓{self.order}".translate(number2sub)
    def __eq__(self,other):
        return type(self)==type(other) and self.G==other.G and self.order==other.order
    def __hash__(self):
        return hash((type(self),self.G,self.order))
    @property
    def T(self):
        return self

class SU2Irreps(Rep):
    """ (Real) Irreducible representations of SO3 """
    is_regular=False
    def __init__(self,order):
        assert order>0, "Use Scalar for 𝜓₀"
        self.G=SU(2)
        self.order = order
    def size(self):
        return 2*self.order + 1
    def rho(self,M):
        irreps = o3.Irreps('1x' + str(self.order) + 'e')
        alpha,beta,gamma = matrix_to_angles(M)
        return WignerD(self.order,alpha,beta,gamma)
    def __str__(self):
        number2sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return f"𝜓{self.order}".translate(number2sub)
    def __eq__(self,other):
        return type(self)==type(other) and self.G==other.G and self.order==other.order
    def __hash__(self):
        return hash((type(self),self.G,self.order))
    @property
    def T(self):
        return self

class SO13Irreps(Rep):
    """ (Real) Irreducible representations of SO3 """
    is_regular=False
    def __init__(self,order):
        assert order>0, "Use Scalar for 𝜓₀"
        self.G=SO13()
        self.order = order
    def size(self):
        return 2*sum(self.order) + 1
    def rho(self,M):
        alpha,beta,gamma = Matrix_to_euler(M)
        return LorentzD(self.order,alpha,beta,gamma)
    def __str__(self):
        number2sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return f"𝜓{self.order}".translate(number2sub)
    def __eq__(self,other):
        return type(self)==type(other) and self.G==other.G and self.order==other.order
    def __hash__(self):
        return hash((type(self),self.G,self.order))
    @property
    def T(self):
        return self

print(SO3Irreps(1).__str__())