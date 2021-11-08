import torch 

class Group(object) : 
    Lie_algebra = NotImplemented
    Discrete_Generators = NotImplemented 
    is_orthogonal = False
    is_permutation = False
    d = NotImplemented
    def __init__(self,*args,**kwargs):
        if self.d is NotImplemented: 
            if self.lie_algebra is not NotImplemented and len(self.lie_algebra):
                self.d= self.lie_algebra[0].shape[-1]
            if self.discrete_generators is not NotImplemented and len(self.discrete_generators):
                self.d= self.discrete_generators[0].shape[-1]

        
        if self.lie_algebra is NotImplemented:
            self.lie_algebra = torch.zeros((0,self.d,self.d))
        if self.discrete_generators is NotImplemented:
            self.discrete_generators = torch.zeros((0,self.d,self.d))
        
        if self.is_permutation: self.is_orthogonal=True
        if self.is_orthogonal is None:
            self.is_orthogonal = True
            if len(self.lie_algebra)!=0:
                A_dense =torch.stack([Ai@torch.eye(self.d) for Ai in self.lie_algebra])
                self.is_orthogonal &= rel_err(-A_dense.transpose((0,2,1)),A_dense)<1e-6
            if len(self.discrete_generators)!=0:
                h_dense = torch.stack([hi@torch.eye(self.d) for hi in self.discrete_generators])
                self.is_orthogonal &= rel_err(h_dense.transpose((0,2,1))@h_dense,torch.eye(self.d))<1e-6

        # Set regular flag automatically if not specified
        if self.is_orthogonal and (self.is_permutation is None):
            self.is_permutation=True
            self.is_permutation &= (len(self.lie_algebra)==0)  # no infinitesmal generators and all rows have one 1
            if len(self.discrete_generators)!=0:
                h_dense = torch.stack([hi@torch.eye(self.d) for hi in self.discrete_generators])
                self.is_permutation &= ((h_dense==1).astype(int).sum(-1)==1).all()

        def exp(self,A): 
            """Matrix exponential"""
            return torch.matrix_exp(A)

        def n_constraints(self) : 
            return len(self.lie_algebra)+len(self.discrete_generators) 
        
        def sample(self):
            """"""
            return self.samples(1)[0]


        def generate_example(self,N):
            """ """
            A_dense = torch.stack([Ai@torch.eye(self.d) for Ai in self.lie_algebra]) if len(self.lie_algebra) else torch.zeros((0,self.d,self.d))
            h_dense = torch.stack([hi@torch.eye(self.d) for hi in self.discrete_generators]) if len(self.discrete_generators) else torch.zeros((0,self.d,self.d))
            z = torch.random.randn(N,A_dense.shape[0])
            if self.z_scale is not None:
                z*= self.z_scale
            k = torch.random.randint(-5,5,size=(N,h_dense.shape[0],3))
            torch_seed=  torch.random.randint(100)
            return noise2samples(z,k,A_dense,h_dense,torch_seed)

    def __str__(self):
        return repr(self)

    
    def __repr__(self):
        outstr = f"{self.__class__}"
        if self.args:
            outstr += '('+''.join(repr(arg) for arg in self.args)+')'
        return outstr
    def __hash__(self):
        return hash(repr(self))

    def __mul__(self,other):
        return DirectProduct(self,other)

def rel_err(A,B):
        return torch.mean(torch.abs(A-B))/(torch.mean(torch.abs(A)) + torch.mean(torch.abs(B))+1e-6)



class Trivial(Group):
    """ The trivial group G={I} in n dimensions. If you want to see how the
        inductive biases of EMLP perform without any symmetry, use Trivial(n)"""
    def __init__(self,n):
        self.d = n
        super().__init__(n)



class SO(Group):
    """ The special orthogonal group SO(n) in n dimensions"""
    def __init__(self,n):
        self.lie_algebra = torch.zeros(((n*(n-1))//2,n,n))
        k=0
        for i in range(n):
            for j in range(i):
                self.lie_algebra[k,i,j] = 1
                self.lie_algebra[k,j,i] = -1
                k+=1
        super().__init__(n)



class O(SO):
    """ The Orthogonal group O(n) in n dimensions"""
    def __init__(self,n):
        self.discrete_generators = torch.eye(n)[None]
        self.discrete_generators[0,0,0]=-1
        super().__init__(n)


class SU(Group):  # Of dimension n^2-1
    """ The special unitary group SU(n) in n dimensions (complex)"""
    def __init__(self,n):
        if n==1: return Trivial(1)
        lie_algebra_real = torch.zeros((n**2-1,n,n))
        lie_algebra_imag = torch.zeros((n**2-1,n,n))
        k=0
        for i in range(n):
            for j in range(i):
                # Antisymmetric real generators
                lie_algebra_real[k,i,j] = 1
                lie_algebra_real[k,j,i] = -1
                k+=1
                # symmetric imaginary generators
                lie_algebra_imag[k,i,j] = 1
                lie_algebra_imag[k,j,i] = 1
                k+=1
        for i in range(n-1):
            # diagonal traceless imaginary generators
            lie_algebra_imag[k,i,i] = 1
            for j in range(n):
                if i==j: continue
                lie_algebra_imag[k,j,j] = -1/(n-1)
            k+=1
        self.lie_algebra = lie_algebra_real + lie_algebra_imag*1j
        super().__init__(n)



class U(Group):  # Of dimension n^2
    """ The unitary group U(n) in n dimensions (complex)"""
    def __init__(self,n):
        lie_algebra_real = torch.zeros((n**2,n,n))
        lie_algebra_imag = torch.zeros((n**2,n,n))
        k=0
        for i in range(n):
            for j in range(i):
                # Antisymmetric real generators
                lie_algebra_real[k,i,j] = 1
                lie_algebra_real[k,j,i] = -1
                k+=1
                # symmetric imaginary generators
                lie_algebra_imag[k,i,j] = 1
                lie_algebra_imag[k,j,i] = 1
                k+=1
        for i in range(n):
            # diagonal imaginary generators
            lie_algebra_imag[k,i,i] = 1
            k+=1
        self.lie_algebra = lie_algebra_real + lie_algebra_imag*1j
        super().__init__(n)



class SL(Group):
    """ The special linear group SL(n) in n dimensions"""
    def __init__(self,n):
        self.lie_algebra = torch.zeros((n*n-1,n,n))
        k=0
        for i in range(n):
            for j in range(n):
                if i==j: continue #handle diag elements separately
                self.lie_algebra[k,i,j] = 1
                k+=1
        for l in range(n-1):
            self.lie_algebra[k,l,l] = 1
            self.lie_algebra[k,-1,-1] = -1
            k+=1
        super().__init__(n)



class GL(Group):
    """ The general linear group GL(n) in n dimensions"""
    def __init__(self,n):
        self.lie_algebra = torch.zeros((n*n,n,n))
        k=0
        for i in range(n):
            for j in range(n):
                self.lie_algebra[k,i,j] = 1
                k+=1
        super().__init__(n)


class Scaling(Group):
    """ The scaling group in n dimensions"""
    def __init__(self,n):
        self.lie_algebra = torch.eye(n)[None]
        super().__init__(n)


class Parity(Group):  # """ The spacial parity group in 1+3 dimensions"""
    discrete_generators = -torch.eye(4)[None]
    discrete_generators[0,0,0] = 1

class TimeReversal(Group):  # """ The time reversal group in 1+3 dimensions"""
    discrete_generators = torch.eye(4)[None]
    discrete_generators[0,0,0] = -1

class SO13p(Group):
    """ The component of Lorentz group connected to identity"""
    lie_algebra = torch.zeros((6,4,4))
    lie_algebra[3:,1:,1:] = SO(3).lie_algebra
    for i in range(3):
        lie_algebra[i,1+i,0] = lie_algebra[i,0,1+i] = 1.

    # Adjust variance for samples along boost generators. For equivariance checks
    # the exps for high order tensors can get very large numbers
    z_scale = torch.array([.3,.3,.3,1,1,1]) # can get rid of now


class SO13(SO13p):
    discrete_generators = -torch.eye(4)[None]


class O13(SO13p):
    """ The full lorentz group (including Parity and Time reversal)"""
    discrete_generators = torch.eye(4)[None] +torch.zeros((2,1,1))
    discrete_generators[0] *= -1
    discrete_generators[1,0,0] = -1



class SO11p(Group):
    """ The identity component of O(1,1) (Lorentz group in 1+1 dimensions)"""
    lie_algebra = torch.array([[0.,1.],[1.,0.]])[None]


class O11(SO11p):
    """ The Lorentz group O(1,1) in 1+1 dimensions """
    discrete_generators = torch.eye(2)[None]+torch.zeros((2,1,1))
    discrete_generators[0]*=-1
    discrete_generators[1,0,0] = -1


class Sp(Group):
    """ Symplectic group Sp(m) in 2m dimensions (sometimes referred to
        instead as Sp(2m) )"""
    def __init__(self,m):
        self.lie_algebra = torch.zeros((m*(2*m+1),2*m,2*m))
        k=0
        for i in range(m): # block diagonal elements
            for j in range(m):
                self.lie_algebra[k,i,j] = 1
                self.lie_algebra[k,m+j,m+i] = -1
                k+=1
        for i in range(m):
            for j in range(i+1):
                self.lie_algebra[k,m+i,j] = 1
                self.lie_algebra[k,m+j,i] = 1
                k+=1
                self.lie_algebra[k,i,m+j] = 1
                self.lie_algebra[k,j,m+i] = 1
                k+=1
        super().__init__(m)

   
class Z(Group):
    r""" The cyclic group Z_n (discrete translation group) of order n.
        Features a regular base representation."""
    def __init__(self,n):
        self.discrete_generators = [LazyShift(n)]
        super().__init__(n)



class S(Group): #The permutation group
    r""" The permutation group S_n with an n dimensional regular representation."""
    def __init__(self,n):
        # Here we choose n-1 generators consisting of swaps between the first element
        # and every other element
        perms = torch.arange(n)[None]+torch.zeros((n-1,1)).astype(int)
        perms[:,0] = torch.arange(1,n)
        perms[torch.arange(n-1),torch.arange(1,n)[None]]=0
        self.discrete_generators = [LazyPerm(perm) for perm in perms]
        super().__init__(n)

        # We can also have chosen the 2 generator soln described in the paper, but
        # adding superflous extra generators surprisingly can sometimes actually *decrease* 
        # the runtime of the iterative krylov solver by improving the conditioning 
        # of the constraint matrix