import operator
from collections.abc import Iterable
from functools import reduce
from typing import Dict

import torch


def spreshape(sp, ncol):
    """
    reshape a sparse matrix 
    """
    row, col, val = sp.indices()[0],sp.indices()[1],sp.values()
    index = row * sp.size(1) + col
    row = index // ncol
    col = index % ncol
    index = torch.stack((row,col))
    sz = torch.Size((sp.size(0) // ncol,ncol))
    return torch.sparse_coo_tensor(
        index,
        val,
        sz,   
    ).coalesce()    



def ravel_multi_index(arr, shape):  # pragma: no cover
    """
    implements a subset of the functionality of np.ravel_multi_index.
    """
    shape = torch.tensor(shape)
    total = 0
    for i, a in enumerate(arr[:-1], 1):
        total += a * torch.prod(shape[i:])
    total += arr[-1]
    return total
    
    

    
#from sparse.pydata
#https://sparse.pydata.org/en/latest/_modules/sparse/_coo/core.html#COO.reshape
def reshape(a, shape, device='cpu', order="C"):
        """
        Returns a new :obj:`COO` array that is a reshaped version of this array.

        Parameters
        ----------
        shape : tuple[int]
            The desired shape of the output array.

        Returns
        -------
        torch.sparse_COO
            The reshaped output array.

        See Also
        --------

        Notes
        -----
        The :code:`order` parameter is provided just for compatibility with
        Numpy and isn't actually supported.

        """
        
        if isinstance(shape, Iterable):
            shape = tuple(shape)
        else:
            shape = (shape,)

        if order not in {"C", None}:
            raise NotImplementedError("The `order` parameter is not supported")
            
        if list(a.shape) == shape:
            return a
        size = reduce((lambda x, y: x * y), list(a.shape))
        
        if any(d == -1 for d in shape):
            extra = int(size / torch.prod(torch.tensor([d for d in shape if d != -1])))
            shape = tuple([d if d != -1 else extra for d in shape])
                
        if size != reduce(operator.mul, shape, 1):
            raise ValueError(
                "cannot reshape array of size {} into shape {}".format(list(a.size), shape)
            )


        # TODO: this self.size enforces a 2**64 limit to array size
        linear_loc = ravel_multi_index(a._indices(),a.shape)

        idx_dtype = a._indices().dtype

        coords = torch.empty((len(shape), len(a._values())), dtype=idx_dtype, device=device)
        strides = 1
        for i, d in enumerate(shape[::-1]):
            coords[-(i + 1), :] = (linear_loc // strides) % d
            strides *= d

        result = torch.sparse_coo_tensor(
            coords,
            a.values(),
            shape,
        ).coalesce()

        return result



def unfold(tensor,mode,device='cpu'):
    """Unfolds the mode-`mode` unfolding into a tensor
    
    Parameters
    ----------
    tensor : torch.sparse.tensor
            
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding
    
    Returns
    -------
    torch.sparse.tensor
        folded_tensor of shape `shape`
    """
    if not tensor.is_sparse:
        raise ValueError('Matricisization only for sparse tensors')

    if tensor.dim() <= 2 :
        tensor = tensor.transpose(mode,0) #for consitency
        return tensor 
    
    else :
        tensor = tensor.transpose(mode,0).coalesce()
        tensor = reshape(tensor,(tensor.shape[mode],-1),device=device)
        sz = [tensor.size()[1],tensor.size()[0]]
        tensor_coo_t = torch.sparse_coo_tensor(tensor._indices()[[1,0]],tensor.values(),torch.Size(sz))
        return tensor_coo_t.coalesce()
 

def fold(unfolded_tensor, mode, shape, device='cpu'):
    """Refolds the mode-`mode` unfolding into a tensor of shape `shape`
    
        In other words, refolds the n-mode unfolded tensor
        into the original tensor of the specified shape.
    
    Parameters
    ----------
    unfolded_tensor : torch.sparse.tensor
        unfolded tensor of shape ``(shape[mode], -1)``
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding
    
    Returns
    -------
    torch.sparse.tensor
        folded_tensor of shape `shape`
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    tensor = reshape(unfolded_tensor.transpose(0,1).coalesce(), full_shape, device=device)
    return tensor.transpose(0,mode).coalesce()



"""def expand_sparse(tensor,n_copies, N = 0) :
    if N == tensor.dim():
        return tensor
    else : 
        tensor = torch.cat([tensor for n in range(n_copies)],N)
    return expand_sparse(tensor,n_copies,N+1)"""


def expand_sparse(tensor,n_copies, N = 1) :
    if N == 0:
        return tensor
    else : 
        tensor = torch.cat([tensor for n in range(n_copies)],N)
    return expand_sparse(tensor,n_copies,N-1)




def expand_indices(ind,num_elements,correlation):
    n_rep = num_elements**correlation #number of permutation of the basis for element
    mem = 0
    ind_exp = torch.tensor([mem])
    flag_rep = 0
    for i in range(0,len(ind)) :
        if ind[i] == mem :
            flag_rep += 1

        if ind[i] != mem or i+1 == len(ind): 
            ind_rep = []
            for n_ele in range(n_rep):
                ind_rep += [ind_exp[-1] + 1 + n_ele]*(flag_rep)
            ind_rep = torch.tensor(ind_rep)
            ind_exp = torch.cat((ind_exp,ind_rep))
            if flag_rep != 1 and i+1 == len(ind) :
                flag_rep = 0
            else :
                flag_rep = 1
            mem = ind[i]
      
        if flag_rep == 1 and i+1 == len(ind) :
            ind_rep = []
            for n_ele in range(n_rep):
                ind_rep += [ind_exp[-1] + 1 + n_ele]*(flag_rep)
            ind_rep = torch.tensor(ind_rep)
            ind_exp = torch.cat((ind_exp,ind_rep))
    
    return ind_exp[1:] -1
            
        

def expand_element(tensor,num_elements) :
    correlation = len(tensor.size()) - 1
    sz_unit = tensor.size()
    size_0 = [sz_unit[0]*num_elements**correlation] #expand for different element combination of the basis
    index_elem = expand_indices(tensor.indices()[0],num_elements,correlation)
    tensor = expand_sparse(tensor,num_elements,N=correlation).coalesce()   
    index = torch.cat((index_elem.unsqueeze(dim=0),tensor.indices()[1:]),dim=0)
    sz_exp = tensor.size()[1:]
    size = size_0 + list(sz_exp)
    return torch.sparse_coo_tensor(index,tensor.values(),torch.Size(size)).coalesce()

class tensor_contract_nd_update_sparse(torch.nn.Module):
    
    """Contract the c_tilde::torch.SparseTensor tensor coefficient and update in 3 steps:
        1. At initialization, unfold the c_tilde tensor in t dimension
        2. """
    
    def __init__(self,
                 c_tilde : torch.sparse_coo,
                 correlation : int,
                 device = 'cuda', #best to use CPU because faster
                ):
        super().__init__()
        
        self.device = device
        self.correlation = correlation
        c_tilde = torch.sparse_coo_tensor(c_tilde.indices(),c_tilde.values(),c_tilde.size()).coalesce()
        mock = torch.sparse.mm(c_tilde,torch.ones(c_tilde.size()[1])[:,None].to(self.device).double())
        mock = mock.reshape((c_tilde.size()[1]**(self.correlation-2),c_tilde.size()[1]))
        self.idx = torch.nonzero(mock,as_tuple=True)#to speed the
        del mock
        
    def forward(self,
                data : Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor] : #Pass the previous contracted tensor in a_update of size N_basis*(correlation_order-1),N_basis
        
        atomic_basis = data['atomic_basis']
        c_tildes_dict_w = data['c_tildes_dict_w']
        if 'a_update' in data : #check if first update 
            self.c_tilde_u = ((c_tildes_dict_w[self.correlation] + data['a_update'][0]).coalesce(),(data['a_update'][1]).coalesce()) #update the c_tilde with previous correlation order 
            a = complex_spmm(self.c_tilde_u,atomic_basis) #sparse matrix of size [N_basis*(correlation_order-1),1]
        else :
            a = complex_spmm((c_tildes_dict_w[self.correlation],),atomic_basis) 

        a = (a[0].reshape(len(atomic_basis[0])**(self.correlation-2),len(atomic_basis[0])),
             a[1].reshape(len(atomic_basis[0])**(self.correlation-2),len(atomic_basis[0])))
        a = (torch.sparse_coo_tensor(torch.stack(self.idx),a[0][self.idx],a[0].size()),
             torch.sparse_coo_tensor(torch.stack(self.idx),a[1][self.idx],a[1].size()))
        data['a_update']  = a
        return data
    

class vector_contract(torch.nn.Module):
    
    def __init__(self,):
        super().__init__()
    def forward(self,data : Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor] :
                
        atomic_basis = data['atomic_basis']
        c_tilde_dict_w = data['c_tildes_dict_w']
        self.c_tilde = c_tilde_dict_w[1].transpose(0,1).coalesce() #size [1,N_basis] sparse tensor
        if 'a_update' in data : #check if first update 
            self.c_tilde_u = ((self.c_tilde + data['a_update'][0]).coalesce(),(data['a_update'][1]).coalesce()) #update the c_tilde with previous correlation order 
            data['a_update'] = complex_spmm(self.c_tilde_u,atomic_basis)
        else :
            self.c_tilde_u = self.c_tilde
            data['a_update'] = complex_spmm((self.c_tilde_u,),atomic_basis)
        return data
