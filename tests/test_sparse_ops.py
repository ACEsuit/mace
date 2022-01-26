import torch
from torch_sparse import spmm

from LieACE.modules.sparse_tools import tensor_contract_nd_update_sparse, unfold, vector_contract



_size = 300
d = 0.999999999
dim = 4
size_values = torch.round(torch.tensor((1-d)*_size**dim))
idx = torch.LongTensor(torch.randint(_size,[dim,int(size_values.item())]))
values = torch.randn(int(size_values))
c_tilde4 = torch.sparse.FloatTensor(idx ,values, torch.Size([_size]*dim)).coalesce()
c_tilde4_u = unfold(c_tilde4,0)
c_tilde4_u = c_tilde4_u.to('cuda')
c_tilde4_u.requires_grad = True


_size = 300
d = 0.999999
dim = 3
size_values = torch.round(torch.tensor((1-d)*_size**dim))
idx = torch.LongTensor(torch.randint(_size,[dim,int(size_values.item())]))
values = torch.randn(int(size_values))
c_tilde3 = torch.sparse.FloatTensor(idx ,values, torch.Size([_size]*dim)).coalesce()
c_tilde3_u = unfold(c_tilde3,0)
c_tilde3_u = c_tilde3_u.to('cuda')
c_tilde3_u.requires_grad = True

_size = 300
d = 0.95
dim = 2
size_values = torch.round(torch.tensor((1-d)*_size**dim))
idx = torch.LongTensor(torch.randint(_size,[dim,int(size_values.item())]))
values = torch.randn(int(size_values))
c_tilde2 = torch.sparse.FloatTensor(idx ,values, torch.Size([_size]*dim)).coalesce()
c_tilde2_u = unfold(c_tilde2,0)
c_tilde2_u = c_tilde2_u.to('cuda')
c_tilde2_u.requires_grad = True


_size = 300
d = 0.0
dim = 1
size_values = torch.round(torch.tensor((1-d)*_size**dim))
idx = torch.LongTensor(torch.randint(_size,[dim,int(size_values.item())]))
values = torch.randn(int(size_values))
c_tilde1 = torch.sparse.FloatTensor(idx ,values, torch.Size([_size]*dim)).coalesce()
c_tilde1 = c_tilde1.to('cuda')
c_tilde1 = c_tilde1.to_dense()[None,:].to_sparse()
c_tilde1.requires_grad = True




contract = torch.nn.Sequential(
    tensor_contract_nd_update_sparse(c_tilde3_u.float(),dim=0,correlation=3),
    tensor_contract_nd_update_sparse(c_tilde2_u.float(),dim=0,correlation=2),
    vector_contract(c_tilde1.float()),
    )

atomic_basis = torch.randn(96)[:,None]
atomic_basis = atomic_basis.to('cuda')
atomic_basis.requires_grad = True
data = {'atomic_basis' : atomic_basis}





with torch.autograd.profiler.profile(use_cuda=True) as prof:
    a = contract(data)
print(prof)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    a = spmm(c_tilde4_u.indices(),c_tilde4_u.values(),27000000,300,atomic_basis)
    a = a.reshape(len(atomic_basis)**(contract[0].correlation-2),len(atomic_basis))
    a = torch.sparse_coo_tensor(torch.stack(contract[0].idx),a[contract[0].idx],a.size())
print(prof)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    a = spmm(c_tilde4_u.indices(),c_tilde4_u.values(),27000000,300,atomic_basis)
    a = a.reshape(len(atomic_basis)**(contract[0].correlation-2),len(atomic_basis))
    a = torch.sparse_coo_tensor(torch.stack(contract[0].idx),a[contract[0].idx],a.size())
print(prof)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    a = contract[0].weights * spmm(c_tilde4_u.indices(),c_tilde4_u.values(),27000000,300,atomic_basis)
    
print(prof)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    a = contract[0].weights * spmm(c_tilde4_u.indices(),c_tilde4_u.values(),27000000,300,atomic_basis)
    a =  a.reshape(len(atomic_basis)**(contract[0].correlation-2),len(atomic_basis))
print(prof)


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    a = contract[0].weights * spmm(c_tilde4_u.indices(),c_tilde4_u.values(),27000000,300,atomic_basis)
    a =  a.reshape(len(atomic_basis)**(contract[0].correlation-2),len(atomic_basis))
    a = torch.sparse_coo_tensor(torch.stack(contract[0].idx),a[contract[0].idx],a.size())
print(prof)


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    contract(data)
print(prof)

spmm(c_tilde4_u.indices(),c_tilde4_u.values(),64000000,400,atomic_basis)

c_tilde3_p = c_tilde3.to_dense().permute(2,1,0)
c_tilde2_p = c_tilde2.to_dense().permute(1,0)
c_tilde1_p = c_tilde1.to_dense().to('cpu')
a_3 = torch.tensordot(c_tilde3_p,atomic_basis.to('cpu'),dims=1).squeeze()
a_2 = torch.tensordot(c_tilde2_p + a_3,atomic_basis.to('cpu'),dims=1)
a_1 = torch.tensordot(c_tilde1_p + a_2.permute(1,0),atomic_basis.to('cpu'),dims=1)

torch.tensordot(c_tilde2_p,dims=0)

