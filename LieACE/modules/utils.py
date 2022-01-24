
from collections import OrderedDict
from typing import Tuple,Dict,List

import numpy as np
import torch
import torch.nn
import torch.utils.data
from torch_scatter import scatter_sum
from torch_sparse import spmm

from ace_torch.functions.sparse_tensor_dot import tensor_contract_nd_update_sparse,unfold,fold,expand_element,vector_contract
from ace_torch.functions.cg import create_U,Rot3DCoeffs #if we precompute them, then load here whatever we precomputed





def create_U_element(node_deg,
                     n_elements,
                     i,
                     A,
                     device): 
        """ Create the U matrix of the element specified by the one hot
        :data a data structure of torch geometric with a field called degree storing degree classes for each atom.
        This field should be of a tuple of size number of elements in the molecule.
        :one_hot a liste of int corresponding to the one hot of the element
        """
        vu = node_deg[i].max_corr() #max correlation of the element
        c_tildes_dict = {}
        c_tildes_dict['degree'] = node_deg[i]
        for l in range(1,vu+1):
            U,index_mm = create_U(A,l,node_deg[i]) #create the U matrix of the speficied element different degrees for different correlation
            U = expand_element(U.coalesce(),n_elements) #make copies to match the number of elements
            c_tildes_dict[l] =  unfold(U,0).to(device) #unfold U in the write dimension
        return c_tildes_dict


class c_tildes_weight(torch.nn.Module):
    def __init__(self,
                 c_tildes_dict,
                 num_elements : int,
                 device = 'cpu'):
        super().__init__()
        
        self.device = device
        self.max_corr = c_tildes_dict['degree'].max_corr() 
        nmax = c_tildes_dict['degree'].max_n()
        lmax = c_tildes_dict['degree'].max_l()
        self.size = {}
        self.weights = torch.nn.ParameterDict()
        weight = torch.empty((c_tildes_dict[1].size()[0],1),device=self.device)
        weight = torch.nn.init.uniform_(weight, a=-1.0, b=1.0)
        self.weights[str(1)] = torch.nn.Parameter(weight)
        for vu in range(2,self.max_corr + 1):
            weight = torch.empty((c_tildes_dict[vu].size()[-1],1),device=self.device)
            weight = torch.nn.init.uniform_(weight, a=-1.0, b=1.0)
            self.weights[str(vu)] = torch.nn.Parameter(weight) #creates weights
            self.size[vu] = tuple((((nmax + 1) * (lmax + 1)**2)*num_elements for i in range(vu)))

    def forward(self,
                c_tildes_dict):
        c_tildes_dict_w = {} #initialize the weighted dict of c_tildes
        c_tildes_dict_w['degree'] = c_tildes_dict['degree'] #pass down the dgree argument
        c_tildes_dict_w[1] = spmm(c_tildes_dict[1].transpose(0,1)._indices(),c_tildes_dict[1]._values(),
                                              c_tildes_dict[1].size()[1],c_tildes_dict[1].size()[0],
                                              self.weights[str(1)]).to_sparse()
        for vu in range(2,self.max_corr + 1):
            c_tildes_dict_w[vu] = spmm(c_tildes_dict[vu]._indices(),c_tildes_dict[vu]._values(),
                                              c_tildes_dict[vu].size()[0],c_tildes_dict[vu].size()[1],
                                              self.weights[str(vu)]).to_sparse().coalesce()
            c_tildes_dict_w[vu] = unfold(fold(c_tildes_dict_w[vu],0,self.size[vu],device=self.device),0,device=self.device)
        return c_tildes_dict_w


def compute_forces(energy: torch.Tensor, positions: torch.Tensor, training=True) -> torch.Tensor:
    gradient = torch.autograd.grad(
        outputs=energy,  # [n_graphs, ]
        inputs=positions,  # [n_nodes, 3]
        grad_outputs=torch.ones_like(energy),
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        only_inputs=True,  # Diff only w.r.t. inputs
        allow_unused=False,
    )[0]  # [n_nodes, 3]

    return -1 * gradient

def get_edge_vectors_and_lengths(
        positions: torch.Tensor,  # [n_nodes, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
        shifts: torch.Tensor,  # [n_edges, 3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender, receiver = edge_index
    # From ase.neighborlist:
    # D = positions[j]-positions[i]+S.dot(cell)
    # where shifts = S.dot(cell)
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    return vectors, lengths



def compute_mean_std_atomic_inter_energy(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs)
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        avg_atom_inter_es_list.append((batch.energy - graph_e0s) / graph_sizes)  # {[n_graphs], }

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    std = to_numpy(torch.std(avg_atom_inter_es)).item()

    return mean, std


def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs)
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append((batch.energy - graph_e0s) / graph_sizes)  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()

    return mean, rms