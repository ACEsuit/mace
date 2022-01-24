import torch
from collections import OrderedDict
from ace_torch.functions.sparse_tensor_dot import tensor_contract_nd_update_sparse,unfold,fold,expand_element,vector_contract
from typing import Dict,List
from ace_torch.functions.cg import create_U,Rot3DCoeffs #if we precompute them, then load here whatever we precomputed
from torch_sparse import spmm

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