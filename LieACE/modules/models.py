from typing import Dict, Any, Type, List
import torch
from .blocks import (LinearNodeEmbeddingBlock, NonLinearBlock, AtomicEnergiesBlock, RadialEmbeddingBlock,
                    EdgeEmbeddingBlock, AtomicBaseBlock,  VectorizeBlock)
from spherical_harmonics import SphericalHarmonics
import numpy as np



class ACE_layer(torch.nn.Module):
    def __init__(
        self,
        r_cut : float,
        degrees : List,
        num_polynomial_cutoff : int,
        num_elements : int,
        A : Dict[int,Any],#Rot3DCoeff
        non_linear = False,
        device = 'cpu',
    ):
        super().__init__() 
        
        #Embedding
        self.num_elements = num_elements
        self.node_embedding = LinearNodeEmbeddingBlock(num_in = num_elements, num_out = num_elements) #change to higher embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        self.sh = SphericalHarmonics(lmax=lmax)
        self.atomic_basis = AtomicBasis.atomic_base(num_elements=self.num_elements)
        
        
        
        self.c_tildes_dicts = { i : ProdBasis.create_U_element(
            node_deg = degrees,
            n_elements = self.num_elements,
            i = i,
            A = A,
            device = device) for i in range(self.num_elements)}
        
        
        self.weightings = torch.nn.ModuleList()
        self.contract = torch.nn.ModuleList()
        

        
        for i in range(self.num_elements):

            self.weightings.append(ProdBasis.c_tildes_weight(
                self.c_tildes_dicts[i],num_elements = num_elements,device=device))
            self.contract.append(ProdBasis.vectorize_basis(self.weightings[i](self.c_tildes_dicts[i]),
                                                           device=device))
        
        self.c_tildes_dicts_w = {} #init the weights c_tildes_dicts
        
        self.non_linear = non_linear
        
    def forward(self, data : AtomicData) -> Dict[str, Any]:
        

        
        data = self.atomic_basis(data)
        
        for elem in range(self.num_elements): 
            self.c_tildes_dicts_w[elem] = self.weightings[elem](self.c_tildes_dicts[elem])
           
        ace_features = torch.empty(data.num_nodes,1)
        
        for i in range(data.num_nodes): #TODO : multithread this loop
            element = int((data.node_attrs[i] == 1).nonzero(as_tuple=False).squeeze())
            max_n = data.node_degree[element].max_n() + 1 # #TODO : change that for different molecules in the same batch!!!!
            max_l = data.node_degree[element].max_l() ##TODO : change that for different molecules in the same batch!!!!
            max_lm = (max_l + 1)**2
            ace_features[i,:] = self.contract[element](
                [data.node_features[0][i][:,:max_n,:max_lm],data.node_features[1][i][:,:max_n,:max_lm]], #atomic basis real part and imag part
                self.c_tildes_dicts_w[element])[0] #the real part of the invariant ACE feature
        
        if self.non_linear : 
        
            data['ace_features'] = torch.tanh(ace_features)
            
        else : 
        
            data['ace_features'] = ace_features    
            
        return data