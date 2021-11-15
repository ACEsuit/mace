import torch

from Lie_groups.Linear_ops import kronsum


class Clebsch_Gordan():
    def __init__(self,Group,lmax : int, irreps) :
        self.Group = Group
        self.lmax = lmax
        self.irreps = irreps
        self.lie_generators = Group.lie_generators()
        self.discrete_generators = Group.discrete_generators()

    def constraint_matrix(self,l1,l2,l3) : 
        #Create the constraint matrix for CG(l1,l2,l3). The Nullspace of this constraint matrix corresponds to the 
        # vectorization of the CG coefficients. 

        self.irreps_l1 = self.irreps(l1)
        self.irreps_l2 = self.irreps(l2)
        self.irreps_l3 = self.irreps(l3)
        
        self.continous_A = [kronsum(self.irreps_l1(Ti), self.irreps_l2(Ti)) for Ti in self.lie_generators] #replace with lazy for saving cost
        self.discrete_A = [torch.kron(self.irreps_l1(hi), self.irreps_l2(hi)) for hi in self.discrete_generators]  #replace with lazy for cost
        self.A =  self.continous_A + self.discrete_A #replace with lazy for cost

        self.continous_B = [self.irreps_l3(Ti) for Ti in self.lie_generators]
        self.discrete_B = [self.irreps_l3(hi) for hi in self.discrete_generators]
        self.B = self.continous_B + self.discrete_B

        const_m = kronsum(self.A.T,-self.B)    

    
    def dict(self) : 
        

