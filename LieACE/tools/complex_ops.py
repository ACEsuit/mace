import torch
from torch_sparse import spmm

def complex_spmm(sparse_a,b):
    
        if len(sparse_a) == 2 and len(b) == 2 :
        
            real_part = spmm(
                    sparse_a[0].indices(),sparse_a[0].values(),
                    sparse_a[0].size()[0],sparse_a[0].size()[1],
                    b[0]) - spmm(
                    sparse_a[1].indices(),sparse_a[1].values(),
                    sparse_a[1].size()[0],sparse_a[1].size()[1],
                    b[1]) #r1*r2 - im1*im2
            
                        
            imag_part = spmm(
                    sparse_a[0].indices(),sparse_a[0].values(),
                    sparse_a[0].size()[0],sparse_a[0].size()[1],
                    b[1]) + spmm(
                    sparse_a[1].indices(),sparse_a[1].values(),
                    sparse_a[1].size()[0],sparse_a[1].size()[1],
                    b[0]) #r1*im2 + r2*im1 
            

            return (real_part,imag_part)            
        

        
        elif len(sparse_a) == 1 and len(b) == 2 : 

            real_part = spmm(
                    sparse_a[0].indices(),sparse_a[0].values(),
                    sparse_a[0].size()[0],sparse_a[0].size()[1],
                    b[0]) #r1*r2
            
            imag_part = spmm(
                    sparse_a[0].indices(),sparse_a[0].values(),
                    sparse_a[0].size()[0],sparse_a[0].size()[1],
                    b[1]) #r2*im1 
            
            return (real_part,imag_part)
            