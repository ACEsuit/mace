import time
from mace.calculators import mace_mp
import numpy as np
from ase.build import fcc111


calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",device='cuda', )#device='cpu')

initial = fcc111('Pt', size=(7, 8, 1), vacuum=10.0, orthogonal=True)
initial.calc = calc

s=time.time()
e0=initial.get_potential_energy()
h_autograd=calc.get_hessian(atoms=initial,method="loop")
print("h:",h_autograd)

e=time.time()
print(f"This system need {e-s} seconds")

s=time.time()
indicies =[i for i in range(len(initial))]
delta =1e-4
ndim = 3
hessian = np.zeros((len(indicies) * ndim, len(indicies) * ndim))
atoms_h = initial.copy()

atoms_h.set_constraint()

#finite difference approach
for i,index in enumerate(indicies):
    for j in range(ndim):
        # Perturb the position of atom i in dimension j
        atoms_i = atoms_h.copy()
        atoms_i.positions[index, j] += delta
        
        # Calculate forces for the perturbed configuration
        atoms_i.calc=calc
        forces_i = atoms_i.get_forces()

        # Same just in the other direction
        atoms_j = atoms_h.copy()
        atoms_j.positions[index, j] -= delta
        atoms_j.calc=calc
        forces_j = atoms_j.get_forces()
 
        # central difference
        hessian[:, i * ndim + j] = -(forces_i - forces_j)[indicies].flatten() / (2 * delta)

e=time.time()
hessian= hessian.reshape(-1,len(initial),3)

print(f"This system need {e-s} seconds")

is_close = np.allclose(h_autograd, hessian, atol=1e-6)
print("Are autograd and numerical close within tolerance of 1e-6? ", is_close)