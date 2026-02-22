
import numpy as np
import torch
from ase.constraints import FixAtoms
from mace import modules, tools
from mace.calculators import MACECalculator
from e3nn import o3

torch.set_default_dtype(torch.float64)

def test_partial_hessian():
    # Setup simple model
    atomic_numbers = [1, 8]
    table = tools.AtomicNumberTable(atomic_numbers)
    atomic_energies = np.array([1.0, 3.0], dtype=float)
    
    model_config = dict(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=1,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=2,
        atomic_numbers=table.zs,
        correlation=2,
        radial_type="bessel",
    )
    model = modules.MACE(**model_config)
    
    # Create Calculator
    calc = MACECalculator(models=[model], device="cpu", default_dtype="float64")
    
    # Create Atoms
    from ase import Atoms
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    atoms.calc = calc
    
    # 1. Full Hessian
    h_full = calc.get_hessian(atoms)
    print("Full Hessian shape:", h_full.shape)
    
    # 2. Fix first atom (H at 0,0,0)
    c = FixAtoms(indices=[0])
    atoms.set_constraint(c)
    
    h_partial = calc.get_hessian(atoms)
    print("Partial Hessian shape:", h_partial.shape)
    
    assert h_partial.shape == h_full.shape
    
    # Verify rows for atom 0 are zero
    assert np.allclose(h_partial[0:3, :], 0.0)
    
    # Verify rows for atoms 1 and 2 match full hessian
    free_indices = [1, 2]
    for i in free_indices:
        # We compare slice [3*i : 3*i+3] (rows) and all columns
        row_start = 3 * i
        row_end = 3 * i + 3
        
        diff = np.abs(h_partial[row_start:row_end, :] - h_full[row_start:row_end, :])
        print(f"Max diff for atom {i}: {diff.max()}")
        assert np.allclose(h_partial[row_start:row_end, :], h_full[row_start:row_end, :], atol=1e-10)
        
    print("Partial Hessian matches Full Hessian for free atoms.")

    h2d, free_idx = calc.get_hessian_2d_free(atoms)
    free_idx = np.asarray(free_idx, dtype=int)
    assert np.array_equal(free_idx, np.array([1, 2], dtype=int))
    assert h2d.shape == (3 * len(free_idx), 3 * len(free_idx))

    m = h_full.shape[1]
    h4 = h_full.reshape(m, 3, m, 3)
    expected = h4[np.ix_(free_idx, [0, 1, 2], free_idx, [0, 1, 2])].reshape(
        3 * len(free_idx), 3 * len(free_idx)
    )
    assert np.allclose(h2d, expected, atol=1e-10)

if __name__ == "__main__":
    test_partial_hessian()
