#!/usr/bin/env python
"""MACE geometry optimization on Intel XPU (Aurora HPC).

Aurora setup:
    module load frameworks   # provides PyTorch + IPEX + oneCCL

    # Pre-download model on login node (compute nodes may lack internet):
    export MACE_CACHE_DIR=/lus/flare/projects/ChemGraph/thang/mace_models
    python -c "import os; os.environ['XDG_CACHE_HOME'] = os.environ['MACE_CACHE_DIR']; from mace.calculators import mace_mp; mace_mp(model='medium', device='cpu')"

Usage:
    python run_mace_xpu.py structure.xyz [model_name] [fmax]

    model_name options: small, medium (default), large, medium-mpa-0, ...
    fmax: force convergence criterion in eV/A (default: 0.0001)
"""
import os
import sys

# ---- Set cache dir BEFORE importing mace ----
CACHE_DIR = os.environ.get(
    "MACE_CACHE_DIR",
    "/lus/flare/projects/ChemGraph/thang/mace_models",
)
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

# ---- Imports (safe now that env is configured) ----
from ase.io import read, write  # noqa: E402
from ase.optimize import BFGS  # noqa: E402
from mace.calculators import mace_mp  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <structure.xyz> [model_name] [fmax]")
        sys.exit(1)

    xyz_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "medium"
    fmax = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

    # Load structure
    atoms = read(xyz_path)
    print(f"Loaded {len(atoms)} atoms from {xyz_path}")
    print(f"Cache directory: {CACHE_DIR}/mace/")

    # Create MACE calculator on XPU
    calc = mace_mp(
        model=model_name,
        device="xpu",
        default_dtype="float32",
        dispersion=False,
    )
    atoms.calc = calc

    print(atoms)

    # Geometry optimization
    out_prefix = os.path.splitext(os.path.basename(xyz_path))[0]
    traj_path = f"{out_prefix}_opt.traj"
    out_path = f"{out_prefix}_optimized.xyz"

    dyn = BFGS(atoms, trajectory=traj_path)
    dyn.run(fmax=fmax)

    write(out_path, atoms)
    print(f"\nOptimization converged (fmax={fmax} eV/A)")
    print(f"Final energy:    {atoms.get_potential_energy():.6f} eV")
    print(f"Trajectory:      {traj_path}")
    print(f"Optimized geom:  {out_path}")


if __name__ == "__main__":
    main()
