#!/usr/bin/env python
"""Multi-XPU batch MACE geometry optimization on Aurora (12 tiles).

Each MPI rank is pinned to one XPU tile via ZE_AFFINITY_MASK
(handled by mpiexec). Each rank sees its tile as xpu:0.

Usage:
    mpiexec -n 12 --ppn 12 --cpu-bind list:... \\
        python run_mace_xpu_batch.py structures.xyz [model_name] [fmax]

    model_name options: small, medium (default), large, medium-mpa-0, ...
    fmax: force convergence criterion in eV/A (default: 0.0001)
"""
import os
import sys

# Force unbuffered stdout so output appears immediately under MPI
os.environ["PYTHONUNBUFFERED"] = "1"

# Pin each MPI rank to its own XPU tile BEFORE importing torch.
# PALS_LOCAL_RANKID is set by PBS/PALS on Aurora.
# Without this, all ranks compete on the same default tile.
local_rank = int(os.environ.get("PALS_LOCAL_RANKID", os.environ.get("PMI_LOCAL_RANK", "0")))
print(local_rank)
os.environ["ZE_AFFINITY_MASK"] = str(local_rank)

# Set cache dir BEFORE importing mace (foundations_models.py:122 evaluates at import)
CACHE_DIR = os.environ.get(
    "MACE_CACHE_DIR",
    "/lus/flare/projects/ChemGraph/thang/mace_models",
)
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

from mpi4py import MPI  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.io import read  # noqa: E402
from ase.io.extxyz import write_extxyz  # noqa: E402
from ase.optimize import BFGS  # noqa: E402
from mace.calculators import mace_mp  # noqa: E402

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def log(msg):
    print(msg, flush=True)


def main():
    if len(sys.argv) < 2:
        if rank == 0:
            log(f"Usage: mpiexec -n N {sys.argv[0]} <structures.xyz> [model_name] [fmax]")
        sys.exit(1)

    xyz_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "medium"
    fmax = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

    log(f"[Rank {rank}/{size}] ZE_AFFINITY_MASK={local_rank}, reading {xyz_path}")

    # All ranks read structures (small files — simpler than bcast)
    frames = read(xyz_path, index=":")
    if rank == 0:
        log(f"Loaded {len(frames)} structures, distributing across {size} XPU tiles")
        log(f"Model: {model_name}, Cache: {CACHE_DIR}/mace/")

    # Split work: if only 1 structure, all ranks compute it (verification mode)
    if len(frames) == 1:
        if rank == 0:
            log("Single structure: all ranks will compute it (verification mode)")
        my_frames = [(0, frames[0].copy())]
    else:
        my_frames = [(idx, frames[idx]) for idx in range(rank, len(frames), size)]

    # Each rank creates calculator on its own tile (xpu:0 via ZE_AFFINITY_MASK)
    log(f"[Rank {rank}] Loading model '{model_name}' on xpu")
    calc = mace_mp(model=model_name, device="xpu", default_dtype="float32", dispersion=False)
    log(f"[Rank {rank}] Model loaded, optimizing {len(my_frames)} structures (fmax={fmax})")

    # Geometry optimization
    out_prefix = os.path.splitext(os.path.basename(xyz_path))[0]
    my_results = []
    for idx, atoms in my_frames:
        atoms.calc = calc
        traj_path = f"{out_prefix}_rank{rank}_struct{idx}.traj"

        log(f"[Rank {rank}] struct {idx}: starting optimization ({len(atoms)} atoms)")
        dyn = BFGS(atoms, trajectory=traj_path)
        dyn.run(fmax=fmax)

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        max_force = abs(forces).max()

        # Create a clean Atoms copy for gathering (no calc, no XPU tensors)
        clean_atoms = Atoms(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
        )

        log(
            f"[Rank {rank}] struct {idx}: converged in {dyn.nsteps} steps  "
            f"E={energy:.6f} eV  E/atom={energy/len(atoms):.6f}  "
            f"max|F|={max_force:.6f} eV/A"
        )
        my_results.append({
            "idx": idx,
            "rank": rank,
            "natoms": len(atoms),
            "energy": energy,
            "energy_per_atom": energy / len(atoms),
            "max_force": max_force,
            "steps": dyn.nsteps,
            "atoms": clean_atoms,
        })

    if not my_frames:
        log(f"[Rank {rank}] No structures assigned (fewer structures than ranks)")

    log(f"[Rank {rank}] Done, gathering results")

    # Gather to rank 0
    all_results = comm.gather(my_results, root=0)

    if rank == 0:
        flat = sorted(
            [r for sublist in all_results for r in sublist],
            key=lambda x: x["idx"],
        )

        # Write all optimized structures (rank 0 does all I/O)
        # Use write_extxyz with plain open() to bypass ASE's MPI-aware paropen
        for r in flat:
            out_path = f"{out_prefix}_rank{r['rank']}_struct{r['idx']}_opt.xyz"
            with open(out_path, "w") as f:
                write_extxyz(f, [r["atoms"]])
            log(f"Wrote: {out_path}")

        log(f"\n--- Summary ({len(flat)} structures across {size} ranks, fmax={fmax}) ---")
        log(f"{'idx':>5} {'atoms':>6} {'steps':>6} {'energy (eV)':>14} {'E/atom (eV)':>14} {'max|F| (eV/A)':>14}")
        log("-" * 66)
        for r in flat:
            log(
                f"{r['idx']:5d} {r['natoms']:6d} {r['steps']:6d} {r['energy']:14.6f} "
                f"{r['energy_per_atom']:14.6f} {r['max_force']:14.6f}"
            )


if __name__ == "__main__":
    main()
