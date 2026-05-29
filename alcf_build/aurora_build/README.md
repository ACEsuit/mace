# MACE on Aurora (Intel XPU)

Scripts for running MACE geometry optimizations on ALCF Aurora using Intel Data Center Max GPUs (XPU).

## Setup

Aurora's `frameworks` module provides Intel's XPU-enabled PyTorch (IPEX). MACE's `setup.cfg` declares `torch>=1.12` which causes `pip install` to pull the upstream CUDA torch from PyPI, clobbering the Intel build. The setup script avoids this:

```bash
source alcf_build/aurora_build/setup_env.sh
```

This will:
1. Load the `frameworks` module (PyTorch + IPEX + oneCCL)
2. Create a venv with `--system-site-packages` (inherits Intel torch)
3. Install MACE with `pip install --no-deps -e .` (skips PyPI torch)
4. Install all non-torch dependencies separately

### Model cache

MACE downloads foundation models to `$XDG_CACHE_HOME/mace/`. By default this points to `~/.cache/mace/`, which is problematic on HPC (small home dirs, compute nodes without internet).

Set `MACE_CACHE_DIR` to control the location:

```bash
export MACE_CACHE_DIR=/lus/flare/projects/ChemGraph/thang/mace_models
```

Pre-download models on a login node before submitting jobs:

```bash
export XDG_CACHE_HOME=$MACE_CACHE_DIR
python -c "from mace.calculators import mace_mp; mace_mp(model='medium', device='cpu')"
```

**Note:** `XDG_CACHE_HOME` must be set before importing MACE because `foundations_models.py` evaluates the cache path at module import time.

## Single-tile geometry optimization

```bash
python alcf_build/aurora_build/run_mace_xpu.py structure.xyz [model_name] [fmax]
```

- `model_name`: `small`, `medium` (default), `large`, `medium-mpa-0`, etc.
- `fmax`: force convergence in eV/A (default: `0.01`)
- Output: `{name}_opt.traj` (trajectory) and `{name}_optimized.xyz`

## Multi-tile batch geometry optimization

Distributes structures across all 12 XPU tiles (6 GPUs x 2 tiles) using MPI:

```bash
mpiexec -n 12 --ppn 12 \
    --cpu-bind list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79 \
    python alcf_build/aurora_build/run_mace_xpu_batch.py structures.xyz [model_name] [fmax]
```

Each MPI rank is pinned to a separate tile via `ZE_AFFINITY_MASK` (set automatically from `PALS_LOCAL_RANKID`). Structures are distributed round-robin across ranks.

If a single structure is provided, all ranks compute it (verification mode) to confirm all tiles are working.

### PBS job submission

```bash
qsub alcf_build/aurora_build/submit_batch.sh -- structures.xyz medium 0.01
```

Edit `submit_batch.sh` to set your project allocation (`-A`).

## Files

| File | Description |
|------|-------------|
| `setup_env.sh` | Environment setup (venv + deps without PyPI torch) |
| `run_mace_xpu.py` | Single-tile geometry optimization |
| `run_mace_xpu_batch.py` | Multi-tile batch geometry optimization (MPI) |
| `submit_batch.sh` | PBS job script with CPU binding for Aurora |

## Known issues

- **ASE MPI-aware I/O**: When `mpi4py` is imported, ASE restricts file writes to rank 0. The batch script gathers all optimized structures to rank 0 and writes using `write_extxyz` with Python's built-in `open()` to bypass this.
- **IPEX `ipex.optimize()` bug**: In `mace/calculators/mace.py`, the original code discarded the return value of `ipex.optimize()`. Fixed on this branch to write back to `self.models[i]` and call `model.eval()` first (IPEX requires eval mode for inference without an optimizer).
- **XPU not auto-detected**: MACE's device defaulting (`foundations_models.py:286`) only checks for CUDA. You must pass `device="xpu"` explicitly.
- **float32 non-determinism**: Energies may differ by ~5 uV/atom across tiles due to GPU floating-point reduction order. This is normal.
