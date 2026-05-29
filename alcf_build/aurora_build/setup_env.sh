#!/bin/bash
# Aurora (Intel XPU) environment setup for MACE
#
# Creates a venv that inherits Intel's XPU-enabled PyTorch from the
# frameworks module, then installs MACE without pulling PyPI torch.
#
# Usage:
#   source alcf_build/aurora_build/setup_env.sh
#
# Why --no-deps?
#   setup.cfg declares torch>=1.12. Intel's XPU PyTorch is a custom build
#   that pip doesn't recognize, so pip downloads upstream (CUDA) torch.
#   --no-deps skips all dependency resolution; we install the rest manually.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 1. Load Intel frameworks (PyTorch + IPEX + oneCCL for XPU)
module load frameworks

# 2. Create venv inheriting system torch
python3 -m venv "$REPO_ROOT/.venv" --system-site-packages

# 3. Activate
source "$REPO_ROOT/.venv/bin/activate"

# 4. Install mace WITHOUT pulling torch from PyPI
pip install --no-deps -e "$REPO_ROOT"

# 5. Install non-torch dependencies
pip install \
    "e3nn==0.4.4" \
    numpy \
    opt_einsum \
    ase \
    torch-ema \
    prettytable \
    matscipy \
    h5py \
    torchmetrics \
    python-hostlist \
    configargparse \
    GitPython \
    pyYAML \
    tqdm \
    lmdb \
    orjson \
    matplotlib \
    pandas

# 6. Set model cache location
export MACE_CACHE_DIR="${MACE_CACHE_DIR:-/lus/flare/projects/ChemGraph/thang/mace_models}"
export XDG_CACHE_HOME="$MACE_CACHE_DIR"

echo ""
echo "MACE environment ready (Aurora XPU)"
echo "  venv:      $REPO_ROOT/.venv"
echo "  torch:     $(python -c 'import torch; print(torch.__version__)')"
echo "  XPU avail: $(python -c 'import torch; print(torch.xpu.is_available())')"
echo "  cache:     $MACE_CACHE_DIR/mace/"
