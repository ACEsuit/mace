#!/bin/bash -l
# Multi-GPU (real nccl) regression test for the get_symmetric_displacement fast path
# (the "num_graphs == 1" branch) on MPCDF Raven A100 nodes.
#
# Locally we verified the branch with: forward equivalence (CPU+CUDA), a batch_size=1
# training run (CPU+CUDA), and a gloo/CPU DDP run. The one thing a single local GPU cannot
# cover is REAL multi-GPU/nccl, which this job provides: a 4-way DistributedDataParallel
# stress-training run with batch_size=1 per rank, so every rank takes the new
# "num_graphs == 1" path on a genuine A100 + nccl all-reduce.
#
# Layout mirrors the existing Raven profiling jobscript:
#   - batch jobs have NO internet: the conda/venv + repo must be built on a login node,
#     and the dummy dataset is generated in-job (needs only ase+numpy, no download).
#   - all output goes under /ptmp, never into the repo.
#
# Submit:   sbatch scratch/stress_test/raven_stress_test.sbatch
#
#SBATCH -o /ptmp/ademo/isambard/arndm/results/logs/stress_test_%j.out   # TODO: your /ptmp path
#SBATCH -e /ptmp/ademo/isambard/arndm/results/logs/stress_test_%j.err   # TODO: your /ptmp path
#SBATCH -D ./
#SBATCH -J mace_stress_test
#
#SBATCH --constraint="gpu"
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=18
#SBATCH --mem=0
#
#SBATCH --mail-type=none
#SBATCH --time=0:30:00

set -euo pipefail

# --- Environment (built beforehand on a login node; no internet in batch jobs) ---
MACE_REPO=${MACE_REPO:-~/mace/mace_parallel_work/test_stress_opt}          # TODO: path to THIS branch's clone on Raven
RESULTS_BASE=${RESULTS_BASE:-/ptmp/ademo/isambard/arndm/results}   # TODO: your /ptmp path
DATA_DIR="$RESULTS_BASE/stress_test/data"
WORK_DIR="$RESULTS_BASE/stress_test/run_${SLURM_JOB_ID}"

source "$MACE_REPO/activate_env.sh"                 # TODO: must `conda activate mace` / load venv
cd "$MACE_REPO"

# Repo is not installed editable in the env -> make sure THIS branch is imported.
export PYTHONPATH="$MACE_REPO:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1           # e3nn 0.4.4 + torch>=2.6 torch.load
export OMP_NUM_THREADS=1

mkdir -p "$DATA_DIR" "$WORK_DIR" "$RESULTS_BASE/logs"

echo "$(date): job $SLURM_JOB_ID on $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"

# --- 1. Quick single-task sanity: the forward + (gloo) DDP unit tests on this node ---
srun --ntasks=1 --gres=gpu:a100:1 python -m pytest -q -s \
    tests/cl-generated/test_stress_single_graph.py \
    tests/cl-generated/test_distributed_stress.py

# --- 2. Generate dummy data (single task, no internet needed) ---
srun --ntasks=1 python scratch/stress_test/make_dummy_data.py \
    --out_dir "$DATA_DIR" --n_train 40 --n_valid 10

# --- 3. Real 4-GPU nccl DDP stress training, batch_size=1 -> every rank hits the
#        num_graphs == 1 fast path on a real A100 + nccl all-reduce. ---
srun python scripts/run_train.py \
    --name='stress_fastpath' \
    --model='MACE' \
    --num_interactions=2 \
    --num_channels=32 \
    --max_L=1 \
    --correlation=3 \
    --r_max=4.0 \
    --E0s='average' \
    --train_file="$DATA_DIR/train.xyz" \
    --valid_file="$DATA_DIR/valid.xyz" \
    --energy_key='REF_energy' \
    --forces_key='REF_forces' \
    --stress_key='REF_stress' \
    --loss='stress' \
    --energy_weight=1.0 \
    --forces_weight=10.0 \
    --stress_weight=1.0 \
    --batch_size=1 \
    --valid_batch_size=1 \
    --max_num_epochs=4 \
    --eval_interval=1 \
    --default_dtype='float32' \
    --model_dir="$WORK_DIR" \
    --checkpoints_dir="$WORK_DIR" \
    --results_dir="$WORK_DIR" \
    --log_dir="$WORK_DIR" \
    --device='cuda' \
    --distributed \
    --launcher slurm \
    --seed=2222

echo "$(date): job $SLURM_JOB_ID finished OK"
