#!/bin/bash
#SBATCH --job-name=train-mace            # job name
#SBATCH --account=gax@h100           # account
#SBATCH --partition=gpu_p6
#SBATCH -C h100                      # target H100 nodes
#SBATCH --nodes=8                    # number of node
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node (here = number of GPUs per node)
#SBATCH --gres=gpu:4                 # number of GPUs per node (max 4 for H100 nodes)
#SBATCH --cpus-per-task=16            # number of CPUs per task (here 1/4 of the node)
#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=mace-train-%A_%a.out # name of output file
#SBATCH --error=mace-train-%A_%a.out  # name of error file (here, in common with the output file)
#SBATCH --array=0-3%1                  # Array index range


##SBATCH --hint=nomultithread         # hyperthreading deactivated

# Access arguments
bs=16
lr=0.005
gpu=32
conf=jz_mp_config_r6.0.yaml
r=6.0
num_channel=128 # [[64], -128-, [256]]
mlp_irreps="16x0e" # [["8x0e"], -"16x0e"-, ["32x0e"], ["64x0e"]]
num_radial=10 # [6, [8], -10-, [12]]
seed=123
stress=0.0
interaction_first="RASimpleDensityIntBlock"
interaction="RASimpleDensityResidualIntBlock"
agnostic_first=False
num_interactions=2
loss="omat24"
force_weight=20
energy_weight=20

# Cleans out modules loaded in interactive and inherited by default
module purge
 
module load arch/h100
# Loading modules
module load pytorch-gpu/py3/2.3.1
 
# Echo of launched commands
set -x 

# set path
export PATH="$PATH:/linkhome/rech/genrre01/unh55hx/.local/bin"
DATA_DIR=/lustre/fsn1/projects/rech/gax/unh55hx/data/multihead_dataset

# Running code
srun bash run_multihead_5arg.sh ${bs} ${lr} ${gpu} ${conf} ${r} ${num_channel} ${num_radial} ${mlp_irreps} ${seed} ${stress} ${interaction_first} ${interaction} ${num_interactions} ${agnostic_first} ${loss} ${force_weight} ${energy_weight}

