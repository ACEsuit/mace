#!/bin/bash
#SBATCH --job-name=train-mace            # job name
#SBATCH --account=gax@h100           # account
#SBATCH --partition=gpu_p6
#SBATCH --exclude=jzxh150
#SBATCH -C h100                      # target H100 nodes
#SBATCH --nodes=2                   # number of node 16
#SBATCH --ntasks-per-node=4         # number of MPI tasks per node (here = number of GPUs per node)
#SBATCH --gres=gpu:4                 # number of GPUs per node (max 4 for H100 nodes)
#SBATCH --cpus-per-task=16            # number of CPUs per task (here 1/4 of the node)
#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=mace-train-%A_%a.out # name of output file
#SBATCH --error=mace-train-%A_%a.out  # name of error file (here, in common with the output file)
##SBATCH --array=0-1%1                  # Array index range


##SBATCH --hint=nomultithread         # hyperthreading deactivated

# Access arguments
bs=16
lr=0.005
gpu=8
conf=jz_mp_config_r6.0.yaml
#conf=jz_mp_and_salex_stableonly_r6.0_lmdb.yaml
#conf=jz_mp_and_salex_stable-first_r6.0_lmdb.yaml
r=6.0
num_channel=128 # [[64], -128-, [256]]
mlp_irreps="16x0e" # [["8x0e"], -"16x0e"-, ["32x0e"], ["64x0e"]]
num_radial=10 # [6, [8], -10-, [12]]
seed=123
stress=0.0
#interaction_first="RealAgnosticDensityInjuctedNoScaleInteractionBlock"
#interaction="RealAgnosticDensityInjuctedNoScaleResidualInteractionBlock"

#interaction_first="RASimpleDensityMultiSCAttnIntBlock"
#interaction="RASimpleDensityMultiSCAttnResidualIntBlock"

interaction_first="RASimpleDensityMultiSCAttnIntBlock"
interaction="RASimpleDensityMultiSCAttnResidualIntBlock"

#interaction_first="RASimpleDensityAttnIntBlock"
#interaction="RASimpleDensityAttnResidualIntBlock"

#interaction_first="RealAgnosticInteractionBlock"
#interaction="RealAgnosticResidualInteractionBlock"
agnostic_first=False
num_interactions=2
max_L=1
max_ell=3

contraction="EquivariantProductBasisDensityAttnBlock"
#contraction="EquivariantProductBasisBlock"

# Cleans out modules loaded in interactive and inherited by default
module purge
 
source /lustre/fsn1/worksf/projects/rech/gax/unh55hx/miniconda3/etc/profile.d/conda.sh

conda activate mace-multihead
 

# Running code
bash run_multihead_5arg.sh ${bs} ${lr} ${gpu} ${conf} ${r} ${num_channel} ${num_radial} ${mlp_irreps} ${seed} ${stress} ${interaction_first} ${interaction} ${num_interactions} ${agnostic_first} ${max_L} ${max_ell} ${contraction}

