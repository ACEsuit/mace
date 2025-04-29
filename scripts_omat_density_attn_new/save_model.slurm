#!/bin/bash
#SBATCH --job-name=train-mace            # job name
#SBATCH --account=gax@h100           # account
#SBATCH --partition=gpu_p6s
#SBATCH -C h100                      # target H100 nodes
# Here, reservation of 3x24=72 CPUs (for 3 tasks) and 3 GPUs (1 GPU per task) on one node only:
#SBATCH --nodes=1                    # number of node
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node (here = number of GPUs per node)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 4 for H100 nodes)
# Knowing that here we only reserve one GPU per task (i.e. 1/4 of GPUs),
# the ideal is to reserve 1/4 of CPUs for each task:
#SBATCH --cpus-per-task=2            # number of CPUs per task (here 1/4 of the node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading deactivated
#SBATCH --time=1:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=mace-train-%A_%a.out # name of output file
#SBATCH --error=mace-train-%A_%a.out  # name of error file (here, in common with the output file)
#SBATCH --exclude=jzxh063
##SBATCH --array=0-3%1                  # Array index range

# Access arguments
bs=32
lr=0.005
gpu=16
conf=jz_omat_r6.0_no_rattle_lmdb.yaml
r=6.0
num_channel=128 # [[64], -128-, [256]]
mlp_irreps="16x0e" # [["8x0e"], -"16x0e"-, ["32x0e"], ["64x0e"]]
num_radial=10 # [6, [8], -10-, [12]]
seed=123
stress=0.0
#interaction_first="RealAgnosticDensityInjuctedNoScaleInteractionBlock"
#interaction="RealAgnosticDensityInjuctedNoScaleResidualInteractionBlock"

#interaction_first="RASimpleDensityMultiAttnIntBlock"
#interaction="RASimpleDensityMultiAttnResidualIntBlock"

interaction_first="RASimpleDensityMultiSCAttnIntBlock"
interaction="RASimpleDensityMultiSCAttnResidualIntBlock"

#interaction_first="RASimpleDensityCutoffMultiSCAttnIntBlock"
#interaction="RASimpleDensityCutoffMultiSCAttnResidualIntBlock"

#interaction_first="RASimpleDensityAttnIntBlock"
#interaction="RASimpleDensityAttnResidualIntBlock"

#interaction_first="RealAgnosticInteractionBlock"
#interaction="RealAgnosticResidualInteractionBlock"

agnostic_first=False
num_interactions=2
max_L=1
max_ell=3
ckpt_path=35
contraction="EquivariantProductBasisDensityAttnBlock"
#contraction="EquivariantProductBasisBlock"

# Cleans out modules loaded in interactive and inherited by default
module purge
 
source /lustre/fsn1/worksf/projects/rech/gax/unh55hx/miniconda3/etc/profile.d/conda.sh

conda activate mace-multihead
 
# Running code
#srun bash run_multihead_5arg.sh ${bs} ${lr} ${gpu} ${conf} ${r} ${num_channel} ${num_radial} ${mlp_irreps} ${seed} ${stress} ${interaction_first} ${interaction} ${num_interactions} ${agnostic_first} ${max_L} ${max_ell}
echo "srun bash run_multihead_5arg.sh ${bs} ${lr} ${gpu} ${conf} ${r} ${num_channel} ${num_radial} ${mlp_irreps} ${seed} ${stress} ${interaction_first} ${interaction} ${num_interactions} ${agnostic_first} ${max_L} ${max_ell}"
# Running code
srun bash save_model.sh ${bs} ${lr} ${gpu} ${conf} ${r} ${num_channel} ${num_radial} ${mlp_irreps} ${seed} ${stress} ${interaction_first} ${interaction} ${num_interactions} ${agnostic_first} ${ckpt_path} ${max_L} ${max_ell} ${contraction}

