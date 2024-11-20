#!/bin/bash
#SBATCH --job-name=train-mace            # job name
#SBATCH --account=gax@h100           # account
#SBATCH -C h100                      # target H100 nodes
# Here, reservation of 3x24=72 CPUs (for 3 tasks) and 3 GPUs (1 GPU per task) on one node only:
#SBATCH --nodes=1                    # number of node
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node (here = number of GPUs per node)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 4 for H100 nodes)
# Knowing that here we only reserve one GPU per task (i.e. 1/4 of GPUs),
# the ideal is to reserve 1/4 of CPUs for each task:
#SBATCH --cpus-per-task=12            # number of CPUs per task (here 1/4 of the node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading deactivated
#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=mace-train-%A_%a.out # name of output file
#SBATCH --error=mace-train-%A_%a.out  # name of error file (here, in common with the output file)
##SBATCH --array=0-3%1                  # Array index range

# Access arguments
bs=32
lr=0.005
gpu=16
conf=jz_mp_config_r6.0.yaml
r=6.0
num_channel=128 # [[64], -128-, [256]]
mlp_irreps="16x0e" # [["8x0e"], -"16x0e"-, ["32x0e"], ["64x0e"]]
num_radial=10 # [6, [8], -10-, [12]]
seed=123
stress=10
interaction_first="RealAgnosticDensityInjuctedNoScaleInteractionBlock"
interaction="RealAgnosticDensityInjuctedNoScaleResidualInteractionBlock"
agnostic_first=False
num_interactions=2
ckpt_path=212

# Cleans out modules loaded in interactive and inherited by default
module purge
 
# Loading modules
module load pytorch-gpu/py3/2.3.1
 
# Echo of launched commands
set -x 

# set path
export PATH="$PATH:/linkhome/rech/genrre01/unh55hx/.local/bin"
DATA_DIR=/lustre/fsn1/projects/rech/gax/unh55hx/data/multihead_dataset

# Running code
srun bash save_model.sh ${bs} ${lr} ${gpu} ${conf} ${r} ${num_channel} ${num_radial} ${mlp_irreps} ${seed} ${stress} ${interaction_first} ${interaction} ${num_interactions} ${agnostic_first} ${ckpt_path}

#mace_run_train \
#    --name="Test_Multihead_MultiGPU_SpiceMP_MACE" \
#    --model="MACE" \
#    --num_interactions=2 \
#    --num_channels=224 \
#    --max_L=0 \
#    --correlation=3 \
#    --r_max=5.0 \
#    --forces_weight=1000 \
#    --energy_weight=40 \
#    --weight_decay=5e-10 \
#    --clip_grad=1.0 \
#    --batch_size=32 \
#    --valid_batch_size=128 \
#    --max_num_epochs=210 \
#    --patience=50 \
#    --eval_interval=1 \
#    --ema \
#    --num_workers=8 \
#    --error_table='PerAtomMAE' \
#    --default_dtype="float64"\
#    --device=cuda \
#    --seed=123 \
#    --save_cpu \
#    --restart_latest \
#    --loss="weighted" \
#    --scheduler_patience=20 \
#    --lr=0.01 \
#    --swa \
#    --swa_lr=0.00025 \
#    --swa_forces_weight=100 \
#    --start_swa=190 \
#    --config="multihead_config/jz_spice_mp_config.yaml" \
#    --distributed \
