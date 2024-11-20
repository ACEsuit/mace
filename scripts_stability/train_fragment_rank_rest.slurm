#!/bin/bash
#SBATCH --job-name=train-mace            # job name
#SBATCH --account=gax@h100           # account
#SBATCH -C h100                      # target H100 nodes
# Here, reservation of 3x24=72 CPUs (for 3 tasks) and 3 GPUs (1 GPU per task) on one node only:
#SBATCH --nodes=1                     # number of node
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
#####SBATCH --array=0-3%1                  # Array index range

# Access arguments
bs=32
lr=0.005
gpu=1
conf=jz_mp_config_r6.0.yaml
r=6.0
num_channel=128 # [[64], -128-, [256]]
mlp_irreps="16x0e" # [["8x0e"], -"16x0e"-, ["32x0e"], ["64x0e"]]
num_radial=10 # [6, [8], -10-, [12]]
seed=123
stress=100
eps=0.00000001
optim=adam
betas="(0.9,0.999)"

# Cleans out modules loaded in interactive and inherited by default
module purge
 
# Loading modules
module load pytorch-gpu/py3/2.3.1
 
# Echo of launched commands
set -x 

# set path
export PATH="$PATH:/linkhome/rech/genrre01/unh55hx/.local/bin"
DATA_DIR=/lustre/fsn1/projects/rech/gax/unh55hx/data/multihead_dataset

sync_file="slurm_sync_cache_${SLURM_ARRAY_TASK_ID}"

# Set the default master port and world size
MASTER_PORT=17763           # Set an available port
WORLD_SIZE=${gpu}                # Total number of processes (master + workers)

# async start

# Function to update the sync file count and get the next rank
get_next_rank_and_update_sync_file() {
    # Extract the current count from the sync file (second line)
    count=$(tail -n 1 "$sync_file")
    next_rank=$count
    count=$((count + 1))
    
    # Overwrite the sync file with the updated count
    sed -i "2s/.*/$count/" "$sync_file"
    
    # Return the next rank
    echo $next_rank
    sync
}


## sleep random miliseconds
#sync
#b=$(shuf -i 0-1000 -n 1)
#sleep 0.$(printf "%03d" $b)
#sync

exec 200>"/tmp/will_lockfile.lock"
flock -x 200

# Check if sync_file exists
if [ -f "$sync_file" ]; then
    # sync_file exists, read the MASTER_ADDR from the file
    MASTER_ADDR=$(head -n 1 "$sync_file")  # Read the first line (MASTER_ADDR)
    echo "Reading MASTER_ADDR from sync file: $MASTER_ADDR"

    # Get the next rank from the sync file and update the count
    RANK=$(get_next_rank_and_update_sync_file)
else
    # sync_file does not exist, this is the master node
    MASTER_ADDR=$(hostname -i)
    echo "This is the master node (RANK=0) with IP: $MASTER_ADDR"
    
    # Initialize the sync file with MASTER_ADDR and the count
    echo "$MASTER_ADDR" > "$sync_file"
    echo "1" >> "$sync_file"  # Start the count at 1 (master process counts as 1)
    
    RANK=0                         # This is the master node
    # Sync the file system to ensure changes are written to disk
    sync
fi

flock -u 200

# Export environment variables for PyTorch distributed training
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE
export RANK=$RANK

export NCCL_DEBUG=INFO

# Running code
srun bash run_multihead_5arg.sh ${bs} ${lr} ${gpu} ${conf} ${r} ${num_channel} ${num_radial} ${mlp_irreps} ${seed} ${stress} ${eps} ${optim} ${betas}

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
