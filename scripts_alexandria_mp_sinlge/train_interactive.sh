bs=32
lr=0.005
gpu=3
conf=jz_alex_config_r6.0_all.yaml
r=6.0
num_channel=128 # [[64], -128-, [256]]
mlp_irreps="16x0e" # [["8x0e"], -"16x0e"-, ["32x0e"], ["64x0e"]]
num_radial=10 # [6, [8], -10-, [12]]
seed=123
stress=0.0
interaction_first="RealAgnosticDensityInjuctedNoScaleInteractionBlock"
interaction="RealAgnosticDensityInjuctedNoScaleResidualInteractionBlock"
agnostic_first=False
num_interactions=2

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
bash run_multihead_5arg.sh ${bs} ${lr} ${gpu} ${conf} ${r} ${num_channel} ${num_radial} ${mlp_irreps} ${seed} ${stress} ${interaction_first} ${interaction} ${num_interactions} ${agnostic_first}

