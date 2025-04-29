bs=32
lr=0.005
gpu=32
conf=jz_omat_alex_mp_refit_reduced_r6.0_lmdb.yaml
r=6.0
num_channel=128 # [[64], -128-, [256]]
mlp_irreps="16x0e" # [["8x0e"], -"16x0e"-, ["32x0e"], ["64x0e"]]
num_radial=10 # [6, [8], -10-, [12]]
seed=123
stress=0.0
#interaction_first="RealAgnosticDensityInteractionBlock"
#interaction="RealAgnosticDensityResidualInteractionBlock"
interaction_first="RASimpleDensityIntBlock"
interaction="RASimpleDensityResidualIntBlock"
agnostic_first=False
num_interactions=2
max_L=1
max_ell=3

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
bash run_multihead_5arg_ft.sh ${bs} ${lr} ${gpu} ${conf} ${r} ${num_channel} ${num_radial} ${mlp_irreps} ${seed} ${stress} ${interaction_first} ${interaction} ${num_interactions} ${agnostic_first} ${max_L} ${max_ell}

