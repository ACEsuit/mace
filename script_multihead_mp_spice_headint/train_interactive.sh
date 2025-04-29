bs=32
lr=0.005
gpu=3
conf=jz_spice_mp_config_r6.0.yaml
r=6.0
num_channel=128 # [[64], -128-, [256]]
mlp_irreps="16x0e" # [["8x0e"], -"16x0e"-, ["32x0e"], ["64x0e"]]
num_radial=10 # [6, [8], -10-, [12]]
seed=123
stress=0.0
interaction_first="RASimpleDensityHeadIntBlock"
interaction="RASimpleDensityResidualHeadIntBlock"
agnostic_first=False
num_interactions=2
max_L=1
max_ell=3

# Cleans out modules loaded in interactive and inherited by default
module purge
module load arch/h100
# Loading modules
module load pytorch-gpu/py3/2.3.1
set -x

# Running code
bash run_multihead_5arg.sh ${bs} ${lr} ${gpu} ${conf} ${r} ${num_channel} ${num_radial} ${mlp_irreps} ${seed} ${stress} ${interaction_first} ${interaction} ${num_interactions} ${agnostic_first} ${max_L} ${max_ell}

