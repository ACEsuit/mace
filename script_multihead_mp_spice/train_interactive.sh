# Access arguments
bs=16
lr=0.005
gpu=32
conf=jz_spice_mp_config_r6.0.yaml
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

#interaction_first="RASimpleDensityMultiSCAttnIntBlock"
#interaction="RASimpleDensityMultiSCAttnResidualIntBlock"

#interaction_first="RASimpleDensityCutoffMultiSCAttnIntBlock"
#interaction="RASimpleDensityCutoffMultiSCAttnResidualIntBlock"

#interaction_first="RASimpleDensityAttnIntBlock"
#interaction="RASimpleDensityAttnResidualIntBlock"

#interaction_first="RealAgnosticInteractionBlock"
#interaction="RealAgnosticResidualInteractionBlock"

interaction_first="RASimpleDensityIntBlock"
interaction="RASimpleDensityResidualIntBlock"
agnostic_first=False
num_interactions=2
max_L=1
max_ell=3

#contraction="EquivariantProductBasisDensityAttnBlock"
contraction="EquivariantProductBasisBlock"

# Cleans out modules loaded in interactive and inherited by default
module purge

source /lustre/fsn1/worksf/projects/rech/gax/unh55hx/miniconda3/etc/profile.d/conda.sh

conda activate mace-multihead

# Running code

bash run_multihead_5arg_attn.sh ${bs} ${lr} ${gpu} ${conf} ${r} ${num_channel} ${num_radial} ${mlp_irreps} ${seed} ${stress} ${interaction_first} ${interaction} ${num_interactions} ${agnostic_first} ${max_L} ${max_ell} ${contraction}
