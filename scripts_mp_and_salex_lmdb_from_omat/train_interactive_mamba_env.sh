bs=32
lr=0.005
gpu=3
conf=jz_mp_and_salex_r6.0_lmdb.yaml
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
max_L=1
max_ell=3

# Cleans out modules loaded in interactive and inherited by default
module purge

# mamba source and activate env

__conda_setup="$('/lustre/fsn1/projects/rech/gax/unh55hx/miniforge3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
	    eval "$__conda_setup"
    else
	        if [ -f "/lustre/fsn1/projects/rech/gax/unh55hx/miniforge3/etc/profile.d/conda.sh" ]; then
			        . "/lustre/fsn1/projects/rech/gax/unh55hx/miniforge3/etc/profile.d/conda.sh"
				    else
					            export PATH="/lustre/fsn1/projects/rech/gax/unh55hx/miniforge3/bin:$PATH"
						        fi
fi
unset __conda_setup

if [ -f "/lustre/fsn1/projects/rech/gax/unh55hx/miniforge3/etc/profile.d/mamba.sh" ]; then
	    . "/lustre/fsn1/projects/rech/gax/unh55hx/miniforge3/etc/profile.d/mamba.sh"
fi

mamba activate mace

# echo
set -x

# Running code
bash run_multihead_5arg_inter.sh ${bs} ${lr} ${gpu} ${conf} ${r} ${num_channel} ${num_radial} ${mlp_irreps} ${seed} ${stress} ${interaction_first} ${interaction} ${num_interactions} ${agnostic_first} ${max_L} ${max_ell}

