#!/bin/bash
module load pytorch-gpu/py3/2.3.1
conda activate mace-tn
export PYTHONPATH=${SCRATCH}/.conda/envs/mace-tn/lib/python3.11/site-packages/
ROOT_DIR=/lustre/fsn1/projects/rech/gax/unh55hx/tensornetwork/mace-main
cd $ROOT_DIR

echo $PYTHONPATH

python mace/cli/run_train.py \
    --name="MACE_3bpa" \
    --train_file="/lustre/fsn1/projects/rech/gax/unh55hx/tensornetwork/data/dataset_3BPA/train_300K.xyz" \
    --valid_fraction=0.1 \
    --test_file="/lustre/fsn1/projects/rech/gax/unh55hx/tensornetwork/data/dataset_3BPA/test_300K.xyz" \
    --energy_weight=27.0 \
    --forces_weight=1000.0 \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{1: -13.587222780835477, 6: -1029.4889999855063, 7: -1484.9814568572233, 8: -2041.9816003861047}' \
    --model="ScaleShiftMACE" \
    --interaction_first="RealAgnosticResidualInteractionBlock" \
    --interaction="RealAgnosticResidualInteractionBlock" \
    --num_interactions=2 \
    --max_ell=3 \
    --hidden_irreps='256x0e + 256x1o + 256x2e' \
    --num_cutoff_basis=5 \
    --correlation=3 \
    --r_max=5.0 \
    --scaling='rms_forces_scaling' \
    --batch_size=5 \
    --max_num_epochs=2000 \
    --patience=256 \
    --weight_decay=5e-7 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --default_dtype="float32"\
    --clip_grad=None \
    --device=cuda \
    --seed=123 \


# --statistics_file='./h5_data/statistics.json' \
