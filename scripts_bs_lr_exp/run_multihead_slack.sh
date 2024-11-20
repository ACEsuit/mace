DATA_DIR=/lustre/fsn1/projects/rech/gax/unh55hx/data/multihead_dataset
module load pytorch-gpu/py3/2.3.1
export PATH="$PATH:/linkhome/rech/genrre01/unh55hx/.local/bin"
mace_run_train \
    --name="Test_Multihead_Agnesi" \
    --r_max=6.0 \
    --forces_weight=10 \
    --energy_weight=1 \
    --stress_weight=10 \
    --clip_grad=100 \
    --batch_size=32 \
    --valid_batch_size=128 \
    --num_workers=8 \
    --default_dtype="float64" \
    --seed=124 \
    --loss="universal" \
    --scheduler_patience=5 \
    --config="multihead_config/jz_spice_mp_config_r6.0.yaml" \
    --error_table='PerAtomMAE' \
    --model="MACE" \
    --interaction_first="RealAgnosticInteractionBlock" \
    --interaction="RealAgnosticResidualInteractionBlock" \
    --num_interactions=2 \
    --correlation=3 \
    --max_ell=3 \
    --max_L=1 \
    --num_channels=128 \
    --num_radial_basis=10 \
    --MLP_irreps="16x0e" \
    --scaling='rms_forces_scaling' \
    --lr=0.005 \
    --weight_decay=1e-8 \
    --ema \
    --ema_decay=0.995 \
    --pair_repulsion \
    --distance_transform="Agnesi" \
    --max_num_epochs=250 \
    --patience=40 \
    --amsgrad \
    --device=cuda \
    --clip_grad=100 \
    --keep_checkpoints \
    --restart_latest \
    --distributed \
    --save_cpu

