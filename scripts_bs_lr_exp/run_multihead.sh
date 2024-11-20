DATA_DIR=/lustre/fsn1/projects/rech/gax/unh55hx/data/multihead_dataset
module load pytorch-gpu/py3/2.3.1
export PATH="$PATH:/linkhome/rech/genrre01/unh55hx/.local/bin"
mace_run_train \
    --name="Test_Multihead_MultiGPU_SpiceMP_MACE" \
    --model="MACE" \
    --num_interactions=2 \
    --num_channels=224 \
    --max_L=2 \
    --correlation=3 \
    --r_max=5.0 \
    --forces_weight=1000 \
    --energy_weight=40 \
    --weight_decay=5e-10 \
    --clip_grad=1.0 \
    --batch_size=32 \
    --valid_batch_size=128 \
    --max_num_epochs=210 \
    --patience=50 \
    --eval_interval=1 \
    --ema \
    --num_workers=8 \
    --error_table='PerAtomMAE' \
    --default_dtype="float64"\
    --seed=0 \
    --save_cpu \
    --restart_latest \
    --loss="weighted" \
    --scheduler_patience=20 \
    --lr=0.01 \
    --swa \
    --swa_lr=0.00025 \
    --swa_forces_weight=100 \
    --start_swa=190 \
    --config="multihead_config/jz_spice_mp_config.yaml" \
    --device=cuda \
    --distributed \
    

    # seed 0 for test, seed 123 for first run
