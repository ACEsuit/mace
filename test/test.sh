export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# mace_run_train \
#     --name="dielectric_model" \
#     --train_file="dielectric.xyz" \
#     --valid_fraction=0.05 \
#     --compute_stress=True \
#     --compute_forces=True \
#     --compute_field=True \
#     --energy_weight=1.0 \
#     --forces_weight=10.0 \
#     --stress_weight=100.0 \
#     --bec_weight=10.0 \
#     --polarisability_weight=100.0 \
#     --config_type_weights='{"Default":1.0}' \
#     --E0s='average' \
#     --model="ScaleShiftMACE" \
#     --hidden_irreps='128x0e + 128x1o' \
#     --loss="universal_field" \
#     --r_max=5.0 \
#     --batch_size=2 \
#     --valid_batch_size=2 \
#     --max_num_epochs=150 \
#     --amsgrad \
#     --restart_latest \
#     --device=cuda \
#     --error_table="PerAtomRMSEstressvirialsfield" \

mace_run_train \
    --name="dielectric_model" \
    --train_file="dielectric.xyz" \
    --valid_fraction=0.10 \
    --compute_stress=True \
    --compute_forces=True \
    --compute_field=True \
    --energy_weight=1.0 \
    --forces_weight=10.0 \
    --stress_weight=100.0 \
    --polarisation_weight=0.0 \
    --bec_weight=10.0 \
    --polarisability_weight=100.0 \
    --config_type_weights='{"Default":1.0}' \
    --E0s='average' \
    --model="ScaleShiftFieldMACE" \
    --loss="universal_field" \
    --r_max=5.0 \
    --batch_size=1 \
    --valid_batch_size=1 \
    --max_num_epochs=150 \
    --amsgrad \
    --restart_latest \
    --device="cuda:1" \
    --model_dir="dielectric_model" \
    --log_dir="dielectric_log" \
    --checkpoints_dir="dielectric_checkpoints" \
    --results_dir="dielectric_results" \
    --E0s="average" \
    --num_interactions=2 \
    --correlation=3 \
    --num_channels=128 \
    --max_L=2 \
    --seed=124 \
    --save_cpu \
    --restart_latest \
    --default_dtype="float32" \
    --error_table="PerAtomRMSEstressvirialsfield" \
    --weight_decay=5e-10 \
    --clip_grad=10.0 \