#!/usr/bin/env bash
set -euo pipefail

for MODE in none isotropic traceless_moi moi; do
  echo
  echo "========== ${MODE} =========="

  PYTHONPATH=. python -m mace.cli.run_train \
    --name "smoke_${MODE}" \
    --train_file publication_splits/nested_subsets/train_64.xyz \
    --valid_file publication_splits/random_valid.xyz \
    --test_file publication_splits/random_test.xyz \
    --energy_key energy \
    --forces_key forces \
    --E0s='{0:0.0}' \
    --rigid_feature_mode "${MODE}" \
    --model ScaleShiftMACE \
    --num_interactions 2 \
    --num_channels 64 \
    --max_L 3 \
    --correlation 3 \
    --num_radial_basis 16 \
    --num_cutoff_basis 5 \
    --r_max 5.0 \
    --energy_weight 7.5 \
    --forces_weight 10.0 \
    --batch_size 8 \
    --valid_batch_size 8 \
    --max_num_epochs 1 \
    --patience 1 \
    --seed 29 \
    --device cpu \
    --default_dtype float64
done
