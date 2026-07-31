#!/usr/bin/env bash
set -u
set -o pipefail

###############################################################################
# Production staged rigid-feature benchmark
#
# Stage 1: energy-only warmup
#   energy_weight = 1
#   forces_weight = 0
#   max epochs    = 100
#
# Stage 2: force-aware fine-tune
#   energy_weight = 100
#   forces_weight = 1
#   restart_latest from Stage 1
#
# Feature modes:
#   none, isotropic, traceless_moi, moi
#
# Full publication split:
#   publication_splits/random_train.xyz
#   publication_splits/random_valid.xyz
#   publication_splits/random_test.xyz
###############################################################################

ROOT="$(pwd)"

TRAIN_FILE="publication_splits/random_train.xyz"
VALID_FILE="publication_splits/random_valid.xyz"
TEST_FILE="publication_splits/random_test.xyz"

MODES=(
  none
  isotropic
  traceless_moi
  moi
)

SEEDS=(
  29
  101
  202
  303
  404
)

PROD_ROOT="production_runs"
CONSOLE_DIR="${PROD_ROOT}/console"
TIMING_DIR="${PROD_ROOT}/timing"
ARTIFACT_DIR="${PROD_ROOT}/artifacts"
METADATA_DIR="${PROD_ROOT}/metadata"
SUMMARY_DIR="${PROD_ROOT}/summary"

ARCHIVE_ROOT="production_archive"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${STAMP}_pre_production_cleanup"

mkdir -p \
  "${CONSOLE_DIR}" \
  "${TIMING_DIR}" \
  "${ARTIFACT_DIR}" \
  "${METADATA_DIR}" \
  "${SUMMARY_DIR}" \
  "${ARCHIVE_DIR}"

MASTER_LOG="${PROD_ROOT}/production_master.log"
SUMMARY_TSV="${SUMMARY_DIR}/production_runs.tsv"

touch "${MASTER_LOG}"

echo "============================================================" | tee -a "${MASTER_LOG}"
echo "Production staged feature-mode benchmark" | tee -a "${MASTER_LOG}"
echo "Started: $(date)" | tee -a "${MASTER_LOG}"
echo "Root: ${ROOT}" | tee -a "${MASTER_LOG}"
echo "Archive: ${ARCHIVE_DIR}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

###############################################################################
# Metadata
###############################################################################

{
  echo "root=${ROOT}"
  echo "train_file=${TRAIN_FILE}"
  echo "valid_file=${VALID_FILE}"
  echo "test_file=${TEST_FILE}"
  echo "stage1_energy_weight=1"
  echo "stage1_forces_weight=0"
  echo "stage1_epochs=100"
  echo "stage2_energy_weight=100"
  echo "stage2_forces_weight=1"
  echo "stage2_max_epochs=320"
  echo "modes=${MODES[*]}"
  echo "seeds=${SEEDS[*]}"
  echo "started=$(date)"
} > "${METADATA_DIR}/production_config.txt"

git rev-parse HEAD > "${METADATA_DIR}/git_commit.txt" 2>/dev/null || true
git branch --show-current > "${METADATA_DIR}/git_branch.txt" 2>/dev/null || true
git status --short > "${METADATA_DIR}/git_status_start.txt" 2>/dev/null || true
python -m pip freeze > "${METADATA_DIR}/pip_freeze.txt" 2>/dev/null || true

python - <<'PY' > production_runs/metadata/python_versions.txt 2>/dev/null || true
import sys
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
except Exception as exc:
    print("torch: unavailable", exc)
try:
    import e3nn
    print("e3nn:", e3nn.__version__)
except Exception as exc:
    print("e3nn: unavailable", exc)
try:
    import mace
    print("mace:", getattr(mace, "__version__", "unknown"))
except Exception as exc:
    print("mace: unavailable", exc)
PY

###############################################################################
# Safety checks
###############################################################################

for file in "${TRAIN_FILE}" "${VALID_FILE}" "${TEST_FILE}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: Missing required split file: ${file}" | tee -a "${MASTER_LOG}"
    exit 1
  fi
done

if ! PYTHONPATH=. python -m mace.cli.run_train --help | grep -q -- '--rigid_feature_mode'; then
  echo "ERROR: this checkout does not expose --rigid_feature_mode" | tee -a "${MASTER_LOG}"
  exit 1
fi

###############################################################################
# Archive old matching production artifacts, then remove active copies.
###############################################################################

echo | tee -a "${MASTER_LOG}"
echo "Archiving old production artifacts..." | tee -a "${MASTER_LOG}"

mkdir -p \
  "${ARCHIVE_DIR}/checkpoints" \
  "${ARCHIVE_DIR}/logs" \
  "${ARCHIVE_DIR}/results" \
  "${ARCHIVE_DIR}/root_models" \
  "${ARCHIVE_DIR}/production_runs"

# Archive previous production_runs contents.
if [[ -d "${PROD_ROOT}" ]]; then
  rsync -a \
    --exclude "${STAMP}_pre_production_cleanup" \
    "${PROD_ROOT}/" \
    "${ARCHIVE_DIR}/production_runs/" \
    2>/dev/null || true
fi

# Archive matching checkpoints/logs/results/root model files.
for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    base="prod_${mode}_full_staged_seed${seed}"

    cp -p checkpoints/${base}_run-${seed}* \
      "${ARCHIVE_DIR}/checkpoints/" \
      2>/dev/null || true

    cp -p logs/${base}_run-${seed}* \
      "${ARCHIVE_DIR}/logs/" \
      2>/dev/null || true

    cp -p results/${base}_run-${seed}* \
      "${ARCHIVE_DIR}/results/" \
      2>/dev/null || true

    cp -p ${base}.model ${base}_compiled.model \
      "${ARCHIVE_DIR}/root_models/" \
      2>/dev/null || true

    rm -f checkpoints/${base}_run-${seed}*
    rm -f logs/${base}_run-${seed}*
    rm -f results/${base}_run-${seed}*
    rm -f ${base}.model ${base}_compiled.model
  done
done

find "${ARCHIVE_DIR}" -type f -print0 \
  | xargs -0 shasum -a 256 \
  > "${ARCHIVE_DIR}/SHA256SUMS.txt" \
  2>/dev/null || true

###############################################################################
# Summary header
###############################################################################

printf "mode\tseed\tstage\tstatus\tbest_or_latest_epoch\tmodel\tpng\tlog\n" \
  > "${SUMMARY_TSV}"

###############################################################################
# Helpers
###############################################################################

latest_checkpoint_for() {
  local name="$1"
  local seed="$2"
  find checkpoints -maxdepth 1 -type f \
    -name "${name}_run-${seed}_epoch-*.pt" \
    | sort -V \
    | tail -n 1
}

epoch_from_checkpoint() {
  local path="$1"
  basename "${path}" \
    | sed -E 's/.*_epoch-([0-9]+)\.pt/\1/'
}

copy_artifacts_for() {
  local name="$1"
  local seed="$2"
  local mode="$3"
  local stage="$4"

  local dest="${ARTIFACT_DIR}/${name}"
  mkdir -p "${dest}"

  cp -p checkpoints/${name}_run-${seed}* "${dest}/" 2>/dev/null || true
  cp -p logs/${name}_run-${seed}* "${dest}/" 2>/dev/null || true
  cp -p results/${name}_run-${seed}* "${dest}/" 2>/dev/null || true
  cp -p ${name}.model ${name}_compiled.model "${dest}/" 2>/dev/null || true

  find "${dest}" -type f -print0 \
    | xargs -0 shasum -a 256 \
    > "${dest}/SHA256SUMS.txt" \
    2>/dev/null || true
}

run_stage1_energy_only() {
  local mode="$1"
  local seed="$2"
  local name="$3"
  local log="${CONSOLE_DIR}/${name}_stage1_energy_only.log"
  local timefile="${TIMING_DIR}/${name}_stage1_energy_only.time"

  echo | tee -a "${MASTER_LOG}"
  echo "============================================================" | tee -a "${MASTER_LOG}"
  echo "STAGE 1 ENERGY ONLY: ${name}" | tee -a "${MASTER_LOG}"
  echo "mode=${mode} seed=${seed}" | tee -a "${MASTER_LOG}"
  echo "============================================================" | tee -a "${MASTER_LOG}"

  /usr/bin/time -p \
  env PYTHONPATH=. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python -m mace.cli.run_train \
    --name "${name}" \
    --train_file "${TRAIN_FILE}" \
    --valid_file "${VALID_FILE}" \
    --test_file "${TEST_FILE}" \
    --energy_key energy \
    --forces_key forces \
    --E0s='{0:0.0}' \
    --rigid_feature_mode "${mode}" \
    --model ScaleShiftMACE \
    --num_interactions 2 \
    --num_channels 64 \
    --max_L 3 \
    --correlation 3 \
    --num_radial_basis 16 \
    --num_cutoff_basis 5 \
    --r_max 5.0 \
    --energy_weight 1 \
    --forces_weight 0 \
    --batch_size 16 \
    --valid_batch_size 16 \
    --lr 0.001 \
    --lr_factor 0.5 \
    --scheduler_patience 15 \
    --max_num_epochs 100 \
    --patience 80 \
    --ema \
    --ema_decay 0.99 \
    --seed "${seed}" \
    --device cpu \
    --default_dtype float64 \
    --save_cpu \
    --plot True \
    2> >(tee "${timefile}" >&2) \
    2>&1 | tee "${log}"

  local status=${PIPESTATUS[0]}

  local latest
  latest="$(latest_checkpoint_for "${name}" "${seed}")"

  local epoch="NA"
  if [[ -n "${latest}" ]]; then
    epoch="$(epoch_from_checkpoint "${latest}")"
  fi

  local model_path="checkpoints/${name}_run-${seed}.model"
  local png_path="results/${name}_run-${seed}_train_Default_stage_one.png"

  if [[ "${status}" -eq 0 ]]; then
    printf "%s\t%s\tstage1_energy_only\tok\t%s\t%s\t%s\t%s\n" \
      "${mode}" "${seed}" "${epoch}" "${model_path}" "${png_path}" "${log}" \
      >> "${SUMMARY_TSV}"
  else
    printf "%s\t%s\tstage1_energy_only\tfailed_${status}\t%s\t%s\t%s\t%s\n" \
      "${mode}" "${seed}" "${epoch}" "${model_path}" "${png_path}" "${log}" \
      >> "${SUMMARY_TSV}"
  fi

  copy_artifacts_for "${name}" "${seed}" "${mode}" "stage1"

  return "${status}"
}

run_stage2_finetune() {
  local mode="$1"
  local seed="$2"
  local name="$3"
  local log="${CONSOLE_DIR}/${name}_stage2_ew100_fw1.log"
  local timefile="${TIMING_DIR}/${name}_stage2_ew100_fw1.time"

  local latest
  latest="$(latest_checkpoint_for "${name}" "${seed}")"

  if [[ -z "${latest}" ]]; then
    echo "ERROR: No checkpoint found for ${name}; cannot start Stage 2." | tee -a "${MASTER_LOG}"
    printf "%s\t%s\tstage2_ew100_fw1\tmissing_checkpoint\tNA\tNA\tNA\t%s\n" \
      "${mode}" "${seed}" "${log}" \
      >> "${SUMMARY_TSV}"
    return 2
  fi

  local start_epoch
  start_epoch="$(epoch_from_checkpoint "${latest}")"

  echo | tee -a "${MASTER_LOG}"
  echo "============================================================" | tee -a "${MASTER_LOG}"
  echo "STAGE 2 EW100 FW1: ${name}" | tee -a "${MASTER_LOG}"
  echo "mode=${mode} seed=${seed}" | tee -a "${MASTER_LOG}"
  echo "restarting from ${latest}" | tee -a "${MASTER_LOG}"
  echo "============================================================" | tee -a "${MASTER_LOG}"

  /usr/bin/time -p \
  env PYTHONPATH=. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python -m mace.cli.run_train \
    --name "${name}" \
    --train_file "${TRAIN_FILE}" \
    --valid_file "${VALID_FILE}" \
    --test_file "${TEST_FILE}" \
    --energy_key energy \
    --forces_key forces \
    --E0s='{0:0.0}' \
    --rigid_feature_mode "${mode}" \
    --model ScaleShiftMACE \
    --num_interactions 2 \
    --num_channels 64 \
    --max_L 3 \
    --correlation 3 \
    --num_radial_basis 16 \
    --num_cutoff_basis 5 \
    --r_max 5.0 \
    --energy_weight 100 \
    --forces_weight 1 \
    --batch_size 16 \
    --valid_batch_size 16 \
    --lr 0.0001 \
    --lr_factor 0.5 \
    --scheduler_patience 8 \
    --max_num_epochs 320 \
    --patience 40 \
    --ema \
    --ema_decay 0.99 \
    --seed "${seed}" \
    --device cpu \
    --default_dtype float64 \
    --save_cpu \
    --plot True \
    --restart_latest \
    2> >(tee "${timefile}" >&2) \
    2>&1 | tee "${log}"

  local status=${PIPESTATUS[0]}

  latest="$(latest_checkpoint_for "${name}" "${seed}")"

  local epoch="NA"
  if [[ -n "${latest}" ]]; then
    epoch="$(epoch_from_checkpoint "${latest}")"
  fi

  local model_path="checkpoints/${name}_run-${seed}.model"
  local png_path="results/${name}_run-${seed}_train_Default_stage_one.png"

  if [[ "${status}" -eq 0 ]]; then
    printf "%s\t%s\tstage2_ew100_fw1\tok\t%s\t%s\t%s\t%s\n" \
      "${mode}" "${seed}" "${epoch}" "${model_path}" "${png_path}" "${log}" \
      >> "${SUMMARY_TSV}"
  else
    printf "%s\t%s\tstage2_ew100_fw1\tfailed_${status}\t%s\t%s\t%s\t%s\n" \
      "${mode}" "${seed}" "${epoch}" "${model_path}" "${png_path}" "${log}" \
      >> "${SUMMARY_TSV}"
  fi

  copy_artifacts_for "${name}" "${seed}" "${mode}" "stage2"

  return "${status}"
}

###############################################################################
# Run production grid
###############################################################################

FAILURES=0

for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    NAME="prod_${mode}_full_staged_seed${seed}"

    echo | tee -a "${MASTER_LOG}"
    echo "############################################################" | tee -a "${MASTER_LOG}"
    echo "RUN: ${NAME}" | tee -a "${MASTER_LOG}"
    echo "############################################################" | tee -a "${MASTER_LOG}"

    run_stage1_energy_only "${mode}" "${seed}" "${NAME}"
    STATUS1=$?

    if [[ "${STATUS1}" -ne 0 ]]; then
      echo "Stage 1 failed for ${NAME}; skipping Stage 2." | tee -a "${MASTER_LOG}"
      FAILURES=$((FAILURES + 1))
      continue
    fi

    run_stage2_finetune "${mode}" "${seed}" "${NAME}"
    STATUS2=$?

    if [[ "${STATUS2}" -ne 0 ]]; then
      echo "Stage 2 failed for ${NAME}." | tee -a "${MASTER_LOG}"
      FAILURES=$((FAILURES + 1))
      continue
    fi
  done
done

###############################################################################
# Final metadata and checksums
###############################################################################

git status --short > "${METADATA_DIR}/git_status_end.txt" 2>/dev/null || true

find "${PROD_ROOT}" -type f -print0 \
  | xargs -0 shasum -a 256 \
  > "${METADATA_DIR}/SHA256SUMS.txt" \
  2>/dev/null || true

echo | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"
echo "Production run finished: $(date)" | tee -a "${MASTER_LOG}"
echo "Failures: ${FAILURES}" | tee -a "${MASTER_LOG}"
echo "Summary: ${SUMMARY_TSV}" | tee -a "${MASTER_LOG}"
echo "Artifacts: ${ARTIFACT_DIR}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

if [[ "${FAILURES}" -ne 0 ]]; then
  exit 1
fi

exit 0
