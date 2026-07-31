#!/usr/bin/env bash
set -u
set -o pipefail

###############################################################################
# Resume-safe production staged rigid-feature benchmark
#
# Does NOT delete or archive active artifacts.
#
# For each mode/seed:
#   Stage 1: energy-only warmup, ew=1, fw=0, max 100 epochs
#   Stage 2: restart_latest, ew=100, fw=1, max 320 epochs
#
# Completion detection:
#   Stage 1 complete if its stage1 log contains "Done"
#   Stage 2 complete if its stage2 log contains "Done"
#
# This allows safe resumption after Ctrl+C.
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

MASTER_LOG="${PROD_ROOT}/production_resume_master.log"
SUMMARY_TSV="${SUMMARY_DIR}/production_resume_summary.tsv"

mkdir -p \
  "${CONSOLE_DIR}" \
  "${TIMING_DIR}" \
  "${ARTIFACT_DIR}" \
  "${METADATA_DIR}" \
  "${SUMMARY_DIR}" \
  checkpoints \
  logs \
  results

touch "${MASTER_LOG}"

echo "============================================================" | tee -a "${MASTER_LOG}"
echo "Resume-safe production benchmark" | tee -a "${MASTER_LOG}"
echo "Started: $(date)" | tee -a "${MASTER_LOG}"
echo "Root: ${ROOT}" | tee -a "${MASTER_LOG}"
echo "No files will be deleted by this script." | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

{
  echo "root=${ROOT}"
  echo "train_file=${TRAIN_FILE}"
  echo "valid_file=${VALID_FILE}"
  echo "test_file=${TEST_FILE}"
  echo "stage1_energy_weight=1"
  echo "stage1_forces_weight=0"
  echo "stage1_max_epochs=100"
  echo "stage2_energy_weight=100"
  echo "stage2_forces_weight=1"
  echo "stage2_max_epochs=320"
  echo "modes=${MODES[*]}"
  echo "seeds=${SEEDS[*]}"
  echo "started=$(date)"
} > "${METADATA_DIR}/production_resume_config.txt"

git rev-parse HEAD > "${METADATA_DIR}/git_commit_resume.txt" 2>/dev/null || true
git branch --show-current > "${METADATA_DIR}/git_branch_resume.txt" 2>/dev/null || true
git status --short > "${METADATA_DIR}/git_status_resume_start.txt" 2>/dev/null || true
python -m pip freeze > "${METADATA_DIR}/pip_freeze_resume.txt" 2>/dev/null || true

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

printf "mode\tseed\tstage\taction\tstatus\tlatest_epoch\tmodel\tpng\tlog\n" \
  > "${SUMMARY_TSV}"

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

  if [[ -z "${path}" ]]; then
    echo "NA"
    return
  fi

  basename "${path}" \
    | sed -E 's/.*_epoch-([0-9]+)\.pt/\1/'
}

log_done() {
  local log="$1"

  [[ -f "${log}" ]] && grep -q "Done" "${log}"
}

copy_artifacts_for() {
  local name="$1"
  local seed="$2"

  local dest="${ARTIFACT_DIR}/${name}"
  mkdir -p "${dest}"

  cp -p checkpoints/${name}_run-${seed}* "${dest}/" 2>/dev/null || true
  cp -p logs/${name}_run-${seed}* "${dest}/" 2>/dev/null || true
  cp -p results/${name}_run-${seed}* "${dest}/" 2>/dev/null || true
  cp -p ${name}.model ${name}_compiled.model "${dest}/" 2>/dev/null || true
  cp -p "${CONSOLE_DIR}/${name}"*.log "${dest}/" 2>/dev/null || true

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

  if log_done "${log}"; then
    local latest
    latest="$(latest_checkpoint_for "${name}" "${seed}")"
    local epoch
    epoch="$(epoch_from_checkpoint "${latest}")"

    echo "SKIP stage1 already complete: ${name}" | tee -a "${MASTER_LOG}"

    printf "%s\t%s\tstage1_energy_only\tskip\talready_done\t%s\t%s\t%s\t%s\n" \
      "${mode}" "${seed}" "${epoch}" \
      "checkpoints/${name}_run-${seed}.model" \
      "results/${name}_run-${seed}_train_Default_stage_one.png" \
      "${log}" \
      >> "${SUMMARY_TSV}"

    return 0
  fi

  echo | tee -a "${MASTER_LOG}"
  echo "============================================================" | tee -a "${MASTER_LOG}"
  echo "RUN stage1 energy-only: ${name}" | tee -a "${MASTER_LOG}"
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
    2>&1 | tee "${log}"

  local status=${PIPESTATUS[0]}

  local latest
  latest="$(latest_checkpoint_for "${name}" "${seed}")"
  local epoch
  epoch="$(epoch_from_checkpoint "${latest}")"

  if [[ "${status}" -eq 0 ]]; then
    printf "%s\t%s\tstage1_energy_only\trun\tok\t%s\t%s\t%s\t%s\n" \
      "${mode}" "${seed}" "${epoch}" \
      "checkpoints/${name}_run-${seed}.model" \
      "results/${name}_run-${seed}_train_Default_stage_one.png" \
      "${log}" \
      >> "${SUMMARY_TSV}"
  else
    printf "%s\t%s\tstage1_energy_only\trun\tfailed_${status}\t%s\t%s\t%s\t%s\n" \
      "${mode}" "${seed}" "${epoch}" \
      "checkpoints/${name}_run-${seed}.model" \
      "results/${name}_run-${seed}_train_Default_stage_one.png" \
      "${log}" \
      >> "${SUMMARY_TSV}"
  fi

  copy_artifacts_for "${name}" "${seed}"

  return "${status}"
}

run_stage2_ew100_fw1() {
  local mode="$1"
  local seed="$2"
  local name="$3"

  local log="${CONSOLE_DIR}/${name}_stage2_ew100_fw1.log"
  local timefile="${TIMING_DIR}/${name}_stage2_ew100_fw1.time"

  if log_done "${log}"; then
    local latest
    latest="$(latest_checkpoint_for "${name}" "${seed}")"
    local epoch
    epoch="$(epoch_from_checkpoint "${latest}")"

    echo "SKIP stage2 already complete: ${name}" | tee -a "${MASTER_LOG}"

    printf "%s\t%s\tstage2_ew100_fw1\tskip\talready_done\t%s\t%s\t%s\t%s\n" \
      "${mode}" "${seed}" "${epoch}" \
      "checkpoints/${name}_run-${seed}.model" \
      "results/${name}_run-${seed}_train_Default_stage_one.png" \
      "${log}" \
      >> "${SUMMARY_TSV}"

    return 0
  fi

  local latest
  latest="$(latest_checkpoint_for "${name}" "${seed}")"

  if [[ -z "${latest}" ]]; then
    echo "ERROR: no checkpoint available for stage2: ${name}" | tee -a "${MASTER_LOG}"

    printf "%s\t%s\tstage2_ew100_fw1\trun\tmissing_checkpoint\tNA\t%s\t%s\t%s\n" \
      "${mode}" "${seed}" \
      "checkpoints/${name}_run-${seed}.model" \
      "results/${name}_run-${seed}_train_Default_stage_one.png" \
      "${log}" \
      >> "${SUMMARY_TSV}"

    return 2
  fi

  echo | tee -a "${MASTER_LOG}"
  echo "============================================================" | tee -a "${MASTER_LOG}"
  echo "RUN stage2 ew100/fw1: ${name}" | tee -a "${MASTER_LOG}"
  echo "mode=${mode} seed=${seed}" | tee -a "${MASTER_LOG}"
  echo "restart checkpoint=${latest}" | tee -a "${MASTER_LOG}"
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
    2>&1 | tee "${log}"

  local status=${PIPESTATUS[0]}

  latest="$(latest_checkpoint_for "${name}" "${seed}")"
  local epoch
  epoch="$(epoch_from_checkpoint "${latest}")"

  if [[ "${status}" -eq 0 ]]; then
    printf "%s\t%s\tstage2_ew100_fw1\trun\tok\t%s\t%s\t%s\t%s\n" \
      "${mode}" "${seed}" "${epoch}" \
      "checkpoints/${name}_run-${seed}.model" \
      "results/${name}_run-${seed}_train_Default_stage_one.png" \
      "${log}" \
      >> "${SUMMARY_TSV}"
  else
    printf "%s\t%s\tstage2_ew100_fw1\trun\tfailed_${status}\t%s\t%s\t%s\t%s\n" \
      "${mode}" "${seed}" "${epoch}" \
      "checkpoints/${name}_run-${seed}.model" \
      "results/${name}_run-${seed}_train_Default_stage_one.png" \
      "${log}" \
      >> "${SUMMARY_TSV}"
  fi

  copy_artifacts_for "${name}" "${seed}"

  return "${status}"
}

FAILURES=0

for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    NAME="prod_${mode}_full_staged_seed${seed}"

    echo | tee -a "${MASTER_LOG}"
    echo "############################################################" | tee -a "${MASTER_LOG}"
    echo "CHECKING ${NAME}" | tee -a "${MASTER_LOG}"
    echo "############################################################" | tee -a "${MASTER_LOG}"

    run_stage1_energy_only "${mode}" "${seed}" "${NAME}"
    STATUS1=$?

    if [[ "${STATUS1}" -ne 0 ]]; then
      echo "Stage 1 failed or unavailable for ${NAME}; skipping stage 2." | tee -a "${MASTER_LOG}"
      FAILURES=$((FAILURES + 1))
      continue
    fi

    run_stage2_ew100_fw1 "${mode}" "${seed}" "${NAME}"
    STATUS2=$?

    if [[ "${STATUS2}" -ne 0 ]]; then
      echo "Stage 2 failed for ${NAME}." | tee -a "${MASTER_LOG}"
      FAILURES=$((FAILURES + 1))
      continue
    fi
  done
done

git status --short > "${METADATA_DIR}/git_status_resume_end.txt" 2>/dev/null || true

find "${PROD_ROOT}" -type f -print0 \
  | xargs -0 shasum -a 256 \
  > "${METADATA_DIR}/SHA256SUMS_resume.txt" \
  2>/dev/null || true

echo | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"
echo "Resume-safe production run finished: $(date)" | tee -a "${MASTER_LOG}"
echo "Failures: ${FAILURES}" | tee -a "${MASTER_LOG}"
echo "Summary: ${SUMMARY_TSV}" | tee -a "${MASTER_LOG}"
echo "Artifacts: ${ARTIFACT_DIR}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

if [[ "${FAILURES}" -ne 0 ]]; then
  exit 1
fi

exit 0
