#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Configuration
###############################################################################

SEED=29

MODES=(
  none
  isotropic
  traceless_moi
  moi
)

SIZES=(
  128
  full
)

VALID_FILE="publication_splits/random_valid.xyz"
TEST_FILE="publication_splits/random_test.xyz"

PILOT_ROOT="publication_pilot"
CONSOLE_DIR="${PILOT_ROOT}/console"
TIMING_DIR="${PILOT_ROOT}/timing"
TORQUE_DIR="${PILOT_ROOT}/torque"
MODEL_MAP="${PILOT_ROOT}/model_paths.tsv"

mkdir -p \
  "${PILOT_ROOT}" \
  "${CONSOLE_DIR}" \
  "${TIMING_DIR}" \
  "${TORQUE_DIR}"

printf "name\tmode\ttrain_size\tseed\tmodel_path\n" > "${MODEL_MAP}"

###############################################################################
# Preflight checks
###############################################################################

echo "============================================================"
echo "Rigid-feature publication pilot"
echo "============================================================"
echo "Repository: $(pwd)"
echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Seed: ${SEED}"
echo

required_files=(
  "${VALID_FILE}"
  "${TEST_FILE}"
  "publication_splits/nested_subsets/train_128.xyz"
  "evaluate_test_torques.py"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: Required file not found: ${path}" >&2
    exit 1
  fi
done

PYTHONPATH=. python - <<'PY'
from mace.tools.arg_parser import build_default_arg_parser

parser = build_default_arg_parser()

for mode in (
    "none",
    "isotropic",
    "traceless_moi",
    "moi",
):
    args = parser.parse_args(
        [
            "--name",
            f"preflight_{mode}",
            "--rigid_feature_mode",
            mode,
        ]
    )
    assert args.rigid_feature_mode == mode

print("CLI rigid-feature modes verified.")
PY

PYTHONPATH=. python - <<'PY'
from ase.io import read

paths = (
    "publication_splits/nested_subsets/train_128.xyz",
    "publication_splits/random_train.xyz",
    "publication_splits/random_valid.xyz",
    "publication_splits/random_test.xyz",
)

for path in paths:
    frames = read(path, index=":")

    if not frames:
        raise RuntimeError(f"No frames found in {path}")

    atoms = frames[0]

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    if "quaternions" not in atoms.arrays:
        raise RuntimeError(
            f"Missing quaternions array in {path}"
        )

    print(
        f"{path}: "
        f"frames={len(frames)}, "
        f"energy={energy:.8g}, "
        f"forces_shape={forces.shape}"
    )

print("Dataset preflight verified.")
PY

###############################################################################
# Helper functions
###############################################################################

find_model() {
  local name="$1"

  local model
  model=$(
    find checkpoints models . \
      -maxdepth 2 \
      -type f \
      -name "${name}*.model" \
      2>/dev/null \
      | sort \
      | tail -n 1
  )

  if [[ -z "${model}" ]]; then
    echo "ERROR: No finalized model found for ${name}" >&2
    return 1
  fi

  printf "%s" "${model}"
}

run_training() {
  local mode="$1"
  local size="$2"

  local train_file
  local batch_size
  local name
  local console_log
  local timing_log

  if [[ "${size}" == "full" ]]; then
    train_file="publication_splits/random_train.xyz"
    batch_size=16
  else
    train_file="publication_splits/nested_subsets/train_${size}.xyz"
    batch_size=8
  fi

  name="pilot_${mode}_${size}_seed${SEED}"
  console_log="${CONSOLE_DIR}/${name}.log"
  timing_log="${TIMING_DIR}/${name}.time"

  echo
  echo "============================================================"
  echo "Training: ${name}"
  echo "Mode: ${mode}"
  echo "Train set: ${train_file}"
  echo "Batch size: ${batch_size}"
  echo "============================================================"

  rm -f "${console_log}" "${timing_log}"

  {
    /usr/bin/time -p \
      env PYTHONPATH=. \
      python -m mace.cli.run_train \
        --name "${name}" \
        --train_file "${train_file}" \
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
        --energy_weight 7.5 \
        --forces_weight 10.0 \
        --batch_size "${batch_size}" \
        --valid_batch_size 16 \
        --lr 0.001 \
        --lr_factor 0.5 \
        --scheduler_patience 20 \
        --max_num_epochs 300 \
        --patience 60 \
        --ema \
        --ema_decay 0.99 \
        --seed "${SEED}" \
        --device cpu \
        --default_dtype float64 \
        --save_cpu \
        --plot True
  } > >(tee "${console_log}") \
    2> >(tee -a "${console_log}" "${timing_log}" >&2)

  local model_path
  model_path="$(find_model "${name}")"

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${name}" \
    "${mode}" \
    "${size}" \
    "${SEED}" \
    "${model_path}" \
    >> "${MODEL_MAP}"

  echo "Finalized model: ${model_path}"
}

run_torque_evaluation() {
  local name="$1"
  local mode="$2"
  local size="$3"
  local model_path="$4"

  local prefix="${TORQUE_DIR}/${name}"
  local console_log="${TORQUE_DIR}/${name}.log"

  echo
  echo "============================================================"
  echo "Torque evaluation: ${name}"
  echo "Model: ${model_path}"
  echo "============================================================"

  PYTHONPATH=. \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python evaluate_test_torques.py \
    --model "${model_path}" \
    --data "${TEST_FILE}" \
    --epsilon 1e-4 \
    --output-prefix "${prefix}" \
    2>&1 | tee "${console_log}"
}

###############################################################################
# Training matrix
###############################################################################

for size in "${SIZES[@]}"; do
  for mode in "${MODES[@]}"; do
    run_training "${mode}" "${size}"
  done
done

###############################################################################
# Verify finalized models
###############################################################################

echo
echo "============================================================"
echo "Verifying model modes and parameter counts"
echo "============================================================"

PYTHONPATH=. python - "${MODEL_MAP}" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch


model_map = Path(sys.argv[1])

rows = []

with model_map.open(
    newline="",
    encoding="utf-8",
) as handle:
    reader = csv.DictReader(
        handle,
        delimiter="\t",
    )
    rows = list(reader)

if not rows:
    raise RuntimeError("Model map is empty.")

parameter_counts = {}

for row in rows:
    path = Path(row["model_path"])

    model = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    stored_mode = getattr(
        model,
        "rigid_feature_mode",
        None,
    )

    expected_mode = row["mode"]

    if stored_mode != expected_mode:
        raise AssertionError(
            f"{path}: expected mode {expected_mode!r}, "
            f"found {stored_mode!r}"
        )

    count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    parameter_counts[row["name"]] = count

    print(
        f"{row['name']:42s} "
        f"mode={stored_mode:16s} "
        f"parameters={count}"
    )

unique_counts = set(parameter_counts.values())

if len(unique_counts) != 1:
    raise AssertionError(
        "Feature modes do not have identical parameter counts: "
        f"{parameter_counts}"
    )

print()
print(
    "All models have identical parameter counts:",
    unique_counts.pop(),
)
PY

###############################################################################
# Torque evaluations
###############################################################################

tail -n +2 "${MODEL_MAP}" |
while IFS=$'\t' read -r name mode size seed model_path; do
  run_torque_evaluation \
    "${name}" \
    "${mode}" \
    "${size}" \
    "${model_path}"
done

###############################################################################
# Collect summaries
###############################################################################

cat > "${PILOT_ROOT}/collect_results.py" <<'PY'
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch


ROOT = Path("publication_pilot")
MODEL_MAP = ROOT / "model_paths.tsv"
CONSOLE_DIR = ROOT / "console"
TIMING_DIR = ROOT / "timing"
TORQUE_DIR = ROOT / "torque"


def find_last_float(
    text: str,
    patterns: tuple[str, ...],
) -> float | None:
    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue

    return None


def read_timing(name: str) -> dict[str, float | None]:
    path = TIMING_DIR / f"{name}.time"

    if not path.exists():
        return {
            "wall_seconds": None,
            "user_seconds": None,
            "sys_seconds": None,
        }

    text = path.read_text(
        errors="replace"
    )

    return {
        "wall_seconds": find_last_float(
            text,
            (r"^real\s+([0-9eE+\-.]+)$",),
        ),
        "user_seconds": find_last_float(
            text,
            (r"^user\s+([0-9eE+\-.]+)$",),
        ),
        "sys_seconds": find_last_float(
            text,
            (r"^sys\s+([0-9eE+\-.]+)$",),
        ),
    }


def read_training_metrics(
    name: str,
) -> dict[str, float | None]:
    path = CONSOLE_DIR / f"{name}.log"

    if not path.exists():
        return {}

    text = path.read_text(
        errors="replace"
    )

    return {
        "best_epoch": find_last_float(
            text,
            (
                r"best[_ ]epoch[^0-9]*([0-9]+)",
                r"epoch[^0-9]*([0-9]+).*best",
            ),
        ),
        "valid_energy_rmse": find_last_float(
            text,
            (
                r"valid.*?energy.*?rmse[^0-9eE+\-.]*"
                r"([0-9eE+\-.]+)",
                r"energy_rmse[^0-9eE+\-.]*"
                r"([0-9eE+\-.]+)",
            ),
        ),
        "valid_force_rmse": find_last_float(
            text,
            (
                r"valid.*?force.*?rmse[^0-9eE+\-.]*"
                r"([0-9eE+\-.]+)",
                r"forces_rmse[^0-9eE+\-.]*"
                r"([0-9eE+\-.]+)",
            ),
        ),
    }


def read_torque_metrics(
    name: str,
) -> dict[str, float | None]:
    data_path = (
        TORQUE_DIR
        / f"{name}_data.npz"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing torque data: {data_path}"
        )

    data = np.load(data_path)

    reference = np.asarray(
        data["torque_reference"],
        dtype=np.float64,
    )
    prediction = np.asarray(
        data["torque_prediction"],
        dtype=np.float64,
    )

    error = prediction - reference

    mae = float(
        np.mean(np.abs(error))
    )
    rmse = float(
        np.sqrt(np.mean(error**2))
    )

    reference_rms = float(
        np.sqrt(np.mean(reference**2))
    )

    relative_rmse = (
        100.0 * rmse / reference_rms
        if reference_rms > 0.0
        else float("nan")
    )

    centered = (
        reference - np.mean(reference)
    )

    denominator = float(
        np.sum(centered**2)
    )

    r2 = (
        1.0
        - float(np.sum(error**2))
        / denominator
        if denominator > 0.0
        else float("nan")
    )

    residuals = np.asarray(
        data["rotational_residuals"],
        dtype=np.float64,
    )

    residual_norms = np.linalg.norm(
        residuals,
        axis=1,
    )

    return {
        "torque_mae": mae,
        "torque_rmse": rmse,
        "torque_relative_rmse_percent": (
            relative_rmse
        ),
        "torque_r2": r2,
        "rotational_residual_rms": float(
            np.sqrt(
                np.mean(
                    residual_norms**2
                )
            )
        ),
        "rotational_residual_max": float(
            np.max(residual_norms)
        ),
        "predicted_torque_rms": float(
            np.sqrt(
                np.mean(
                    prediction**2
                )
            )
        ),
    }


with MODEL_MAP.open(
    newline="",
    encoding="utf-8",
) as handle:
    model_rows = list(
        csv.DictReader(
            handle,
            delimiter="\t",
        )
    )

rows = []

for model_row in model_rows:
    name = model_row["name"]
    model_path = Path(
        model_row["model_path"]
    )

    model = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    row = {
        **model_row,
        "parameter_count": parameter_count,
        **read_timing(name),
        **read_training_metrics(name),
        **read_torque_metrics(name),
    }

    rows.append(row)

fieldnames = []

for row in rows:
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)

csv_path = ROOT / "pilot_summary.csv"

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(rows)

json_path = ROOT / "pilot_summary.json"

json_path.write_text(
    json.dumps(
        rows,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("Pilot summary")
print("=" * 120)

display_columns = (
    "name",
    "wall_seconds",
    "torque_rmse",
    "torque_r2",
    "rotational_residual_rms",
)

for row in rows:
    print(
        f"{row['name']:42s} "
        f"wall={row.get('wall_seconds')} "
        f"torque_rmse={row.get('torque_rmse'):.8g} "
        f"torque_r2={row.get('torque_r2'):.8g} "
        f"rot_res={row.get('rotational_residual_rms'):.8g}"
    )

print()
print("Created:")
print(f"  {csv_path}")
print(f"  {json_path}")
PY

PYTHONPATH=. python "${PILOT_ROOT}/collect_results.py"

###############################################################################
# Final report
###############################################################################

echo
echo "============================================================"
echo "Pilot complete"
echo "============================================================"
echo
echo "Models:"
column -t -s $'\t' "${MODEL_MAP}" 2>/dev/null || cat "${MODEL_MAP}"
echo
echo "Summary:"
cat "${PILOT_ROOT}/pilot_summary.csv"
echo
echo "Artifacts:"
echo "  ${PILOT_ROOT}/pilot_summary.csv"
echo "  ${PILOT_ROOT}/pilot_summary.json"
echo "  ${PILOT_ROOT}/model_paths.tsv"
echo "  ${CONSOLE_DIR}/"
echo "  ${TIMING_DIR}/"
echo "  ${TORQUE_DIR}/"
