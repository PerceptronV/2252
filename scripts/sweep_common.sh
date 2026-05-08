#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

SEEDS="${SEEDS:-0,1,2,3,4}"
KS="${KS:-2,3,4,6,8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/results/sweeps}"
GENERATE_DATASETS="${GENERATE_DATASETS:-1}"
RUN_SWEEP="${RUN_SWEEP:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

DATASET_FILES=()

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

start_logging() {
  local log_file="$1"
  mkdir -p "$(dirname "${log_file}")"
  printf 'Logging to %s\n' "${log_file}"
  exec >> "${log_file}" 2>&1
  log "Logging to ${log_file}"
}

run_step() {
  local label="$1"
  shift
  log "START ${label}"
  log "CMD $*"
  "$@"
  log "FINISH ${label}"
}

dataset_path() {
  local size="$1"
  local name="$2"
  printf '%s/datasets/sweep_generated/%s/%s.npz' "${REPO_DIR}" "${size}" "${name}"
}

add_dataset() {
  local path="$1"
  DATASET_FILES+=("${path}")
}

generate_dataset() {
  local label="$1"
  local output="$2"
  shift 2

  add_dataset "${output}"
  if [[ "${GENERATE_DATASETS}" != "1" && "${GENERATE_DATASETS}" != "true" ]]; then
    log "SKIP generation ${label}: GENERATE_DATASETS=${GENERATE_DATASETS}"
    return 0
  fi

  mkdir -p "$(dirname "${output}")"
  run_step "generate ${label}" "${PYTHON_BIN}" "${REPO_DIR}/scripts/datasets.py" "$@" --output "${output}" --name "$(basename "${output}" .npz)"
}

relative_dataset_list() {
  local selected=()
  local idx=0
  local rel

  for path in "${DATASET_FILES[@]}"; do
    if (( SHARD_COUNT > 1 )) && (( idx % SHARD_COUNT != SHARD_INDEX )); then
      idx=$((idx + 1))
      continue
    fi
    rel="${path#${REPO_DIR}/}"
    selected+=("${rel}")
    idx=$((idx + 1))
  done

  if (( ${#selected[@]} == 0 )); then
    log "No datasets selected for SHARD_INDEX=${SHARD_INDEX}, SHARD_COUNT=${SHARD_COUNT}"
    return 1
  fi

  local IFS=,
  printf '%s' "${selected[*]}"
}

run_sweep() {
  local sweep_name="$1"
  local output_root="$2"
  local dataset_spec
  dataset_spec="$(relative_dataset_list)"

  if [[ "${RUN_SWEEP}" != "1" && "${RUN_SWEEP}" != "true" ]]; then
    log "SKIP sweep ${sweep_name}: RUN_SWEEP=${RUN_SWEEP}"
    log "Would sweep datasets: ${dataset_spec}"
    return 0
  fi

  run_step "sweep ${sweep_name}" \
    "${PYTHON_BIN}" "${REPO_DIR}/scripts/sweep_benchmarks.py" \
    --datasets "${dataset_spec}" \
    --ks "${KS}" \
    --seeds "${SEEDS}" \
    --output-root "${output_root}" \
    --name "${sweep_name}"

  local sweep_dir="${output_root}/${sweep_name}"
  log "CSV ${sweep_dir}/summary.csv"
  log "JSONL ${sweep_dir}/summary.jsonl"
  log "FAILURES ${sweep_dir}/failures.jsonl"
  log "MANIFEST ${sweep_dir}/manifest.json"
}
