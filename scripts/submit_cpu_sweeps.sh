#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/n/home13/asaluja/sgt/2252"
SLURM_SCRIPT="${REPO_DIR}/scripts/slurm_sweep_cpu.sh"
PARTITION="${PARTITION:-serial_requeue}"
TIME="${TIME:-2-00:00:00}"

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/results/slurm"

if [[ ! -x "${REPO_DIR}/.venv/bin/python" ]]; then
  echo "Missing ${REPO_DIR}/.venv/bin/python" >&2
  echo "Run: cd ${REPO_DIR} && python3 -m venv .venv && .venv/bin/python -m pip install -e ." >&2
  exit 1
fi

submit_prep() {
  local size="$1"
  sbatch --parsable \
    --partition="${PARTITION}" \
    --time="${TIME}" \
    --job-name="sgt_${size}_prep" \
    --export=ALL,SIZE="${size}",MODE=prep \
    "${SLURM_SCRIPT}"
}

submit_sweep() {
  local size="$1"
  local shards="$2"
  local dependency="$3"
  local array_spec="0-$((shards - 1))"

  sbatch --parsable \
    --partition="${PARTITION}" \
    --time="${TIME}" \
    --job-name="sgt_${size}_sweep" \
    --array="${array_spec}" \
    --dependency="afterok:${dependency}" \
    --export=ALL,SIZE="${size}",MODE=sweep,SHARDS="${shards}" \
    "${SLURM_SCRIPT}"
}

small_prep="$(submit_prep small)"
small_sweep="$(submit_sweep small 1 "${small_prep}")"

medium_prep="$(submit_prep medium)"
medium_sweep="$(submit_sweep medium 2 "${medium_prep}")"

large_prep="$(submit_prep large)"
large_sweep="$(submit_sweep large 8 "${large_prep}")"

printf 'small prep:  %s\n' "${small_prep}"
printf 'small sweep: %s\n' "${small_sweep}"
printf 'medium prep:  %s\n' "${medium_prep}"
printf 'medium sweep: %s\n' "${medium_sweep}"
printf 'large prep:  %s\n' "${large_prep}"
printf 'large sweep: %s\n' "${large_sweep}"
