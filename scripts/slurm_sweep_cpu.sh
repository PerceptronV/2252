#!/usr/bin/env bash
#SBATCH --job-name=sgt_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --partition=serial_requeue
#SBATCH --output=/n/home13/asaluja/sgt/2252/results/slurm/%x_%A_%a.out
#SBATCH --error=/n/home13/asaluja/sgt/2252/results/slurm/%x_%A_%a.err

set -euo pipefail

REPO_DIR="/n/home13/asaluja/sgt/2252"
SIZE="${SIZE:?set SIZE to small, medium, or large}"
MODE="${MODE:-sweep}"
SHARDS="${SHARDS:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/results/slurm"

export PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/results/sweeps}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

case "${SIZE}" in
  small) sweep_script="${REPO_DIR}/scripts/run_sweep_small.sh" ;;
  medium) sweep_script="${REPO_DIR}/scripts/run_sweep_medium.sh" ;;
  large) sweep_script="${REPO_DIR}/scripts/run_sweep_large.sh" ;;
  *) echo "unknown SIZE=${SIZE}; expected small, medium, or large" >&2; exit 2 ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create it with: python3 -m venv .venv && .venv/bin/python -m pip install -e ." >&2
  exit 1
fi

case "${MODE}" in
  prep)
    export GENERATE_DATASETS=1
    export RUN_SWEEP=0
    export SHARD_INDEX=0
    export SHARD_COUNT=1
    export SWEEP_NAME="${SWEEP_NAME:-${SIZE}_experimental_sweep_prep}"
    ;;
  sweep)
    export GENERATE_DATASETS=0
    export RUN_SWEEP=1
    export SHARD_INDEX="${SHARD}"
    export SHARD_COUNT="${SHARDS}"
    if [[ -z "${SWEEP_NAME:-}" ]]; then
      if [[ "${SIZE}" == "medium" && "${SHARDS}" -gt 1 ]]; then
        export SWEEP_NAME="${SIZE}_experimental_sweep_shard${SHARD}_of_${SHARDS}"
      else
        export SWEEP_NAME="${SIZE}_experimental_sweep"
      fi
    else
      export SWEEP_NAME
    fi
    ;;
  *)
    echo "unknown MODE=${MODE}; expected prep or sweep" >&2
    exit 2
    ;;
esac

echo "REPO_DIR=${REPO_DIR}"
echo "SIZE=${SIZE}"
echo "MODE=${MODE}"
echo "SHARD_INDEX=${SHARD_INDEX}"
echo "SHARD_COUNT=${SHARD_COUNT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

bash "${sweep_script}"
