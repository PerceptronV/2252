#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sweep_common.sh"

SIZE="large"
SWEEP_NAME_BASE="${SWEEP_NAME:-large_experimental_sweep}"
if (( SHARD_COUNT > 1 )); then
  SWEEP_NAME="${SWEEP_NAME_BASE}_shard${SHARD_INDEX}_of_${SHARD_COUNT}"
else
  SWEEP_NAME="${SWEEP_NAME_BASE}"
fi
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${SWEEP_NAME}.out}"
start_logging "${LOG_FILE}"

log "Large sweep: thousands of nodes, five consistent algorithm seeds (${SEEDS})."
log "Sharding: SHARD_INDEX=${SHARD_INDEX}, SHARD_COUNT=${SHARD_COUNT}. Use zero-based shard indexes on separate nodes."
log "GPU note: current NumPy/SciPy/NetworkX implementation is CPU-bound; GPUs will not accelerate this without porting eigensolvers and clustering kernels."

# Essential real/social reference first.
generate_dataset "social les_miserables" "$(dataset_path "${SIZE}" "social_les_miserables")" social --source les_miserables

# SBM assortativity.
generate_dataset "sbm assortative strong" "$(dataset_path "${SIZE}" "sbm_assort_strong_600x3")" sbm --sizes 600,600,600 --p-in 0.08 --p-out 0.004 --seed 0
generate_dataset "sbm assortative medium" "$(dataset_path "${SIZE}" "sbm_assort_medium_600x3")" sbm --sizes 600,600,600 --p-in 0.08 --p-out 0.015 --seed 0
generate_dataset "sbm assortative weak" "$(dataset_path "${SIZE}" "sbm_assort_weak_600x3")" sbm --sizes 600,600,600 --p-in 0.08 --p-out 0.03 --seed 0

# SBM imbalance.
generate_dataset "sbm imbalance 400_800" "$(dataset_path "${SIZE}" "sbm_imbalance_400_800")" sbm --sizes 400,800 --p-in 0.07 --p-out 0.012 --seed 1
generate_dataset "sbm imbalance 300_600_1200" "$(dataset_path "${SIZE}" "sbm_imbalance_300_600_1200")" sbm --sizes 300,600,1200 --p-in 0.06 --p-out 0.01 --seed 1

# LFR mixing sweep.
# NetworkX's LFR generator is brittle at n=2000 with low mu and a very high
# max degree; n=1500 keeps this in the large-graph regime while avoiding
# repeated ExceededMaxIterations failures during prep.
for mu in 05 10 20 30 40; do
  mu_value="0.${mu}"
  generate_dataset "lfr mixing mu=${mu_value}" "$(dataset_path "${SIZE}" "lfr_n1500_mu${mu}")" lfr --n 1500 --tau1 3.0 --tau2 1.5 --mu "${mu_value}" --average-degree 12 --max-degree 120 --min-community 50 --seed 2 --max-iters 3000
done

# LFR degree heterogeneity.
generate_dataset "lfr heterogeneity flatter" "$(dataset_path "${SIZE}" "lfr_n1500_tau25_max180")" lfr --n 1500 --tau1 2.5 --tau2 1.5 --mu 0.20 --average-degree 12 --max-degree 180 --min-community 50 --seed 3 --max-iters 3000
generate_dataset "lfr heterogeneity steeper" "$(dataset_path "${SIZE}" "lfr_n1500_tau35_max80")" lfr --n 1500 --tau1 3.5 --tau2 1.5 --mu 0.20 --average-degree 12 --max-degree 80 --min-community 50 --seed 3 --max-iters 3000

# Ring of cliques scale and noisy bridges.
generate_dataset "ring cliques 50x40 bridge1" "$(dataset_path "${SIZE}" "ring_50x40_b1")" ring_of_cliques --num-cliques 50 --clique-size 40 --bridge-edges 1
generate_dataset "ring cliques 80x30 bridge4" "$(dataset_path "${SIZE}" "ring_80x30_b4")" ring_of_cliques --num-cliques 80 --clique-size 30 --bridge-edges 4

# Core-periphery attachment sweep.
generate_dataset "core periphery sparse" "$(dataset_path "${SIZE}" "corep_250_50x35_a1")" core_periphery_cliques --core-size 250 --num-periphery 50 --periphery-size 35 --attachments-per-periphery 1
generate_dataset "core periphery denser" "$(dataset_path "${SIZE}" "corep_300_70x30_a4")" core_periphery_cliques --core-size 300 --num-periphery 70 --periphery-size 30 --attachments-per-periphery 4

# Disconnected / nearly disconnected.
generate_dataset "disconnected cliques" "$(dataset_path "${SIZE}" "disconnected_600_600_600")" disconnected_cliques --sizes 600,600,600 --bridge-edges 0
generate_dataset "nearly disconnected cliques" "$(dataset_path "${SIZE}" "nearly_disconnected_600_600_600")" disconnected_cliques --sizes 600,600,600 --bridge-edges 3

# Leaves and isolated vertices.
generate_dataset "sbm leaves isolates" "$(dataset_path "${SIZE}" "sbm_leaves_isolates_500x3")" sbm_with_leaves --sizes 500,500,500 --p-in 0.07 --p-out 0.012 --leaves-per-block 75 --isolated 100 --seed 4

run_sweep "${SWEEP_NAME}" "${OUTPUT_ROOT}"
log "DONE ${SWEEP_NAME}"
