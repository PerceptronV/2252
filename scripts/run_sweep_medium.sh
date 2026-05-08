#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sweep_common.sh"

SIZE="medium"
SWEEP_NAME="${SWEEP_NAME:-medium_experimental_sweep}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${SWEEP_NAME}.out}"
start_logging "${LOG_FILE}"

log "Medium sweep: hundreds of nodes, five consistent algorithm seeds (${SEEDS})."
log "GPU note: current NumPy/SciPy/NetworkX implementation is CPU-bound; GPUs will not accelerate this without porting eigensolvers and clustering kernels."

# Essential real/social graphs first.
generate_dataset "social les_miserables" "$(dataset_path "${SIZE}" "social_les_miserables")" social --source les_miserables
generate_dataset "social davis_southern_women" "$(dataset_path "${SIZE}" "social_davis_southern_women")" social --source davis_southern_women

# SBM assortativity.
generate_dataset "sbm assortative strong" "$(dataset_path "${SIZE}" "sbm_assort_strong_150x3")" sbm --sizes 150,150,150 --p-in 0.18 --p-out 0.01 --seed 0
generate_dataset "sbm assortative medium" "$(dataset_path "${SIZE}" "sbm_assort_medium_150x3")" sbm --sizes 150,150,150 --p-in 0.18 --p-out 0.035 --seed 0
generate_dataset "sbm assortative weak" "$(dataset_path "${SIZE}" "sbm_assort_weak_150x3")" sbm --sizes 150,150,150 --p-in 0.18 --p-out 0.07 --seed 0

# SBM imbalance.
generate_dataset "sbm imbalance 80_160" "$(dataset_path "${SIZE}" "sbm_imbalance_80_160")" sbm --sizes 80,160 --p-in 0.16 --p-out 0.025 --seed 1
generate_dataset "sbm imbalance 60_120_240" "$(dataset_path "${SIZE}" "sbm_imbalance_60_120_240")" sbm --sizes 60,120,240 --p-in 0.14 --p-out 0.02 --seed 1

# LFR mixing sweep.
for mu in 05 10 20 30 40; do
  mu_value="0.${mu}"
  generate_dataset "lfr mixing mu=${mu_value}" "$(dataset_path "${SIZE}" "lfr_n500_mu${mu}")" lfr --n 500 --tau1 3.0 --tau2 1.5 --mu "${mu_value}" --average-degree 10 --max-degree 80 --min-community 20 --seed 2 --max-iters 2000
done

# LFR degree heterogeneity.
generate_dataset "lfr heterogeneity flatter" "$(dataset_path "${SIZE}" "lfr_n500_tau25_max120")" lfr --n 500 --tau1 2.5 --tau2 1.5 --mu 0.20 --average-degree 10 --max-degree 120 --min-community 20 --seed 3 --max-iters 2000
generate_dataset "lfr heterogeneity steeper" "$(dataset_path "${SIZE}" "lfr_n500_tau35_max50")" lfr --n 500 --tau1 3.5 --tau2 1.5 --mu 0.20 --average-degree 10 --max-degree 50 --min-community 20 --seed 3 --max-iters 2000

# Ring of cliques scale and noisy bridges.
generate_dataset "ring cliques 20x25 bridge1" "$(dataset_path "${SIZE}" "ring_20x25_b1")" ring_of_cliques --num-cliques 20 --clique-size 25 --bridge-edges 1
generate_dataset "ring cliques 30x20 bridge3" "$(dataset_path "${SIZE}" "ring_30x20_b3")" ring_of_cliques --num-cliques 30 --clique-size 20 --bridge-edges 3

# Core-periphery attachment sweep.
generate_dataset "core periphery sparse" "$(dataset_path "${SIZE}" "corep_80_20x20_a1")" core_periphery_cliques --core-size 80 --num-periphery 20 --periphery-size 20 --attachments-per-periphery 1
generate_dataset "core periphery denser" "$(dataset_path "${SIZE}" "corep_100_30x15_a3")" core_periphery_cliques --core-size 100 --num-periphery 30 --periphery-size 15 --attachments-per-periphery 3

# Disconnected / nearly disconnected.
generate_dataset "disconnected cliques" "$(dataset_path "${SIZE}" "disconnected_150_150_150")" disconnected_cliques --sizes 150,150,150 --bridge-edges 0
generate_dataset "nearly disconnected cliques" "$(dataset_path "${SIZE}" "nearly_disconnected_150_150_150")" disconnected_cliques --sizes 150,150,150 --bridge-edges 2

# Leaves and isolated vertices.
generate_dataset "sbm leaves isolates" "$(dataset_path "${SIZE}" "sbm_leaves_isolates_120x3")" sbm_with_leaves --sizes 120,120,120 --p-in 0.16 --p-out 0.025 --leaves-per-block 20 --isolated 30 --seed 4

run_sweep "${SWEEP_NAME}" "${OUTPUT_ROOT}"
log "DONE ${SWEEP_NAME}"
