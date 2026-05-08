#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sweep_common.sh"

SIZE="small"
SWEEP_NAME="${SWEEP_NAME:-small_experimental_sweep}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${SWEEP_NAME}.out}"
start_logging "${LOG_FILE}"

log "Small sweep: current-scale graphs, five consistent algorithm seeds (${SEEDS})."
log "GPU note: current NumPy/SciPy/NetworkX implementation is CPU-bound; GPUs will not accelerate this without porting eigensolvers and clustering kernels."

# Essential social sanity checks first.
generate_dataset "social karate_club" "$(dataset_path "${SIZE}" "social_karate_club")" social --source karate_club
generate_dataset "social davis_southern_women" "$(dataset_path "${SIZE}" "social_davis_southern_women")" social --source davis_southern_women
generate_dataset "social les_miserables" "$(dataset_path "${SIZE}" "social_les_miserables")" social --source les_miserables

# SBM assortativity: fixed size, varying p_in / p_out.
generate_dataset "sbm assortative strong" "$(dataset_path "${SIZE}" "sbm_assort_strong_40x3")" sbm --sizes 40,40,40 --p-in 0.30 --p-out 0.02 --seed 0
generate_dataset "sbm assortative medium" "$(dataset_path "${SIZE}" "sbm_assort_medium_40x3")" sbm --sizes 40,40,40 --p-in 0.30 --p-out 0.06 --seed 0
generate_dataset "sbm assortative weak" "$(dataset_path "${SIZE}" "sbm_assort_weak_40x3")" sbm --sizes 40,40,40 --p-in 0.30 --p-out 0.10 --seed 0

# SBM imbalance.
generate_dataset "sbm imbalance 20_80" "$(dataset_path "${SIZE}" "sbm_imbalance_20_80")" sbm --sizes 20,80 --p-in 0.28 --p-out 0.04 --seed 1
generate_dataset "sbm imbalance 20_40_80" "$(dataset_path "${SIZE}" "sbm_imbalance_20_40_80")" sbm --sizes 20,40,80 --p-in 0.25 --p-out 0.035 --seed 1

# LFR mixing sweep.
for mu in 05 10 20 30 40; do
  mu_value="0.${mu}"
  generate_dataset "lfr mixing mu=${mu_value}" "$(dataset_path "${SIZE}" "lfr_n100_mu${mu}")" lfr --n 100 --tau1 3.0 --tau2 1.5 --mu "${mu_value}" --average-degree 6 --max-degree 20 --min-community 8 --seed 2 --max-iters 1000
done

# LFR degree heterogeneity.
generate_dataset "lfr heterogeneity flatter" "$(dataset_path "${SIZE}" "lfr_n100_tau25_max30")" lfr --n 100 --tau1 2.5 --tau2 1.5 --mu 0.20 --average-degree 6 --max-degree 30 --min-community 8 --seed 3 --max-iters 1000
generate_dataset "lfr heterogeneity steeper" "$(dataset_path "${SIZE}" "lfr_n100_tau35_max15")" lfr --n 100 --tau1 3.5 --tau2 1.5 --mu 0.20 --average-degree 6 --max-degree 15 --min-community 8 --seed 3 --max-iters 1000

# Ring of cliques scale and noisy bridges.
generate_dataset "ring cliques 6x8 bridge1" "$(dataset_path "${SIZE}" "ring_6x8_b1")" ring_of_cliques --num-cliques 6 --clique-size 8 --bridge-edges 1
generate_dataset "ring cliques 8x6 bridge2" "$(dataset_path "${SIZE}" "ring_8x6_b2")" ring_of_cliques --num-cliques 8 --clique-size 6 --bridge-edges 2

# Core-periphery attachment sweep.
generate_dataset "core periphery sparse" "$(dataset_path "${SIZE}" "corep_10_5x6_a1")" core_periphery_cliques --core-size 10 --num-periphery 5 --periphery-size 6 --attachments-per-periphery 1
generate_dataset "core periphery denser" "$(dataset_path "${SIZE}" "corep_12_6x5_a2")" core_periphery_cliques --core-size 12 --num-periphery 6 --periphery-size 5 --attachments-per-periphery 2

# Disconnected / nearly disconnected.
generate_dataset "disconnected cliques" "$(dataset_path "${SIZE}" "disconnected_20_20_20")" disconnected_cliques --sizes 20,20,20 --bridge-edges 0
generate_dataset "nearly disconnected cliques" "$(dataset_path "${SIZE}" "nearly_disconnected_20_20_20")" disconnected_cliques --sizes 20,20,20 --bridge-edges 1

# Leaves and isolated vertices.
generate_dataset "sbm leaves isolates" "$(dataset_path "${SIZE}" "sbm_leaves_isolates_30x3")" sbm_with_leaves --sizes 30,30,30 --p-in 0.28 --p-out 0.04 --leaves-per-block 3 --isolated 6 --seed 4

run_sweep "${SWEEP_NAME}" "${OUTPUT_ROOT}"
log "DONE ${SWEEP_NAME}"
