# Dataset Bundles

This directory contains graph bundles generated or imported by `scripts/datasets.py`.

## Bundle format

Each `.npz` file stores:

- a CSR adjacency matrix
- a `target` label vector
- `num_clusters`
- graph metadata

These bundles are loaded by the `serialized_graph` dataset in `src/data/serialized.py`.

## Generated samples

- `synthetic/sbm_40_40_40_seed0.npz` — 3-block stochastic block model
- `synthetic/lfr_n80_mu010_seed0.npz` — LFR benchmark graph
- `synthetic/ring_of_cliques_6x8.npz` — six 8-node cliques connected in a ring
- `synthetic/core_periphery_10_5x6.npz` — dense core with attached periphery cliques
- `social/karate_club.npz` — builtin Karate Club graph with club labels
- `social/karate_club_from_edgelist.npz` — same graph imported through the generic edge-list parser

## Generate more

```bash
python3 scripts/datasets.py sbm --sizes 40,40,40 --p-in 0.25 --p-out 0.03 --seed 0 --output datasets/synthetic/sbm_40_40_40_seed0.npz
PYTHONPATH=.venv/lib/python3.13/site-packages python3 scripts/datasets.py lfr --n 80 --tau1 3.0 --tau2 1.8 --mu 0.1 --min-degree 3 --min-community 10 --max-iters 5000 --seed 0 --output datasets/synthetic/lfr_n80_mu010_seed0.npz
python3 scripts/datasets.py ring_of_cliques --num-cliques 6 --clique-size 8 --output datasets/synthetic/ring_of_cliques_6x8.npz
python3 scripts/datasets.py core_periphery_cliques --core-size 10 --num-periphery 5 --periphery-size 6 --output datasets/synthetic/core_periphery_10_5x6.npz
PYTHONPATH=.venv/lib/python3.13/site-packages python3 scripts/datasets.py social --source karate_club --output datasets/social/karate_club.npz
python3 scripts/datasets.py edgelist --input path/to/graph.edgelist --labels path/to/labels.txt --output datasets/social/custom_graph.npz
```

## Sweep benchmarks

```bash
python3 scripts/sweep_benchmarks.py \
  --datasets datasets/synthetic/*.npz,datasets/social/karate_club.npz \
  --ks 2,3,4,5,6 \
  --seeds 0
```

This writes:

- per-run result directories under `results/sweeps/<name>/runs/`
- generated configs under `results/sweeps/<name>/configs/`
- flat logs in `summary.jsonl`, `summary.csv`, and `failures.jsonl`
