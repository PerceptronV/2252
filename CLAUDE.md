# CS 2252 — Graph Partitioning Benchmark

A modular research codebase for studying graph-partitioning algorithms. Backs a
CS 2252 course project / NeurIPS 2026 submission (paper draft in `docs/main.tex`).

## Aim

Compare many algorithms across many datasets and metrics with as little
boilerplate as possible. The codebase is plug-and-play along three axes:

- **Dataset** — a `(graph, target_partition)` source (SBM, LFR, Karate, Cora, …)
- **Algorithm** — takes `(graph, k)` and returns predicted cluster labels
  (spectral clustering, Cheeger sweep cut, …)
- **Baseline** — a graph-aware partition extractor over an embedding
  (`k`-means, `peng_kmeans`, sweep-cut rounding, …)
- **Eval** — takes `(graph, predicted, target)` and returns a scalar
  (conductance, ARI, …)

A single YAML config under `configs/` selects one dataset, one algorithm, and a
**list** of evals. The runner instantiates them by name and writes results to
`results/<run_uuid>/` with full metadata for later comparison.

## Structure

```
2252/
├── configs/                    # YAML experiment configs (single source of truth)
│   ├── example.yaml            #   template referencing not-yet-implemented types
│   └── _smoke.yaml             #   runnable smoke test against built-in stubs
├── src/                        # importable Python package (src layout)
│   ├── core/
│   │   ├── graph.py            #   Graph dataclass (sparse adjacency + optional features)
│   │   ├── registry.py         #   name -> class dicts + @register_* decorators
│   │   ├── config.py           #   YAML schema / loader
│   │   ├── runner.py           #   orchestrator: config in, run_dir out
│   │   └── results.py          #   writes config copy, metadata.json, scores.json, predictions.npz
│   ├── data/base.py            #   Dataset ABC (+ _TrivialDataset stub)
│   ├── baselines/
│   │   ├── base.py             #   Baseline ABC (+ _AllZerosBaseline stub)
│   │   └── kmeans.py           #   naive + Peng-style pure-NumPy k-means baselines
│   ├── algorithms/
│   │   ├── base.py             #   Algorithm ABC (+ _AllZerosAlgorithm stub)
│   │   ├── _spectral.py        #   bottom-k eigendecomp of L_sym, shared utility
│   │   └── spectral.py         #   spectral clustering + pluggable baseline
│   └── evals/base.py           #   Eval ABC (+ _LabelAccuracyEval stub)
├── scripts/run.py              # CLI: python scripts/run.py configs/<name>.yaml
├── results/<run_uuid>/         # gitignored per-run output dirs
├── docs/main.tex               # paper draft
├── prompts/                    # session logs, prompt scratch
└── pyproject.toml              # src layout, optional `pip install -e .`
```

## How to run an experiment

```bash
python scripts/run.py configs/_smoke.yaml
# -> run complete: results/<uuid>
```

Each run dir contains:
- `config.yaml` — verbatim copy of the input config
- `metadata.json` — `run_uuid`, ISO timestamp, git SHA + dirty flag, hostname,
  Python version, package versions, CLI args, graph SHA-256
- `scores.json` — `{eval_name: float}` for every eval listed in the config
- `predictions.npz` — `predicted` and `target` label arrays

## Adding a new component

Each axis has the same recipe:

1. Create `src/<axis>/<your_module>.py` with a class subclassing the ABC.
2. Decorate it: `@register_dataset("sbm")` / `@register_algorithm("spectral")` /
   `@register_baseline("kmeans")` / `@register_eval("conductance")`. The
   decorator records the class under that key and sets `cls.name`.
3. Import the new module from `src/<axis>/__init__.py` so registration runs on
   package import.
4. Reference it from a config: `type: <key>`, with `params:` forwarded as
   `**kwargs` to `__init__`.

The runner is decoupled from concrete subclasses — it only resolves names
through the registries. No runner changes are needed when new components land.

## Conventions

- **Graph type**: always the `core.graph.Graph` dataclass (frozen). Adjacency
  is `scipy.sparse.csr_matrix`, symmetric, zero diagonal. `node_features` is
  optional (e.g. for Cora). NetworkX is a dep but not the canonical type — use
  it only inside dataset loaders that need it.
- **Cluster labels**: `np.ndarray` of shape `(n,)` with integer labels. No
  identification across `predicted` / `target` is assumed (use ARI, not
  accuracy, when comparing them).
- **Eval signature**: every `Eval.__call__(graph, predicted, target)` takes all
  three even if some are unused (conductance ignores `target`, ARI ignores
  `graph`). Keeps the runner uniform.
- **Baseline signature**: every `Baseline.fit_predict(graph, embedding, k)`
  receives both the graph and the spectral embedding so geometric clusterers
  and graph-aware roundings can share one modular interface.
- **Spectral normalization**: `algorithms.spectral` supports `normalization:
  auto | unit_row | degree_sqrt | none`. `auto` picks `unit_row` for naive
  `kmeans` and `degree_sqrt` for `peng_kmeans`, matching the Peng-Sun-Zanetti
  embedding `F(u) = (f_1(u), ..., f_k(u)) / sqrt(d_u)`.
- **Stubs**: classes prefixed with `_` (e.g. `_TrivialDataset`) exist only for
  smoke testing the runner. Do not extend them; add real components instead.
- **Imports**: code under `src/` imports as `from core.X import …`,
  `from data.base import …`, etc. — `src/` is the package root, not a package.
  The CLI prepends `src/` to `sys.path`; `pip install -e .` works too.

## Verification

- Spectral utility sanity check (matches output from before the move):
  `PYTHONPATH=src python3 -m algorithms._spectral` →
  `Bottom-3 eigenvalues: [0. 0.004794 1.002632]`, sign split `20 20`.
- Runner end-to-end: `python scripts/run.py configs/_smoke.yaml` writes a
  populated run dir.
- Bad config key surfaces a clear error: an unknown `type:` raises
  `KeyError("unknown registry key 'foo'; available: …")`.
