# Computational Section Slide Outline

This section should directly follow the algorithm slides. The goal is to shift from "what each algorithm does" to "what we are testing, on which graphs, and by which measurements."

## Slide 1: Experimental Objective

**Main question**

How do spectral graph clustering methods behave across graph regimes where the planted communities, low-conductance cuts, degree structure, and requested number of clusters do not always agree?

**Experiment objective**

- Compare the algorithms on a common set of graph instances, cluster counts, random seeds, and evaluation metrics.
- Separate two notions of success:
  - **Label recovery:** does the algorithm recover the intended or planted communities?
  - **Graph cut quality:** does the algorithm return low-boundary clusters according to the graph itself?
- Identify regimes where spectral embedding plus k-means, weighted Peng-style k-means, recursive cut methods, and non-spectral baselines disagree.

**Connection to previous algorithm slides**

The algorithms optimize or approximate different objectives, so we should not expect a single winner on every graph. The computation section asks when those objective differences become visible.

## Slide 2: Shared Experiment Pipeline

**Common pipeline**

1. Load each graph as a sparse undirected adjacency matrix.
2. Attach reference labels when the dataset has planted or known communities.
3. Run each algorithm under the same requested cluster count `k`.
4. Evaluate predicted labels using the same metric suite.
5. Store per-run metadata, scores, runtime, and failures in flat summary files.

**Sweep design**

- Requested cluster counts: `k in {2, 3, 4, 6, 8}` for the current sweep scripts.
- Random seeds: `0, 1, 2, 3, 4` for randomized methods.
- Spectral embedding dimension: set to the requested `k` for the k-means spectral baselines.
- k-means settings: k-means++ initialization, `n_init = 10`, `max_iter = 300`.
- Current implementation stack: Python, NumPy/SciPy sparse matrices, ARPACK-style eigensolvers, NetworkX dataset generation.

**Why this matters**

Keeping the runner fixed lets us attribute differences to algorithmic behavior rather than dataset parsing, metric implementation, or inconsistent hyperparameters.

## Slide 3: Algorithms Included in the Computation

**Algorithms evaluated**

- **Spectral + k-means:** standard spectral embedding followed by unweighted k-means.
- **Peng-style weighted k-means:** degree-aware spectral embedding and weighted clustering objective.
- **Fiedler recursive cut:** repeated two-way sweep cuts from the Fiedler vector.
- **Shi-Malik normalized cut:** normalized-cut spectral partitioning using the random-walk normalization.
- **Markov clustering:** flow-based clustering through expansion and inflation.
- **Louvain:** modularity maximization without a fixed requested `k`.

**What this comparison is designed to expose**

- Spectral embedding methods test whether geometry in the bottom eigenspace is enough.
- Recursive cut methods test whether explicit low-conductance bisection is more stable.
- Louvain and Markov clustering provide non-spectral baselines with different implicit objectives.
- Algorithms that do not naturally enforce the requested `k` are evaluated with `returned_num_clusters` to make that mismatch visible.

## Slide 4: Dataset Families

**Synthetic block models**

- **Balanced SBM:** equal-size planted communities with controlled within-block and cross-block probabilities.
- **Imbalanced SBM:** planted communities with unequal sizes, testing sensitivity to cluster volume.
- **Weak/medium/strong assortativity SBM:** fixed block sizes with increasing cross-community noise.
- **SBM with leaves and isolates:** planted communities plus low-degree and isolated vertices, testing robustness to degree pathologies.
- **What this family tests:** whether each method can recover known planted structure as we vary separation strength, balance, and low-degree noise.
- **Why it matters:** SBM graphs give us the cleanest way to distinguish true recovery failures from ambiguity in the dataset.

**Benchmark community graphs**

- **LFR graphs:** heterogeneous degree and community-size distributions with controlled mixing parameter `mu`.
- **LFR mixing sweep:** increasing `mu` makes communities less separated.
- **LFR degree heterogeneity sweep:** changes the degree exponent and maximum degree while holding the mixing regime comparable.
- **What this family tests:** whether the algorithms remain stable when degree distributions and community sizes look more like real networks.
- **Why it matters:** LFR graphs are where degree-aware methods should show an advantage if weighting is actually helping the spectral embedding.

**Structured stress tests**

- **Ring of cliques:** many dense local communities connected by sparse bridges.
- **Core-periphery cliques:** dense core plus attached peripheral cliques, emphasizing degree heterogeneity.
- **Disconnected / nearly disconnected cliques:** sanity checks for exact or almost-exact cluster separation.
- **What this family tests:** how algorithms behave on graphs with obvious low-conductance pieces, repeated bottlenecks, or hub-dominated structure.
- **Why it matters:** these cases expose when cut quality and planted-label recovery separate from each other.

**Small real-world references**

- **Karate Club:** known two-way social split.
- **Davis Southern Women:** bipartite social affiliation graph.
- **Les Miserables:** character co-appearance graph with modularity-derived reference labels.
- **What this family tests:** whether trends from synthetic graphs survive on small, interpretable real networks.
- **Why it matters:** these graphs provide sanity checks, but the labels are less controlled than in planted synthetic data.

## Slide 5: Dataset Regimes and What They Test

| Dataset regime | What changes | What it tests |
|---|---:|---|
| SBM assortativity | `p_out` increases | Recovery as planted communities become less separated |
| SBM imbalance | block sizes differ | Whether methods over-split or ignore small communities |
| LFR mixing | `mu` increases | Robustness to noisy, realistic community boundaries |
| LFR heterogeneity | degree distribution changes | Whether degree-aware methods stabilize the embedding |
| Ring of cliques | sparse bridges between cliques | Sensitivity to repeated low-conductance local structure |
| Core-periphery cliques | core degree dominates | Whether hubs distort spectral geometry or modularity |
| Disconnected cliques | bridge count is 0 or small | Sanity check for easy separability and near-separability |
| Social graphs | real graph structure | Whether conclusions survive outside planted synthetic data |

**Connection**

These regimes deliberately separate ground-truth recovery from cut quality. For example, over-segmenting a planted block may improve conductance while hurting ARI.

## Slide 6: Metrics We Track

| Metric | One-line meaning |
|---|---|
| **Adjusted Rand Index (ARI)** | Tracks pairwise agreement between predicted clusters and reference labels, corrected for chance. |
| **Normalized Mutual Information (NMI)** | Tracks shared information between predicted clusters and reference labels, with label-name invariance. |
| **Mean conductance** | Tracks average one-vs-rest boundary quality of the returned clusters; lower is better. |
| **Minimum conductance** | Tracks the best-separated predicted cluster. |
| **Maximum conductance** | Tracks the worst-separated predicted cluster. |
| **Modularity** | Tracks how much predicted communities exceed degree-preserving random-graph edge density. |
| **Returned number of clusters** | Tracks whether the algorithm actually returns the requested number of communities. |
| **Runtime seconds** | Tracks wall-clock algorithm time for scalability comparisons. |

## Slide 7: Reading the Metrics Together

**Recovery metrics**

- ARI and NMI answer: "Did we find the reference communities?"
- These are most meaningful on synthetic graphs and real graphs with interpretable labels.

**Graph-structural metrics**

- Conductance and modularity answer: "Did we find graph communities according to internal connectivity?"
- These can disagree with labels when the requested `k` is wrong, when labels are coarse, or when the graph contains smaller low-conductance pieces.

**Diagnostic metrics**

- Min/max conductance expose whether the average hides one bad cluster.
- Returned cluster count is essential for Louvain and Markov clustering because their natural granularity may differ from the requested `k`.
- Runtime is useful mainly after larger sweeps finish; small graphs are not enough for strong scaling claims.

**Key interpretation rule**

High ARI with worse conductance suggests label-faithful recovery; low ARI with better conductance suggests the algorithm found graph cuts that do not match the reference labeling.

## Slide 8: Expected Comparisons Before Results

**Questions to answer once final runs finish**

- Does degree weighting help most on LFR and core-periphery graphs?
- Do recursive cut methods perform best when the dominant structure is a single sparse bottleneck?
- Do geometric spectral methods over-segment planted blocks when `k` exceeds the true number of communities?
- Do Louvain and Markov clustering return a natural number of communities different from the requested `k`?
- Do conductance and ARI disagree most strongly on imbalanced, noisy, or over-requested settings?

**Planned result layout**

- One table summarizing average metric ranks by dataset family.
- One plot for LFR mixing: metric trends as `mu` increases.
- One plot for SBM over-requesting: behavior as requested `k` exceeds planted `k`.
- One plot for runtime: algorithm time versus graph size after medium/large sweeps complete.
- One small case-study slide for any regime where ARI and conductance point in opposite directions.

## Slide 9: Placeholder Result Slide Template

**Dataset family: `[name]`**

**Setup**

- Graphs: `[instances]`
- Requested clusters: `[k values]`
- Seeds: `[seed count]`
- Algorithms: `[algorithm list]`

**Main observation**

- `[One sentence: which methods separate and on which metric.]`

**Metric pattern**

- ARI/NMI: `[label recovery pattern]`
- Conductance: `[cut-quality pattern]`
- Returned clusters: `[granularity pattern, if relevant]`
- Runtime: `[only if large enough to interpret]`

**Takeaway**

`[One sentence connecting the observation back to the algorithm objectives.]`

## Slide 10: Computation Section Closing

**What the experiments are meant to establish**

- The same graph can reward different algorithms depending on whether we measure label recovery, cut quality, modularity, or scalability.
- Degree heterogeneity, cluster imbalance, and requested `k` are the main axes where methods should separate.
- The computation section should therefore be read as an objective-alignment study, not only as a leaderboard.

**Bridge to final discussion**

Once the full results are available, the main discussion should focus on where the metrics disagree and what those disagreements say about the assumptions behind each algorithm.
