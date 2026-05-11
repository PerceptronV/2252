#!/usr/bin/env python3
"""Plot and summarize the three small social graph results."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

REPO_DIR = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = REPO_DIR / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DATASETS = [
    ("social_karate_club", "Karate Club"),
    ("social_davis_southern_women", "Davis Southern Women"),
    ("social_les_miserables", "Les Miserables"),
]

METRICS = [
    ("ari", "ARI", "higher is better"),
    ("nmi", "NMI", "higher is better"),
    ("modularity", "Modularity", "higher is better"),
    ("max_conductance", "Max conductance", "lower is better"),
]

ALGORITHMS = [
    "kmeans",
    "peng_kmeans",
    "fiedler",
    "shi_malik",
    "markov",
    "louvain",
]

LABELS = {
    "kmeans": "Spectral k-means",
    "peng_kmeans": "Peng weighted k-means",
    "fiedler": "Fiedler",
    "shi_malik": "Shi-Malik",
    "markov": "MCL",
    "louvain": "Louvain",
}

COLORS = {
    "kmeans": "#1f77b4",
    "peng_kmeans": "#ff7f0e",
    "fiedler": "#2ca02c",
    "shi_malik": "#d62728",
    "markov": "#9467bd",
    "louvain": "#8c564b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=REPO_DIR / "results" / "sweeps" / "small_experimental_sweep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_DIR / "results" / "plots" / "current_sweeps",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def parse_time(value: str | None, fallback_mtime: float) -> datetime:
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback_mtime, tz=timezone.utc)


def parse_config(config_path: str) -> tuple[str, int, int, str] | None:
    stem = Path(config_path).stem
    match = re.match(r"(?P<dataset>.+)__k(?P<k>\d+)__seed(?P<seed>\d+)__(?P<algorithm>.+)$", stem)
    if not match:
        return None
    return (
        match.group("dataset"),
        int(match.group("k")),
        int(match.group("seed")),
        match.group("algorithm"),
    )


def read_social_runs(sweep_dir: Path) -> list[dict]:
    social_names = {name for name, _ in DATASETS}
    latest: dict[str, tuple[datetime, dict]] = {}
    for scores_path in (sweep_dir / "runs").glob("*/scores.json"):
        metadata_path = scores_path.with_name("metadata.json")
        if not metadata_path.exists():
            continue
        scores = load_json(scores_path)
        metadata = load_json(metadata_path)
        parsed = parse_config(str(metadata.get("config_path", "")))
        if parsed is None:
            continue
        dataset, k, seed, algorithm = parsed
        if dataset not in social_names:
            continue
        timestamp = parse_time(
            metadata.get("timestamp_utc"),
            max(scores_path.stat().st_mtime, metadata_path.stat().st_mtime),
        )
        record = {
            "dataset": dataset,
            "k": k,
            "seed": seed,
            "algorithm": algorithm,
            "scores": scores,
        }
        key = Path(str(metadata["config_path"])).stem
        previous = latest.get(key)
        if previous is None or timestamp >= previous[0]:
            latest[key] = (timestamp, record)
    return [record for _, record in latest.values()]


def aggregate(records: list[dict]) -> dict[tuple[str, int, str, str], float]:
    values: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for record in records:
        for metric, _, _ in METRICS + [("returned_num_clusters", "", "")]:
            value = record["scores"].get(metric)
            if value is not None and math.isfinite(float(value)):
                values[(record["dataset"], record["k"], record["algorithm"], metric)].append(float(value))
    return {key: mean(metric_values) for key, metric_values in values.items()}


def plot_metrics(agg: dict, output_dir: Path) -> Path:
    fig, axes = plt.subplots(len(DATASETS), len(METRICS), figsize=(18, 11), squeeze=False)
    for row, (dataset, dataset_label) in enumerate(DATASETS):
        for col, (metric, metric_label, metric_note) in enumerate(METRICS):
            ax = axes[row][col]
            for algorithm in ALGORITHMS:
                points = sorted(
                    (k, value)
                    for (key_dataset, k, key_algorithm, key_metric), value in agg.items()
                    if key_dataset == dataset and key_algorithm == algorithm and key_metric == metric
                )
                if not points:
                    continue
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    linewidth=1.9,
                    markersize=3.5,
                    color=COLORS[algorithm],
                    label=LABELS[algorithm],
                )
            if row == 0:
                ax.set_title(f"{metric_label}\n({metric_note})")
            if col == 0:
                ax.set_ylabel(dataset_label)
            ax.set_xlabel("Requested k")
            ax.grid(True, alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=False)
    fig.suptitle("Small social graph results", fontsize=17)
    fig.text(
        0.5,
        0.008,
        "MCL and Louvain ignore requested k; use the returned-cluster plot to interpret their flat metric curves.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.085, 1, 0.94])
    out_path = output_dir / "social_graphs_metrics.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_returned_clusters(agg: dict, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 4.2), squeeze=False)
    for col, (dataset, dataset_label) in enumerate(DATASETS):
        ax = axes[0][col]
        for algorithm in ALGORITHMS:
            points = sorted(
                (k, value)
                for (key_dataset, k, key_algorithm, key_metric), value in agg.items()
                if key_dataset == dataset and key_algorithm == algorithm and key_metric == "returned_num_clusters"
            )
            if not points:
                continue
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=1.9,
                markersize=3.5,
                color=COLORS[algorithm],
                label=LABELS[algorithm],
            )
        ax.set_title(dataset_label)
        ax.set_xlabel("Requested k")
        ax.set_ylabel("Returned clusters")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
    fig.suptitle("Returned cluster counts on social graphs", fontsize=15)
    fig.tight_layout(rect=[0, 0.12, 1, 0.9])
    out_path = output_dir / "social_graphs_returned_clusters.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def metric_value(agg: dict, dataset: str, k: int, algorithm: str, metric: str) -> float:
    return agg[(dataset, k, algorithm, metric)]


def write_analysis(agg: dict, output_dir: Path) -> Path:
    lines = [
        "# Social Graph Results",
        "",
        "All values are averaged over the five generated seed configs. MCL and Louvain are deterministic/free-granularity baselines here, so their curves are flat because they ignore requested `k`.",
        "",
        "## Karate Club",
        "",
        f"- At `k=2`, Fiedler and Shi-Malik are strongest on label recovery: ARI `{metric_value(agg, 'social_karate_club', 2, 'fiedler', 'ari'):.3f}`, NMI `{metric_value(agg, 'social_karate_club', 2, 'fiedler', 'nmi'):.3f}`, max conductance `{metric_value(agg, 'social_karate_club', 2, 'fiedler', 'max_conductance'):.3f}`.",
        f"- This is expected: Karate Club has a dominant two-way social split, so a sweep-cut style method matches the natural bottleneck.",
        f"- Louvain returns `{metric_value(agg, 'social_karate_club', 2, 'louvain', 'returned_num_clusters'):.0f}` clusters, has the highest modularity (`{metric_value(agg, 'social_karate_club', 2, 'louvain', 'modularity'):.3f}`), but lower ARI (`{metric_value(agg, 'social_karate_club', 2, 'louvain', 'ari'):.3f}`) because it over-partitions the known two-faction labels.",
        f"- MCL returns `{metric_value(agg, 'social_karate_club', 2, 'markov', 'returned_num_clusters'):.0f}` clusters and matches the spectral k-means label score at ARI `{metric_value(agg, 'social_karate_club', 2, 'markov', 'ari'):.3f}`.",
        "",
        "## Davis Southern Women",
        "",
        f"- MCL is best on ARI/NMI across requested `k`: ARI `{metric_value(agg, 'social_davis_southern_women', 2, 'markov', 'ari'):.3f}`, NMI `{metric_value(agg, 'social_davis_southern_women', 2, 'markov', 'nmi'):.3f}`, returning `{metric_value(agg, 'social_davis_southern_women', 2, 'markov', 'returned_num_clusters'):.0f}` clusters.",
        f"- This does not mean MCL is structurally best: its modularity is negative (`{metric_value(agg, 'social_davis_southern_women', 2, 'markov', 'modularity'):.3f}`) and max conductance is high (`{metric_value(agg, 'social_davis_southern_women', 2, 'markov', 'max_conductance'):.3f}`).",
        "- The reason is label semantics: Davis labels are bipartite node type, not a standard community partition. MCL's flow partition happens to align better with those labels, while modularity-based partitions do not.",
        f"- Louvain has the best modularity (`{metric_value(agg, 'social_davis_southern_women', 2, 'louvain', 'modularity'):.3f}`) but near-zero/negative ARI (`{metric_value(agg, 'social_davis_southern_women', 2, 'louvain', 'ari'):.3f}`), reinforcing that the reference labels and graph-community objective disagree.",
        "",
        "## Les Miserables",
        "",
        f"- Louvain dominates label recovery and modularity: ARI `{metric_value(agg, 'social_les_miserables', 2, 'louvain', 'ari'):.3f}`, NMI `{metric_value(agg, 'social_les_miserables', 2, 'louvain', 'nmi'):.3f}`, modularity `{metric_value(agg, 'social_les_miserables', 2, 'louvain', 'modularity'):.3f}`, returning `{metric_value(agg, 'social_les_miserables', 2, 'louvain', 'returned_num_clusters'):.0f}` clusters.",
        "- This is expected because the reference labels for Les Miserables were generated by greedy modularity communities. Louvain is being evaluated against labels produced by a closely related objective.",
        f"- MCL returns `{metric_value(agg, 'social_les_miserables', 2, 'markov', 'returned_num_clusters'):.0f}` clusters and has decent NMI (`{metric_value(agg, 'social_les_miserables', 2, 'markov', 'nmi'):.3f}`), but lower ARI and modularity than Louvain.",
        f"- Fiedler is best among the explicitly cut-style methods at low `k` on this graph: at `k=2`, ARI `{metric_value(agg, 'social_les_miserables', 2, 'fiedler', 'ari'):.3f}` versus spectral k-means ARI `{metric_value(agg, 'social_les_miserables', 2, 'kmeans', 'ari'):.3f}`.",
        "",
        "## Slide Takeaway",
        "",
        "- Karate Club supports the cut-based story: the dominant sparse bottleneck is well captured by Fiedler/Shi-Malik.",
        "- Davis is a warning that reference labels may not be graph communities; MCL scores better on labels but poorly on modularity/max conductance.",
        "- Les Miserables favors Louvain because the labels are modularity-derived, so the evaluation objective is aligned with Louvain's optimization target.",
    ]
    out_path = output_dir / "social_graphs_analysis.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main() -> int:
    args = parse_args()
    sweep_dir = args.sweep_dir if args.sweep_dir.is_absolute() else REPO_DIR / args.sweep_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_social_runs(sweep_dir)
    agg = aggregate(records)
    metrics_path = plot_metrics(agg, output_dir)
    clusters_path = plot_returned_clusters(agg, output_dir)
    analysis_path = write_analysis(agg, output_dir)
    print(f"Loaded {len(records)} social configs.")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {clusters_path}")
    print(f"Wrote {analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
