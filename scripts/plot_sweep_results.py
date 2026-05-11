#!/usr/bin/env python3
"""Plot completed small/medium sweep results.

The sweep runner writes summary files only at the end, but each completed run
already has scores.json and metadata.json. This script reads those per-run
artifacts, de-duplicates restarted configs by keeping the latest completed run,
and plots whatever is available so far.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable

REPO_DIR = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = REPO_DIR / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRICS = [
    ("ari", "ARI", "higher is better"),
    ("nmi", "NMI", "higher is better"),
    ("max_conductance", "Max conductance", "lower is better"),
    ("modularity", "Modularity", "higher is better"),
]

SINGLE_DATASET_METRICS = [
    ("ari", "ARI", "higher is better"),
    ("nmi", "NMI", "higher is better"),
    ("modularity", "Modularity", "higher is better"),
    ("max_conductance", "Max conductance", "lower is better"),
]

ALL_METRIC_KEYS = sorted(
    {metric for metric, _, _ in METRICS + SINGLE_DATASET_METRICS}
    | {"returned_num_clusters"}
)

ALGORITHM_LABELS = {
    "kmeans": "Spectral k-means",
    "peng_kmeans": "Peng weighted k-means",
    "fiedler": "Fiedler",
    "shi_malik": "Shi-Malik",
    "markov": "MCL",
    "louvain": "Louvain",
}

ALGORITHM_ORDER = [
    "kmeans",
    "peng_kmeans",
    "fiedler",
    "shi_malik",
    "markov",
    "louvain",
]

COLORS = {
    "kmeans": "#1f77b4",
    "peng_kmeans": "#ff7f0e",
    "fiedler": "#2ca02c",
    "shi_malik": "#d62728",
    "markov": "#9467bd",
    "louvain": "#8c564b",
    "small": "#1f77b4",
    "medium": "#ff7f0e",
}

EXPECTED_TOTALS = {
    "small": 22 * 5 * 5 * 6,
    "medium_shard0": 11 * 5 * 5 * 6,
    "medium_shard1": 10 * 5 * 5 * 6,
    "medium": 21 * 5 * 5 * 6,
}


@dataclass(frozen=True)
class RunRecord:
    sweep: str
    scale: str
    shard: str
    dataset: str
    family: str
    variant: str
    variant_sort: float
    k: int
    seed: int
    algorithm: str
    metrics: dict[str, float]
    timestamp: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweeps-root",
        type=Path,
        default=REPO_DIR / "results" / "sweeps",
        help="Directory containing sweep result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_DIR / "results" / "plots" / "current_sweeps",
        help="Directory where plots and coverage summaries are written.",
    )
    parser.add_argument(
        "--sweeps",
        nargs="*",
        default=[
            "small_experimental_sweep",
            "medium_experimental_sweep_shard0_of_2",
            "medium_experimental_sweep_shard1_of_2",
        ],
        help="Sweep directory names to read.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def parse_timestamp(value: str | None, fallback_mtime: float) -> datetime:
    if value:
        try:
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback_mtime, tz=timezone.utc)


def parse_config_name(config_path: str) -> tuple[str, int, int, str] | None:
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


def scale_and_shard(sweep_name: str) -> tuple[str, str]:
    if sweep_name.startswith("small_"):
        return "small", "small"
    if sweep_name.startswith("medium_") and "shard0" in sweep_name:
        return "medium", "medium_shard0"
    if sweep_name.startswith("medium_") and "shard1" in sweep_name:
        return "medium", "medium_shard1"
    if sweep_name.startswith("medium_"):
        return "medium", "medium"
    return sweep_name.split("_", maxsplit=1)[0], sweep_name


def classify_dataset(dataset: str) -> tuple[str, str, float]:
    lfr_mixing = re.match(r"lfr_n\d+_mu(?P<mu>\d+)$", dataset)
    if lfr_mixing:
        mu_raw = lfr_mixing.group("mu")
        mu = int(mu_raw) / 100.0
        return "lfr_mixing", f"mu={mu:.2f}", mu

    lfr_hetero = re.match(r"lfr_n\d+_tau(?P<tau>\d+)_max(?P<maxdeg>\d+)$", dataset)
    if lfr_hetero:
        tau = int(lfr_hetero.group("tau")) / 10.0
        maxdeg = int(lfr_hetero.group("maxdeg"))
        return "lfr_heterogeneity", f"tau1={tau:.1f}, max={maxdeg}", tau

    sbm_assort = re.match(r"sbm_assort_(?P<level>strong|medium|weak)_", dataset)
    if sbm_assort:
        level = sbm_assort.group("level")
        order = {"strong": 0.0, "medium": 1.0, "weak": 2.0}[level]
        return "sbm_assortativity", level.capitalize(), order

    if dataset.startswith("ring_"):
        match = re.match(r"ring_(?P<count>\d+)x(?P<size>\d+)_b(?P<bridges>\d+)$", dataset)
        if match:
            count = int(match.group("count"))
            size = int(match.group("size"))
            bridges = int(match.group("bridges"))
            return "ring_cliques", f"{count}x{size}, b={bridges}", float(count)
        return "ring_cliques", dataset, 0.0

    return "other", dataset, 0.0


def read_completed_runs(sweep_dir: Path) -> list[RunRecord]:
    scale, shard = scale_and_shard(sweep_dir.name)
    latest_by_config: dict[str, tuple[datetime, RunRecord]] = {}
    runs_dir = sweep_dir / "runs"
    if not runs_dir.exists():
        return []

    for scores_path in runs_dir.glob("*/scores.json"):
        metadata_path = scores_path.with_name("metadata.json")
        if not metadata_path.exists():
            continue
        try:
            scores = load_json(scores_path)
            metadata = load_json(metadata_path)
        except (json.JSONDecodeError, OSError):
            continue

        config_path = metadata.get("config_path")
        if not isinstance(config_path, str):
            continue
        parsed = parse_config_name(config_path)
        if parsed is None:
            continue
        dataset, k, seed, algorithm = parsed
        family, variant, variant_sort = classify_dataset(dataset)
        timestamp = parse_timestamp(
            metadata.get("timestamp_utc"),
            max(scores_path.stat().st_mtime, metadata_path.stat().st_mtime),
        )
        metrics = {
            metric: float(scores[metric])
            for metric in ALL_METRIC_KEYS
            if metric in scores and scores[metric] is not None and math.isfinite(float(scores[metric]))
        }
        if not metrics:
            continue

        record = RunRecord(
            sweep=sweep_dir.name,
            scale=scale,
            shard=shard,
            dataset=dataset,
            family=family,
            variant=variant,
            variant_sort=variant_sort,
            k=k,
            seed=seed,
            algorithm=algorithm,
            metrics=metrics,
            timestamp=timestamp,
        )
        key = Path(config_path).stem
        previous = latest_by_config.get(key)
        if previous is None or timestamp >= previous[0]:
            latest_by_config[key] = (timestamp, record)

    return [record for _, record in latest_by_config.values()]


def aggregate(records: Iterable[RunRecord]):
    grouped: dict[tuple, list[float]] = defaultdict(list)
    counts: dict[tuple, int] = defaultdict(int)
    for record in records:
        for metric, value in record.metrics.items():
            key = (
                record.scale,
                record.family,
                record.variant,
                record.variant_sort,
                record.k,
                record.algorithm,
                metric,
            )
            grouped[key].append(value)
            counts[key] += 1
    return {key: mean(values) for key, values in grouped.items()}, counts


def variants_for(records: list[RunRecord], scale: str, family: str) -> list[tuple[str, float]]:
    variants = {
        (record.variant, record.variant_sort)
        for record in records
        if record.scale == scale and record.family == family
    }
    return sorted(variants, key=lambda item: (item[1], item[0]))


def values_for_metric(agg: dict, scale: str, family: str, variant: str, algorithm: str, metric: str):
    values = []
    for key, value in agg.items():
        key_scale, key_family, key_variant, _, k, key_algorithm, key_metric = key
        if (
            key_scale == scale
            and key_family == family
            and key_variant == variant
            and key_algorithm == algorithm
            and key_metric == metric
        ):
            values.append((k, value))
    return sorted(values)


def returned_cluster_summary(
    records: list[RunRecord],
    *,
    scale: str,
    algorithm: str,
    family: str | None = None,
    variant: str | None = None,
    dataset: str | None = None,
) -> str | None:
    values = []
    for record in records:
        if record.scale != scale or record.algorithm != algorithm:
            continue
        if family is not None and record.family != family:
            continue
        if variant is not None and record.variant != variant:
            continue
        if dataset is not None and record.dataset != dataset:
            continue
        value = record.metrics.get("returned_num_clusters")
        if value is not None:
            values.append(int(round(value)))
    if not values:
        return None
    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return f"clusters={unique_values[0]}"
    if len(unique_values) <= 3:
        return "clusters=" + ",".join(str(value) for value in unique_values)
    return f"clusters={unique_values[0]}-{unique_values[-1]}"


def annotate_free_granularity_line(
    ax,
    records: list[RunRecord],
    *,
    scale: str,
    family: str | None,
    variant: str | None,
    dataset: str | None,
    algorithm: str,
    xs: list[int],
    ys: list[float],
) -> None:
    if algorithm not in {"markov", "louvain"} or not xs or not ys:
        return
    summary = returned_cluster_summary(
        records,
        scale=scale,
        family=family,
        variant=variant,
        dataset=dataset,
        algorithm=algorithm,
    )
    if summary is None:
        return
    ax.annotate(
        summary,
        xy=(xs[-1], ys[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=COLORS.get(algorithm),
    )


def save_family_plot(
    records: list[RunRecord],
    agg: dict,
    output_dir: Path,
    *,
    scale: str,
    family: str,
    family_title: str,
    metric: str,
    metric_label: str,
    metric_note: str,
) -> Path | None:
    variants = variants_for(records, scale, family)
    if not variants:
        return None

    cols = min(3, len(variants))
    rows = math.ceil(len(variants) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.6 * rows), squeeze=False)
    plotted_any = False
    for idx, (variant, _) in enumerate(variants):
        ax = axes[idx // cols][idx % cols]
        for algorithm in ALGORITHM_ORDER:
            points = values_for_metric(agg, scale, family, variant, algorithm, metric)
            if not points:
                continue
            plotted_any = True
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=4,
                color=COLORS.get(algorithm),
                label=ALGORITHM_LABELS.get(algorithm, algorithm),
            )
            annotate_free_granularity_line(
                ax,
                records,
                scale=scale,
                family=family,
                variant=variant,
                dataset=None,
                algorithm=algorithm,
                xs=xs,
                ys=ys,
            )
        ax.set_title(variant)
        ax.set_xlabel("Requested k")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.25)

    for idx in range(len(variants), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    if not plotted_any:
        plt.close(fig)
        return None

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=3,
            frameon=False,
        )
    fig.suptitle(f"{family_title}: {metric_label} by requested k ({scale})", fontsize=15)
    fig.text(0.5, 0.015, metric_note, ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    out_path = output_dir / f"{scale}_{family}_{metric}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def save_scale_comparison(
    records: list[RunRecord],
    output_dir: Path,
    *,
    family: str,
    family_title: str,
) -> Path | None:
    present_scales = {record.scale for record in records if record.family == family}
    if not {"small", "medium"}.issubset(present_scales):
        return None

    metric_values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for record in records:
        if record.family != family:
            continue
        for metric, _, _ in METRICS:
            if metric in record.metrics:
                metric_values[(record.scale, record.k, metric)].append(record.metrics[metric])

    if not metric_values:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), squeeze=False)
    plotted_any = False
    for idx, (metric, metric_label, note) in enumerate(METRICS):
        ax = axes[idx // 2][idx % 2]
        for scale in ["small", "medium"]:
            points = []
            for (key_scale, k, key_metric), values in metric_values.items():
                if key_scale == scale and key_metric == metric and values:
                    points.append((k, mean(values)))
            points.sort()
            if not points:
                continue
            plotted_any = True
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=2.2,
                color=COLORS[scale],
                label=scale.capitalize(),
            )
        ax.set_title(f"{metric_label} ({note})")
        ax.set_xlabel("Requested k")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    if not plotted_any:
        plt.close(fig)
        return None

    fig.suptitle(f"Small vs medium: {family_title}", fontsize=15)
    fig.text(
        0.5,
        0.018,
        "Each point averages completed runs over algorithms, seeds, and dataset variants in the family.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    out_path = output_dir / f"scale_comparison_{family}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def save_single_dataset_panel(
    records: list[RunRecord],
    output_dir: Path,
    *,
    scale: str,
    dataset: str,
    title: str,
    filename: str,
    footer: str,
) -> Path | None:
    selected = [
        record
        for record in records
        if record.scale == scale and record.dataset == dataset
    ]
    if not selected:
        return None

    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for record in selected:
        for metric, _, _ in SINGLE_DATASET_METRICS:
            if metric in record.metrics:
                grouped[(record.k, record.algorithm, metric)].append(record.metrics[metric])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), squeeze=False)
    plotted_any = False
    for idx, (metric, metric_label, metric_note) in enumerate(SINGLE_DATASET_METRICS):
        ax = axes[idx // 2][idx % 2]
        for algorithm in ALGORITHM_ORDER:
            points = []
            for (k, key_algorithm, key_metric), values in grouped.items():
                if key_algorithm == algorithm and key_metric == metric and values:
                    points.append((k, mean(values)))
            points.sort()
            if not points:
                continue
            plotted_any = True
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=2.0,
                markersize=4,
                color=COLORS.get(algorithm),
                label=ALGORITHM_LABELS.get(algorithm, algorithm),
            )
            annotate_free_granularity_line(
                ax,
                records,
                scale=scale,
                family=None,
                variant=None,
                dataset=dataset,
                algorithm=algorithm,
                xs=[point[0] for point in points],
                ys=[point[1] for point in points],
            )
        ax.set_title(f"{metric_label} ({metric_note})")
        ax.set_xlabel("Requested k")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.25)

    if not plotted_any:
        plt.close(fig)
        return None

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=3,
            frameon=False,
        )
    fig.suptitle(title, fontsize=15)
    fig.text(0.5, 0.015, footer, ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.14, 1, 0.94])
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def write_coverage_report(records: list[RunRecord], output_dir: Path, plot_paths: list[Path]) -> Path:
    by_shard: dict[str, set[tuple[str, int, int, str]]] = defaultdict(set)
    by_dataset: dict[tuple[str, str], set[tuple[int, int, str]]] = defaultdict(set)
    latest_by_shard: dict[str, datetime] = {}
    for record in records:
        by_shard[record.shard].add((record.dataset, record.k, record.seed, record.algorithm))
        by_dataset[(record.scale, record.dataset)].add((record.k, record.seed, record.algorithm))
        latest_by_shard[record.shard] = max(
            record.timestamp,
            latest_by_shard.get(record.shard, datetime.fromtimestamp(0, tz=timezone.utc)),
        )

    lines = [
        "# Current Sweep Plot Coverage",
        "",
        "Counts are de-duplicated by config name; restarted duplicate runs keep the latest completed result.",
        "",
        "## Sweep Coverage",
        "",
        "| Sweep piece | Finished configs | Expected configs | Latest completed run |",
        "|---|---:|---:|---|",
    ]
    for shard in ["small", "medium_shard0", "medium_shard1"]:
        finished = len(by_shard.get(shard, set()))
        expected = EXPECTED_TOTALS.get(shard, 0)
        latest = latest_by_shard.get(shard)
        latest_text = latest.strftime("%Y-%m-%d %H:%M UTC") if latest else "none"
        lines.append(f"| {shard} | {finished} | {expected} | {latest_text} |")

    medium_finished = len(by_shard.get("medium_shard0", set())) + len(by_shard.get("medium_shard1", set()))
    lines.append(f"| medium total | {medium_finished} | {EXPECTED_TOTALS['medium']} | - |")

    lines.extend(
        [
            "",
            "## Completely Finished Datasets",
            "",
            "A dataset is complete when all `5 k values x 5 seeds x 6 algorithms = 150` configs are present.",
            "",
        ]
    )
    complete = []
    partial = []
    for (scale, dataset), completed in sorted(by_dataset.items()):
        count = len(completed)
        if count >= 150:
            complete.append((scale, dataset, count))
        else:
            partial.append((scale, dataset, count))

    if complete:
        lines.extend(["| Scale | Dataset | Finished configs |", "|---|---|---:|"])
        for scale, dataset, count in complete:
            lines.append(f"| {scale} | `{dataset}` | {count} |")
    else:
        lines.append("No dataset is fully complete yet.")

    lines.extend(["", "## Partial Datasets With Results", "", "| Scale | Dataset | Finished configs |", "|---|---|---:|"])
    for scale, dataset, count in partial:
        lines.append(f"| {scale} | `{dataset}` | {count} |")

    lines.extend(["", "## Generated Plots", ""])
    for path in plot_paths:
        lines.append(f"- `{path.relative_to(REPO_DIR)}`")

    out_path = output_dir / "coverage_report.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main() -> int:
    args = parse_args()
    sweeps_root = args.sweeps_root if args.sweeps_root.is_absolute() else REPO_DIR / args.sweeps_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in output_dir.glob("*.png"):
        if old_plot.name == "scale_controlled_sbm_medium_probs.png":
            continue
        old_plot.unlink()
    old_report = output_dir / "coverage_report.md"
    if old_report.exists():
        old_report.unlink()

    records: list[RunRecord] = []
    for sweep_name in args.sweeps:
        records.extend(read_completed_runs(sweeps_root / sweep_name))

    agg, _ = aggregate(records)
    plot_paths: list[Path] = []

    families = [
        ("lfr_mixing", "LFR mixing sweep"),
        ("sbm_assortativity", "SBM assortativity sweep"),
        ("ring_cliques", "Ring of cliques"),
    ]
    for scale in ["small", "medium"]:
        for family, family_title in families:
            for metric, metric_label, metric_note in METRICS:
                path = save_family_plot(
                    records,
                    agg,
                    output_dir,
                    scale=scale,
                    family=family,
                    family_title=family_title,
                    metric=metric,
                    metric_label=metric_label,
                    metric_note=metric_note,
                )
                if path is not None:
                    plot_paths.append(path)

    for family, family_title in [
        ("lfr_mixing", "LFR mixing sweep"),
        ("sbm_assortativity", "SBM assortativity sweep"),
    ]:
        path = save_scale_comparison(records, output_dir, family=family, family_title=family_title)
        if path is not None:
            plot_paths.append(path)

    single_dataset_specs = [
        (
            "small",
            "sbm_assort_medium_40x3",
            "Small balanced SBM, medium assortativity",
            "single_small_balanced_sbm_medium_assortativity.png",
            "Balanced SBM uses 3 equal blocks of 40 nodes; medium assortativity uses p_in=0.30, p_out=0.06.",
        ),
        (
            "small",
            "lfr_n100_mu20",
            "Small LFR, middle mixing (mu=0.20)",
            "single_small_lfr_mu20.png",
            "LFR n=100 with heterogeneous degrees/community sizes; mu=0.20 is the middle-noise setting in the sweep.",
        ),
        (
            "small",
            "social_karate_club",
            "Karate Club k-sensitivity",
            "single_small_social_karate_club.png",
            "Real-graph labels have natural k=2; k>2 shows how algorithms over-partition relative to the known split.",
        ),
        (
            "small",
            "social_davis_southern_women",
            "Davis Southern Women k-sensitivity",
            "single_small_social_davis_southern_women.png",
            "Reference labels are bipartite node type; use this mainly as a sanity check, not a community gold standard.",
        ),
        (
            "small",
            "social_les_miserables",
            "Les Miserables k-sensitivity",
            "single_small_social_les_miserables.png",
            "Reference labels are greedy-modularity communities, so modularity-aligned methods have an evaluation advantage.",
        ),
    ]
    for scale, dataset, title, filename, footer in single_dataset_specs:
        path = save_single_dataset_panel(
            records,
            output_dir,
            scale=scale,
            dataset=dataset,
            title=title,
            filename=filename,
            footer=footer,
        )
        if path is not None:
            plot_paths.append(path)

    coverage_path = write_coverage_report(records, output_dir, plot_paths)
    print(f"Loaded {len(records)} de-duplicated completed configs.")
    print(f"Wrote coverage report: {coverage_path}")
    for path in plot_paths:
        print(f"Wrote plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
