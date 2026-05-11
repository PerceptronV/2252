#!/usr/bin/env python3
"""Plot runtime comparisons for completed sweep runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

REPO_DIR = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = REPO_DIR / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ALGORITHMS = [
    "kmeans",
    "peng_kmeans",
    "fiedler",
    "shi_malik",
    "markov",
    "louvain",
]

LABELS = {
    "kmeans": "Spectral\nk-means",
    "peng_kmeans": "Peng\nweighted",
    "fiedler": "Fiedler",
    "shi_malik": "Shi-\nMalik",
    "markov": "MCL",
    "louvain": "Louvain",
}

LINE_LABELS = {
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
    match = re.match(
        r"(?P<dataset>.+)__k(?P<k>\d+)__seed(?P<seed>\d+)__(?P<algorithm>.+)$",
        stem,
    )
    if not match:
        return None
    return (
        match.group("dataset"),
        int(match.group("k")),
        int(match.group("seed")),
        match.group("algorithm"),
    )


def read_latest_runs(sweep_dir: Path, dataset: str | None = None) -> list[dict]:
    latest: dict[str, tuple[datetime, dict]] = {}
    for scores_path in (sweep_dir / "runs").glob("*/scores.json"):
        metadata_path = scores_path.with_name("metadata.json")
        if not metadata_path.exists():
            continue
        try:
            metadata = load_json(metadata_path)
        except (json.JSONDecodeError, OSError):
            continue
        parsed = parse_config(str(metadata.get("config_path", "")))
        if parsed is None:
            continue
        run_dataset, k, seed, algorithm = parsed
        if dataset is not None and run_dataset != dataset:
            continue
        runtime = metadata.get("algorithm_runtime_seconds")
        if runtime is None or not math.isfinite(float(runtime)):
            continue
        timestamp = parse_time(
            metadata.get("timestamp_utc"),
            max(scores_path.stat().st_mtime, metadata_path.stat().st_mtime),
        )
        record = {
            "dataset": run_dataset,
            "k": k,
            "seed": seed,
            "algorithm": algorithm,
            "runtime": float(runtime),
            "num_nodes": int(metadata.get("graph", {}).get("num_nodes", 0)),
            "num_edges": int(metadata.get("graph", {}).get("num_edges", 0)),
        }
        key = Path(str(metadata["config_path"])).stem
        previous = latest.get(key)
        if previous is None or timestamp >= previous[0]:
            latest[key] = (timestamp, record)
    return [record for _, record in latest.values()]


def by_run_key(records: list[dict]) -> dict[tuple[int, int, str], dict]:
    return {
        (int(record["k"]), int(record["seed"]), str(record["algorithm"])): record
        for record in records
    }


def grouped_values(records: list[dict], *keys: str) -> dict[tuple, list[float]]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(float(record["runtime"]))
    return grouped


def write_summary(output_dir: Path, small_records: list[dict], medium_records: list[dict]) -> Path:
    lines = [
        "# Runtime Summary",
        "",
        "Runtime is `algorithm_runtime_seconds` from per-run metadata.",
        "",
        "## Controlled SBM, Same Edge Probabilities",
        "",
        "| Algorithm | Small median s | Small mean s | Small n | Medium median s | Medium mean s | Medium n | Mean slowdown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        small = [
            record["runtime"]
            for record in small_records
            if record["algorithm"] == algorithm
        ]
        medium = [
            record["runtime"]
            for record in medium_records
            if record["algorithm"] == algorithm
        ]
        if not small and not medium:
            continue
        small_median = median(small) if small else float("nan")
        small_mean = mean(small) if small else float("nan")
        medium_median = median(medium) if medium else float("nan")
        medium_mean = mean(medium) if medium else float("nan")
        slowdown = medium_mean / small_mean if small and medium and small_mean > 0 else float("nan")
        lines.append(
            "| "
            + " | ".join(
                [
                    LINE_LABELS[algorithm],
                    f"{small_median:.4f}" if small else "-",
                    f"{small_mean:.4f}" if small else "-",
                    str(len(small)),
                    f"{medium_median:.4f}" if medium else "-",
                    f"{medium_mean:.4f}" if medium else "-",
                    str(len(medium)),
                    f"{slowdown:.1f}x" if math.isfinite(slowdown) else "-",
                ]
            )
            + " |"
        )
    out_path = output_dir / "runtime_summary.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def plot_runtime(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    small_sbm = read_latest_runs(
        REPO_DIR / "results" / "sweeps" / "controlled_sbm_40x3_medium_probs",
        "sbm_40x3_medium_probs",
    )
    medium_sbm = read_latest_runs(
        REPO_DIR / "results" / "sweeps" / "medium_experimental_sweep_shard1_of_2",
        "sbm_assort_medium_150x3",
    )
    small_by_key = by_run_key(small_sbm)
    medium_by_key = by_run_key(medium_sbm)
    matched_keys = sorted(set(small_by_key) & set(medium_by_key))
    matched_small = [small_by_key[key] for key in matched_keys]
    matched_medium = [medium_by_key[key] for key in matched_keys]

    single_small = read_latest_runs(
        REPO_DIR / "results" / "sweeps" / "small_experimental_sweep",
        "sbm_assort_medium_40x3",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), gridspec_kw={"width_ratios": [1.05, 1.25]})

    ax = axes[0]
    x_positions = list(range(len(ALGORITHMS)))
    width = 0.36
    small_grouped = grouped_values(matched_small, "algorithm")
    medium_grouped = grouped_values(matched_medium, "algorithm")
    small_medians = [
        median(small_grouped[(algorithm,)]) if (algorithm,) in small_grouped else math.nan
        for algorithm in ALGORITHMS
    ]
    medium_medians = [
        median(medium_grouped[(algorithm,)]) if (algorithm,) in medium_grouped else math.nan
        for algorithm in ALGORITHMS
    ]
    ax.bar(
        [x - width / 2 for x in x_positions],
        small_medians,
        width=width,
        color="#4c78a8",
        label="3x40 nodes",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        medium_medians,
        width=width,
        color="#f58518",
        label="3x150 nodes",
    )
    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([LABELS[algorithm] for algorithm in ALGORITHMS])
    ax.set_ylabel("Median runtime (seconds, log scale)")
    ax.set_title("Matched SBM scale-up\nsame p_in=0.18, p_out=0.035")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    by_k_algorithm = grouped_values(single_small, "k", "algorithm")
    for algorithm in ALGORITHMS:
        points = sorted(
            (k, median(values))
            for (k, key_algorithm), values in by_k_algorithm.items()
            if key_algorithm == algorithm
        )
        if not points:
            continue
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            linewidth=2.1,
            markersize=4.5,
            color=COLORS[algorithm],
            label=LINE_LABELS[algorithm],
        )
    ax.set_yscale("log")
    ax.set_xlabel("Requested k")
    ax.set_ylabel("Median runtime (seconds, log scale)")
    ax.set_title("Same small graph family\nbalanced SBM, 3x40 nodes")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8.5)

    fig.suptitle("Algorithm runtime comparison", fontsize=16)
    fig.text(
        0.5,
        0.015,
        f"Bars use matched configs only ({len(matched_keys)} small/medium runs). "
        "Each point/median averages completed seeds for the same graph and requested k.",
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    plot_path = output_dir / "runtime_algorithm_comparison.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    summary_path = write_summary(output_dir, matched_small, matched_medium)
    return plot_path, summary_path


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_DIR / args.output_dir
    plot_path, summary_path = plot_runtime(output_dir)
    print(f"wrote {plot_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
