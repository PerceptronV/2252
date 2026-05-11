#!/usr/bin/env python3
"""Plot controlled small-vs-medium SBM scale comparison.

This adds one plot without clearing or regenerating the existing plot directory.
It compares:
  - small controlled SBM: sizes 40,40,40, p_in=0.18, p_out=0.035
  - medium SBM: sizes 150,150,150, p_in=0.18, p_out=0.035
"""

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

ALGORITHM_LABELS = {
    "kmeans": "Spectral k-means",
    "peng_kmeans": "Peng weighted k-means",
    "fiedler": "Fiedler",
    "shi_malik": "Shi-Malik",
    "markov": "MCL",
    "louvain": "Louvain",
}

COLORS = {
    "controlled small": "#1f77b4",
    "medium": "#ff7f0e",
}

ALGORITHM_COLORS = {
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
        "--output",
        type=Path,
        default=REPO_DIR / "results" / "plots" / "current_sweeps" / "scale_controlled_sbm_medium_probs.png",
        help="PNG path to write.",
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


def read_latest_dataset_runs(sweep_dir: Path, dataset: str) -> list[dict]:
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
        run_dataset, k, seed, algorithm = parsed
        if run_dataset != dataset:
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
            "scores": scores,
        }
        key = Path(str(metadata["config_path"])).stem
        previous = latest.get(key)
        if previous is None or timestamp >= previous[0]:
            latest[key] = (timestamp, record)
    return [record for _, record in latest.values()]


def aggregate_by_k(records: list[dict]) -> dict[tuple[int, str], float]:
    values: dict[tuple[int, str], list[float]] = defaultdict(list)
    for record in records:
        for metric, _, _ in METRICS:
            value = record["scores"].get(metric)
            if value is not None and math.isfinite(float(value)):
                values[(record["k"], metric)].append(float(value))
    return {key: mean(metric_values) for key, metric_values in values.items()}


def aggregate_by_k_algorithm(records: list[dict]) -> dict[tuple[int, str, str], float]:
    values: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for record in records:
        for metric, _, _ in METRICS:
            value = record["scores"].get(metric)
            if value is not None and math.isfinite(float(value)):
                values[(record["k"], record["algorithm"], metric)].append(float(value))
    return {key: mean(metric_values) for key, metric_values in values.items()}


def run_key(record: dict) -> tuple[int, int, str]:
    return (int(record["k"]), int(record["seed"]), str(record["algorithm"]))


def plot_aggregate_scale(output: Path, controlled: dict, medium: dict, matched_count: int) -> None:
    series = {
        "controlled small": controlled,
        "medium": medium,
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), squeeze=False)
    for idx, (metric, label, note) in enumerate(METRICS):
        ax = axes[idx // 2][idx % 2]
        for series_name, data in series.items():
            points = sorted((k, value) for (k, key_metric), value in data.items() if key_metric == metric)
            if not points:
                continue
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=2.2,
                color=COLORS[series_name],
                label=series_name.capitalize(),
            )
        ax.set_title(f"{label} ({note})")
        ax.set_xlabel("Requested k")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig.suptitle("Controlled SBM scale comparison: same edge probabilities", fontsize=15)
    footer = (
        "Matched configs only: same requested k, seed, and algorithm on both curves. "
        f"Small: 3x40, p_in=0.18, p_out=0.035, {matched_count}/150 configs. "
        f"Medium: 3x150, p_in=0.18, p_out=0.035, {matched_count}/150 configs."
    )
    fig.text(0.5, 0.018, footer, ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_algorithm_scale(
    output_dir: Path,
    controlled: dict[tuple[int, str, str], float],
    medium: dict[tuple[int, str, str], float],
    matched_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for algorithm in ALGORITHMS:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), squeeze=False)
        for idx, (metric, label, note) in enumerate(METRICS):
            ax = axes[idx // 2][idx % 2]
            small_points = sorted(
                (k, value)
                for (k, key_algorithm, key_metric), value in controlled.items()
                if key_algorithm == algorithm and key_metric == metric
            )
            medium_points = sorted(
                (k, value)
                for (k, key_algorithm, key_metric), value in medium.items()
                if key_algorithm == algorithm and key_metric == metric
            )
            color = ALGORITHM_COLORS[algorithm]
            if small_points:
                ax.plot(
                    [point[0] for point in small_points],
                    [point[1] for point in small_points],
                    marker="o",
                    linewidth=2.0,
                    markersize=4,
                    color=color,
                    linestyle="-",
                    label="Controlled small",
                )
            if medium_points:
                ax.plot(
                    [point[0] for point in medium_points],
                    [point[1] for point in medium_points],
                    marker="s",
                    linewidth=2.0,
                    markersize=4,
                    color=color,
                    linestyle="--",
                    label="Medium",
                )
            ax.set_title(f"{label} ({note})")
            ax.set_xlabel("Requested k")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)

        fig.suptitle(f"Controlled SBM scale comparison: {ALGORITHM_LABELS[algorithm]}", fontsize=15)
        footer = (
            f"Matched configs only: {matched_count}/150 per scale. "
            "Solid circle = controlled small, dashed square = medium."
        )
        fig.text(0.5, 0.014, footer, ha="center", fontsize=10)
        fig.tight_layout(rect=[0, 0.09, 1, 0.94])
        fig.savefig(output_dir / f"scale_controlled_sbm_medium_probs_{algorithm}.png", dpi=220)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else REPO_DIR / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    controlled_records = read_latest_dataset_runs(
        REPO_DIR / "results" / "sweeps" / "controlled_sbm_40x3_medium_probs",
        "sbm_40x3_medium_probs",
    )
    medium_records = read_latest_dataset_runs(
        REPO_DIR / "results" / "sweeps" / "medium_experimental_sweep_shard1_of_2",
        "sbm_assort_medium_150x3",
    )

    controlled_by_key = {run_key(record): record for record in controlled_records}
    medium_by_key = {run_key(record): record for record in medium_records}
    matched_keys = sorted(set(controlled_by_key) & set(medium_by_key))
    matched_controlled_records = [controlled_by_key[key] for key in matched_keys]
    matched_medium_records = [medium_by_key[key] for key in matched_keys]

    controlled = aggregate_by_k(matched_controlled_records)
    medium = aggregate_by_k(matched_medium_records)
    plot_aggregate_scale(output, controlled, medium, len(matched_keys))

    controlled_by_algorithm = aggregate_by_k_algorithm(matched_controlled_records)
    medium_by_algorithm = aggregate_by_k_algorithm(matched_medium_records)
    algorithm_output_dir = output.parent
    plot_algorithm_scale(algorithm_output_dir, controlled_by_algorithm, medium_by_algorithm, len(matched_keys))

    print(f"controlled small configs available: {len(controlled_records)}/150")
    print(f"medium configs available: {len(medium_records)}/150")
    print(f"matched configs plotted per curve: {len(matched_keys)}/150")
    print(f"wrote {output}")
    for algorithm in ALGORITHMS:
        print(f"wrote {output.parent / f'scale_controlled_sbm_medium_probs_{algorithm}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
