#!/usr/bin/env python3
"""Run partitioning benchmark sweeps across datasets, k values, and algorithms."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.runner import run_experiment  # noqa: E402
from data.io import load_graph_bundle  # noqa: E402


EVAL_SPECS: list[dict[str, Any]] = [
    {"type": "conductance"},
    {"type": "min_conductance"},
    {"type": "max_conductance"},
    {"type": "modularity"},
    {"type": "returned_num_clusters"},
    {"type": "ari"},
    {"type": "nmi"},
]


def _parse_int_list(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def _algorithm_specs(seed: int, k: int) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "kmeans",
            {
                "type": "spectral",
                "params": {
                    "embedding_dim": k,
                    "normalization": "auto",
                    "seed": seed,
                    "baseline": "kmeans",
                    "baseline_params": {"n_init": 10, "max_iter": 300},
                },
            },
        ),
        (
            "peng_kmeans",
            {
                "type": "spectral",
                "params": {
                    "embedding_dim": k,
                    "normalization": "auto",
                    "seed": seed,
                    "baseline": "peng_kmeans",
                    "baseline_params": {"n_init": 10, "max_iter": 300},
                },
            },
        ),
        (
            "fiedler",
            {
                "type": "fiedler",
                "params": {"seed": seed},
            },
        ),
        (
            "shi_malik",
            {
                "type": "shi_malik",
                "params": {"seed": seed},
            },
        ),
        (
            "markov",
            {
                "type": "markov",
                "params": {},
            },
        ),
        (
            "louvain",
            {
                "type": "louvain",
                "params": {},
            },
        ),
    ]


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_DIR))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark sweeps over graph bundles.")
    parser.add_argument(
        "--datasets",
        default="datasets/synthetic/*.npz,datasets/social/*.npz",
        help="comma-separated glob patterns for dataset bundles",
    )
    parser.add_argument("--ks", default="2,3,4,5,6", help="comma-separated k values to evaluate")
    parser.add_argument("--seeds", default="0", help="comma-separated random seeds")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_DIR / "results" / "sweeps",
        help="directory under which the sweep directory is created",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="optional sweep name; defaults to a timestamp",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dataset_paths: list[Path] = []
    for pattern in [part.strip() for part in args.datasets.split(",") if part.strip()]:
        dataset_paths.extend(sorted(REPO_DIR.glob(pattern)))
    dataset_paths = list(dict.fromkeys(dataset_paths))
    if not dataset_paths:
        raise SystemExit("no dataset bundles matched the provided patterns")

    ks = _parse_int_list(args.ks)
    seeds = _parse_int_list(args.seeds)
    sweep_name = args.name or dt.datetime.now(dt.timezone.utc).strftime("sweep_%Y%m%dT%H%M%SZ")
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = REPO_DIR / output_root
    sweep_dir = output_root / sweep_name
    runs_dir = sweep_dir / "runs"
    configs_dir = sweep_dir / "configs"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for dataset_path in dataset_paths:
        graph, target, true_k = load_graph_bundle(dataset_path)
        dataset_kind = str(graph.metadata.get("kind", "unknown"))

        for k in ks:
            if not (1 <= k <= graph.num_nodes):
                continue
            for seed in seeds:
                for algorithm_name, algorithm_spec in _algorithm_specs(seed, k):
                    config_name = f"{dataset_path.stem}__k{k}__seed{seed}__{algorithm_name}"
                    config_payload = {
                        "name": config_name,
                        "seed": seed,
                        "dataset": {
                            "type": "serialized_graph",
                            "params": {
                                "path": str(dataset_path.relative_to(REPO_DIR)),
                                "num_clusters": k,
                            },
                        },
                        "algorithm": algorithm_spec,
                        "evals": EVAL_SPECS,
                    }
                    config_path = configs_dir / f"{config_name}.yaml"
                    _write_yaml(config_path, config_payload)

                    try:
                        run_dir = run_experiment(
                            config_path,
                            results_root=runs_dir,
                            repo_dir=REPO_DIR,
                            cli_args=sys.argv,
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "dataset": str(dataset_path),
                                "dataset_kind": dataset_kind,
                                "k": k,
                                "seed": seed,
                                "algorithm": algorithm_name,
                                "error": repr(exc),
                            }
                        )
                        continue

                    scores = _load_json(run_dir / "scores.json")
                    metadata = _load_json(run_dir / "metadata.json")
                    row = {
                        "run_uuid": metadata["run_uuid"],
                        "dataset": str(dataset_path.relative_to(REPO_DIR)),
                        "dataset_kind": dataset_kind,
                        "graph_name": graph.name,
                        "true_k": true_k,
                        "requested_k": k,
                        "k": k,
                        "seed": seed,
                        "algorithm": algorithm_name,
                        "runtime_seconds": metadata["algorithm_runtime_seconds"],
                        "run_dir": _display_path(run_dir),
                    }
                    row.update(scores)
                    summary_rows.append(row)

    summary_jsonl = sweep_dir / "summary.jsonl"
    with open(summary_jsonl, "w") as f:
        for row in summary_rows:
            f.write(json.dumps(row) + "\n")

    summary_csv = sweep_dir / "summary.csv"
    if summary_rows:
        fixed_fieldnames = [
            "run_uuid",
            "dataset",
            "dataset_kind",
            "graph_name",
            "true_k",
            "requested_k",
            "k",
            "seed",
            "algorithm",
            "runtime_seconds",
            "run_dir",
        ]
        score_fieldnames = [
            spec["type"]
            for spec in EVAL_SPECS
            if any(spec["type"] in row for row in summary_rows)
        ]
        extra_fieldnames = sorted(
            set().union(*(row.keys() for row in summary_rows))
            - set(fixed_fieldnames)
            - set(score_fieldnames)
        )
        fieldnames = fixed_fieldnames + score_fieldnames + extra_fieldnames
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    failures_jsonl = sweep_dir / "failures.jsonl"
    with open(failures_jsonl, "w") as f:
        for row in failures:
            f.write(json.dumps(row) + "\n")

    manifest = {
        "datasets": [str(path.relative_to(REPO_DIR)) for path in dataset_paths],
        "algorithms": [name for name, _ in _algorithm_specs(seed=0, k=1)],
        "evals": [spec["type"] for spec in EVAL_SPECS],
        "ks": ks,
        "seeds": seeds,
        "num_runs": len(summary_rows),
        "num_failures": len(failures),
    }
    with open(sweep_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(sweep_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
