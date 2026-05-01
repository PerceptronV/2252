"""Experiment runner: YAML config in, results/<uuid>/ out."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import data  # noqa: F401  -- side-effect: register all datasets
import algorithms  # noqa: F401  -- side-effect: register all algorithms
import baselines  # noqa: F401  -- side-effect: register all baselines
import evals as _evals_pkg  # noqa: F401  -- side-effect: register all evals

from core.config import ExperimentConfig, load_config
from core.registry import ALGORITHMS, DATASETS, EVALS, resolve
from core.results import write_results


def run_experiment(
    config_path: str | Path,
    *,
    results_root: str | Path = "results",
    repo_dir: str | Path | None = None,
    cli_args: list[str] | None = None,
) -> Path:
    """Execute a single experiment and return the path of its run directory."""

    config_path = Path(config_path)
    cfg = load_config(config_path)

    run_uuid = uuid.uuid4().hex
    run_dir = Path(results_root) / run_uuid

    dataset_cls = resolve(DATASETS, cfg.dataset.type)
    algorithm_cls = resolve(ALGORITHMS, cfg.algorithm.type)
    eval_clses = [resolve(EVALS, e.type) for e in cfg.evals]

    dataset = dataset_cls(**cfg.dataset.params)
    algorithm = algorithm_cls(**cfg.algorithm.params)
    eval_objs = [cls(**spec.params) for cls, spec in zip(eval_clses, cfg.evals)]

    graph, target = dataset.load()
    predicted = algorithm.fit_predict(graph, k=dataset.num_clusters)

    scores: dict[str, float] = {}
    for ev in eval_objs:
        scores[ev.name] = float(ev(graph, predicted, target))

    write_results(
        run_dir,
        run_uuid=run_uuid,
        config_raw=cfg.raw,
        config_path=config_path,
        graph=graph,
        predicted=predicted,
        target=target,
        scores=scores,
        repo_dir=Path(repo_dir) if repo_dir else config_path.resolve().parent,
        cli_args=cli_args if cli_args is not None else sys.argv,
    )

    return run_dir


def _coerce_eval_specs(cfg: ExperimentConfig) -> list:  # kept for future overrides
    return cfg.evals
