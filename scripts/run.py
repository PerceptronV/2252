#!/usr/bin/env python3
"""CLI entry point: python scripts/run.py <config.yaml>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.runner import run_experiment  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a graph-partitioning experiment.")
    parser.add_argument("config", type=Path, help="path to experiment YAML config")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_DIR / "results",
        help="directory under which run_uuid folders are created (default: ./results)",
    )
    args = parser.parse_args(argv)

    run_dir = run_experiment(
        args.config,
        results_root=args.results_root,
        repo_dir=REPO_DIR,
        cli_args=sys.argv,
    )
    print(f"run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
