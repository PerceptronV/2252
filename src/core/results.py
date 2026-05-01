"""Write a single run's outputs into results/<uuid>/ with full metadata."""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata as _md
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import yaml

from core.graph import Graph

_PACKAGES_TO_RECORD = ("numpy", "scipy", "pyyaml", "networkx")


def _git_metadata(repo_dir: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args,
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return out.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    sha = _run(["git", "rev-parse", "HEAD"])
    if sha is None:
        return {"git_available": False}
    porcelain = _run(["git", "status", "--porcelain"]) or ""
    return {
        "git_available": True,
        "git_sha": sha,
        "git_dirty": bool(porcelain.strip()),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _PACKAGES_TO_RECORD:
        try:
            versions[pkg] = _md.version(pkg)
        except _md.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def _adjacency_sha256(A: sp.csr_matrix) -> str:
    A = A.tocsr()
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(A.indptr).tobytes())
    h.update(np.ascontiguousarray(A.indices).tobytes())
    h.update(np.ascontiguousarray(A.data).tobytes())
    h.update(str(A.shape).encode())
    return h.hexdigest()


def write_results(
    run_dir: Path,
    *,
    run_uuid: str,
    config_raw: dict[str, Any],
    config_path: Path,
    graph: Graph,
    predicted: np.ndarray,
    target: np.ndarray,
    scores: dict[str, float],
    repo_dir: Path,
    cli_args: list[str],
) -> None:
    """Persist everything needed to interpret and reproduce a run."""

    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config_raw, f, sort_keys=False)

    metadata: dict[str, Any] = {
        "run_uuid": run_uuid,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config_path": str(config_path),
        "cli_args": cli_args,
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "graph": {
            "name": graph.name,
            "num_nodes": graph.num_nodes,
            "num_edges": int(graph.adjacency.nnz // 2),
            "adjacency_sha256": _adjacency_sha256(graph.adjacency),
            "has_node_features": graph.node_features is not None,
        },
        **_git_metadata(repo_dir),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(run_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    np.savez(
        run_dir / "predictions.npz",
        predicted=predicted,
        target=target,
    )
