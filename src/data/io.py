"""Utilities for serializing graphs into a common on-disk bundle format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from core.graph import Graph


def _normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def save_graph_bundle(
    path: str | Path,
    *,
    graph: Graph,
    target: np.ndarray,
    num_clusters: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a graph and labels in a portable NPZ bundle."""

    bundle_path = Path(path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    adjacency = graph.adjacency.tocsr()
    target = np.asarray(target, dtype=int)
    if target.shape != (graph.num_nodes,):
        raise ValueError(
            f"target must have shape ({graph.num_nodes},); got {target.shape}"
        )

    payload: dict[str, Any] = {
        "adjacency_indptr": np.asarray(adjacency.indptr, dtype=np.int64),
        "adjacency_indices": np.asarray(adjacency.indices, dtype=np.int64),
        "adjacency_data": np.asarray(adjacency.data, dtype=float),
        "adjacency_shape": np.asarray(adjacency.shape, dtype=np.int64),
        "target": target,
        "num_clusters": np.asarray(int(num_clusters), dtype=np.int64),
        "graph_name": np.asarray(graph.name or bundle_path.stem),
        "metadata_json": np.asarray(json.dumps(_normalize_metadata(metadata))),
    }
    if graph.node_features is not None:
        payload["node_features"] = np.asarray(graph.node_features)

    np.savez(bundle_path, **payload)
    return bundle_path


def load_graph_bundle(path: str | Path) -> tuple[Graph, np.ndarray, int]:
    """Load a graph bundle written by :func:`save_graph_bundle`."""

    bundle_path = Path(path)
    with np.load(bundle_path, allow_pickle=False) as data:
        shape = tuple(int(x) for x in data["adjacency_shape"])
        adjacency = sp.csr_matrix(
            (
                data["adjacency_data"],
                data["adjacency_indices"],
                data["adjacency_indptr"],
            ),
            shape=shape,
        )
        target = np.asarray(data["target"], dtype=int)
        num_clusters = int(np.asarray(data["num_clusters"]).item())
        graph_name = str(np.asarray(data["graph_name"]).item())
        metadata_json = str(np.asarray(data["metadata_json"]).item())
        metadata = json.loads(metadata_json)
        node_features = (
            np.asarray(data["node_features"]) if "node_features" in data.files else None
        )

    graph = Graph(
        adjacency=adjacency,
        num_nodes=adjacency.shape[0],
        node_features=node_features,
        name=graph_name,
        metadata=metadata,
    )
    return graph, target, num_clusters
