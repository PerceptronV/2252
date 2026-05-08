"""Recursive Fiedler sweep-cut partitioning."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from algorithms._spectral import normalized_laplacian
from algorithms.base import Algorithm
from core.graph import Graph
from core.registry import register_algorithm


def _fiedler_vector(
    adjacency: sp.csr_matrix,
    *,
    tol: float,
    max_iter: int | None,
    seed: int | None,
) -> np.ndarray:
    n = adjacency.shape[0]
    if n < 2:
        raise ValueError("Fiedler vector requires at least two nodes")

    laplacian, _ = normalized_laplacian(adjacency)
    if n <= 64:
        vals, vecs = eigh(laplacian.toarray())
        order = np.argsort(vals)
        return vecs[:, order[1]]

    rng = np.random.default_rng(seed)
    shifted = 2.0 * sp.identity(n, format="csr") - laplacian
    vals_shifted, vecs = eigsh(
        shifted,
        k=2,
        which="LA",
        tol=tol,
        maxiter=max_iter,
        v0=rng.standard_normal(n),
    )
    eigvals = 2.0 - vals_shifted
    order = np.argsort(eigvals)
    return vecs[:, order[1]]


def _sweep_cut(
    adjacency: sp.csr_matrix,
    scores: np.ndarray,
) -> tuple[np.ndarray, float]:
    n = adjacency.shape[0]
    if n < 2:
        raise ValueError("sweep cut requires at least two nodes")

    order = np.argsort(scores, kind="mergesort")
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(float, copy=False)
    total_volume = float(np.sum(degrees))

    in_set = np.zeros(n, dtype=bool)
    current_cut = 0.0
    current_volume = 0.0
    best_phi = float("inf")
    best_prefix = 1

    for prefix_idx, node in enumerate(order[:-1], start=1):
        row_start = adjacency.indptr[node]
        row_end = adjacency.indptr[node + 1]
        neighbors = adjacency.indices[row_start:row_end]
        weights = adjacency.data[row_start:row_end]
        internal_weight = float(np.sum(weights[in_set[neighbors]]))

        current_cut += degrees[node] - 2.0 * internal_weight
        current_volume += degrees[node]
        in_set[node] = True

        complement_volume = total_volume - current_volume
        denom = min(current_volume, complement_volume)
        if denom <= 0:
            continue
        phi = current_cut / denom
        if phi < best_phi:
            best_phi = float(phi)
            best_prefix = prefix_idx

    mask = np.zeros(n, dtype=bool)
    mask[order[:best_prefix]] = True
    return mask, best_phi


def _one_vs_rest_conductance(adjacency: sp.csr_matrix, nodes: np.ndarray) -> float:
    n = adjacency.shape[0]
    if nodes.size == 0 or nodes.size == n:
        return 0.0

    mask = np.zeros(n, dtype=bool)
    mask[nodes] = True
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(float, copy=False)
    volume = float(np.sum(degrees[mask]))
    total_volume = float(np.sum(degrees))
    denom = min(volume, total_volume - volume)
    if denom <= 0:
        return 0.0

    cut = float(adjacency[mask][:, ~mask].sum())
    return cut / denom


@register_algorithm("fiedler")
class FiedlerSweepAlgorithm(Algorithm):
    """Recursive normalized-Laplacian Fiedler sweep-cut algorithm."""

    def __init__(self, tol: float = 0.0, max_iter: int | None = None, seed: int | None = None) -> None:
        self.tol = float(tol)
        self.max_iter = max_iter
        self.seed = seed

    def fit_predict(self, graph: Graph, k: int) -> np.ndarray:
        if not (1 <= k <= graph.num_nodes):
            raise ValueError(
                f"k must satisfy 1 <= k <= num_nodes; got k={k}, n={graph.num_nodes}"
            )

        adjacency = graph.adjacency.tocsr().astype(float)
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()

        pieces = [np.arange(graph.num_nodes, dtype=int)]
        while len(pieces) < k:
            split_idx = self._select_piece_to_split(adjacency, pieces)
            if split_idx is None:
                break

            nodes = pieces.pop(split_idx)
            left, right = self._split_piece(adjacency, nodes)
            pieces.extend([left, right])

        while len(pieces) < k:
            pieces.sort(key=len, reverse=True)
            nodes = pieces.pop(0)
            if nodes.size < 2:
                pieces.insert(0, nodes)
                break
            midpoint = nodes.size // 2
            pieces.extend([nodes[:midpoint], nodes[midpoint:]])

        out = np.empty(graph.num_nodes, dtype=int)
        for cluster_id, nodes in enumerate(pieces[:k]):
            out[nodes] = cluster_id
        return out

    def _select_piece_to_split(
        self,
        adjacency: sp.csr_matrix,
        pieces: list[np.ndarray],
    ) -> int | None:
        candidates = [
            (idx, _one_vs_rest_conductance(adjacency, nodes))
            for idx, nodes in enumerate(pieces)
            if nodes.size >= 2
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])[0]

    def _split_piece(
        self,
        adjacency: sp.csr_matrix,
        nodes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        subgraph = adjacency[nodes][:, nodes].tocsr()
        if subgraph.nnz == 0:
            midpoint = nodes.size // 2
            return nodes[:midpoint], nodes[midpoint:]

        scores = _fiedler_vector(
            subgraph,
            tol=self.tol,
            max_iter=self.max_iter,
            seed=self.seed,
        )
        mask, _ = _sweep_cut(subgraph, scores)
        if not np.any(mask) or np.all(mask):
            midpoint = nodes.size // 2
            return nodes[:midpoint], nodes[midpoint:]
        return nodes[mask], nodes[~mask]
