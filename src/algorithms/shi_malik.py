"""Shi-Malik normalized cut via recursive random-walk sweep cuts."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from algorithms._spectral import normalized_laplacian
from algorithms.base import Algorithm
from core.graph import Graph
from core.registry import register_algorithm


def _random_walk_fiedler_vector(
    adjacency: sp.csr_matrix,
    *,
    tol: float,
    max_iter: int | None,
    seed: int | None,
) -> np.ndarray:
    n = adjacency.shape[0]
    if n < 2:
        raise ValueError("Shi-Malik sweep requires at least two nodes")

    laplacian_sym, degrees = normalized_laplacian(adjacency)
    if n <= 64:
        vals, vecs = eigh(laplacian_sym.toarray())
        order = np.argsort(vals)
        sym_vec = vecs[:, order[1]]
    else:
        rng = np.random.default_rng(seed)
        shifted = 2.0 * sp.identity(n, format="csr") - laplacian_sym
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
        sym_vec = vecs[:, order[1]]

    scores = np.zeros(n, dtype=float)
    positive = degrees > 0
    scores[positive] = sym_vec[positive] / np.sqrt(degrees[positive])
    return scores


def _sweep_cut(adjacency: sp.csr_matrix, scores: np.ndarray) -> tuple[np.ndarray, float]:
    n = adjacency.shape[0]
    if n < 2:
        raise ValueError("sweep cut requires at least two nodes")

    order = np.argsort(scores, kind="mergesort")
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(float, copy=False)
    total_volume = float(np.sum(degrees))

    in_set = np.zeros(n, dtype=bool)
    current_cut = 0.0
    current_volume = 0.0
    best_ncut = float("inf")
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
        if current_volume <= 0 or complement_volume <= 0:
            continue
        ncut = current_cut / current_volume + current_cut / complement_volume
        if ncut < best_ncut:
            best_ncut = float(ncut)
            best_prefix = prefix_idx

    mask = np.zeros(n, dtype=bool)
    mask[order[:best_prefix]] = True
    return mask, best_ncut


def _one_vs_rest_ncut(adjacency: sp.csr_matrix, nodes: np.ndarray) -> float:
    n = adjacency.shape[0]
    if nodes.size == 0 or nodes.size == n:
        return 0.0

    mask = np.zeros(n, dtype=bool)
    mask[nodes] = True
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(float, copy=False)
    volume = float(np.sum(degrees[mask]))
    complement_volume = float(np.sum(degrees[~mask]))
    if volume <= 0 or complement_volume <= 0:
        return 0.0

    cut = float(adjacency[mask][:, ~mask].sum())
    return cut / volume + cut / complement_volume


@register_algorithm("shi_malik")
@register_algorithm("shi-malik")
class ShiMalikNormalizedCutAlgorithm(Algorithm):
    """Recursive Shi-Malik normalized cuts using eigenvectors of N'."""

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
            (idx, _one_vs_rest_ncut(adjacency, nodes))
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

        scores = _random_walk_fiedler_vector(
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
