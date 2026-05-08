"""Louvain community detection."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from algorithms.base import Algorithm
from core.graph import Graph
from core.registry import register_algorithm


def _modularity(adj: sp.csr_matrix, labels: np.ndarray) -> float:
    m2 = adj.sum()
    if m2 == 0:
        return 0.0

    degrees = np.asarray(adj.sum(axis=1)).ravel()
    q = 0.0

    for c in np.unique(labels):
        nodes = labels == c
        in_weight = adj[nodes][:, nodes].sum()
        degree_sum = degrees[nodes].sum()
        q += in_weight / m2 - (degree_sum / m2) ** 2

    return float(q)


def _local_phase(adj: sp.csr_matrix, labels: np.ndarray) -> tuple[np.ndarray, bool]:
    n = adj.shape[0]
    moved = False

    current_q = _modularity(adj, labels)

    improved = True
    while improved:
        improved = False

        for i in range(n):
            old_label = labels[i]

            row_start, row_end = adj.indptr[i], adj.indptr[i + 1]
            neighbors = adj.indices[row_start:row_end]

            candidate_labels = set(labels[neighbors])
            candidate_labels.add(old_label)

            best_label = old_label
            best_q = current_q

            for new_label in candidate_labels:
                if new_label == old_label:
                    continue

                trial = labels.copy()
                trial[i] = new_label
                q = _modularity(adj, trial)

                if q > best_q:
                    best_q = q
                    best_label = new_label

            if best_label != old_label:
                labels[i] = best_label
                current_q = best_q
                improved = True
                moved = True

    return labels, moved


def _aggregate_graph(adj: sp.csr_matrix, labels: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    unique_labels, new_labels = np.unique(labels, return_inverse=True)
    num_comms = len(unique_labels)

    rows = []
    cols = []
    data = []

    coo = adj.tocoo()
    for u, v, w in zip(coo.row, coo.col, coo.data):
        rows.append(new_labels[u])
        cols.append(new_labels[v])
        data.append(w)

    new_adj = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(num_comms, num_comms),
    ).tocsr()

    new_adj.sum_duplicates()
    return new_adj, new_labels


@register_algorithm("louvain")
class LouvainAlgorithm(Algorithm):
    """Louvain Modularity Maximization."""

    def __init__(self, max_iter: int = 20, tol: float = 1e-9) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def fit_predict(self, graph: Graph, k: int | None = None) -> np.ndarray:
        adj = graph.adjacency.tocsr().astype(float)
        adj.setdiag(0)
        adj.eliminate_zeros()

        n = graph.num_nodes
        original_to_current = np.arange(n)
        current_adj = adj
        prev_q = _modularity(current_adj, np.arange(current_adj.shape[0]))

        for _ in range(self.max_iter):
            labels = np.arange(current_adj.shape[0])
            labels, moved = _local_phase(current_adj, labels)
            q = _modularity(current_adj, labels)

            new_adj, current_to_next = _aggregate_graph(current_adj, labels)
            original_to_current = current_to_next[original_to_current]

            if not moved or q - prev_q <= self.tol:
                break

            current_adj = new_adj
            prev_q = q

        _, final = np.unique(original_to_current, return_inverse=True)
        return final
