"""Markov clustering via expansion and inflation."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from algorithms.base import Algorithm
from core.graph import Graph
from core.registry import register_algorithm


def _column_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    column_sums = np.asarray(matrix.sum(axis=0)).ravel().astype(float, copy=False)
    scales = np.zeros_like(column_sums, dtype=float)
    positive = column_sums > 0
    scales[positive] = 1.0 / column_sums[positive]
    return (matrix @ sp.diags(scales)).tocsr()


def _expand(matrix: sp.csr_matrix, expansion: int) -> sp.csr_matrix:
    expanded = matrix
    for _ in range(expansion - 1):
        expanded = (expanded @ matrix).tocsr()
    return expanded


def _inflate(matrix: sp.csr_matrix, inflation: float, prune_threshold: float) -> sp.csr_matrix:
    inflated = matrix.copy().tocsr()
    inflated.data = np.power(inflated.data, inflation)
    if prune_threshold > 0:
        inflated.data[inflated.data < prune_threshold] = 0.0
        inflated.eliminate_zeros()
    return _column_normalize(inflated)


def _max_abs_difference(left: sp.csr_matrix, right: sp.csr_matrix) -> float:
    diff = (left - right).tocsr()
    if diff.nnz == 0:
        return 0.0
    return float(np.max(np.abs(diff.data)))


@register_algorithm("markov")
class MarkovClusteringAlgorithm(Algorithm):
    """Markov clustering with the standard e=2, r=2 defaults."""

    def __init__(
        self,
        expansion: int = 2,
        inflation: float = 2.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        prune_threshold: float = 1e-12,
        support_threshold: float = 1e-6,
    ) -> None:
        if expansion < 2:
            raise ValueError(f"expansion must be at least 2; got {expansion}")
        if inflation <= 1:
            raise ValueError(f"inflation must be greater than 1; got {inflation}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be positive; got {max_iter}")
        if tol < 0:
            raise ValueError(f"tol must be non-negative; got {tol}")
        if prune_threshold < 0:
            raise ValueError(f"prune_threshold must be non-negative; got {prune_threshold}")
        if support_threshold < 0:
            raise ValueError(f"support_threshold must be non-negative; got {support_threshold}")

        self.expansion = int(expansion)
        self.inflation = float(inflation)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.prune_threshold = float(prune_threshold)
        self.support_threshold = float(support_threshold)

    def fit_predict(self, graph: Graph, k: int) -> np.ndarray:
        adjacency = graph.adjacency.tocsr().astype(float)
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()

        matrix = adjacency + sp.identity(graph.num_nodes, format="csr")
        matrix = _column_normalize(matrix)

        for _ in range(self.max_iter):
            previous = matrix
            matrix = _inflate(
                _expand(matrix, self.expansion),
                self.inflation,
                self.prune_threshold,
            )
            if _max_abs_difference(matrix, previous) <= self.tol:
                break

        support = matrix.copy().tocsr()
        if self.support_threshold > 0:
            support.data[support.data < self.support_threshold] = 0.0
            support.eliminate_zeros()
        support = ((support + support.T) > 0).astype(int).tocsr()
        _, labels = connected_components(support, directed=False)
        return labels.astype(int)
