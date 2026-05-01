"""Baseline abstract base class plus a constant-prediction stub for smoke tests."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.graph import Graph
from core.registry import register_baseline


class Baseline(ABC):
    """A partition extractor over a graph-aware embedding.

    Baselines are intentionally more general than pure geometric clusterers:
    they receive both the graph and the embedding so future theoretical
    roundings (e.g. sweep cuts) can reuse the same interface.
    """

    name: str = ""

    @abstractmethod
    def fit_predict(self, graph: Graph, embedding: np.ndarray, k: int) -> np.ndarray:
        """Return predicted cluster labels of shape (graph.num_nodes,)."""


@register_baseline("all_zeros")
class _AllZerosBaseline(Baseline):
    """Assign every node to cluster 0. Trivial baseline, used only for smoke tests."""

    def __init__(self) -> None:
        pass

    def fit_predict(self, graph: Graph, embedding: np.ndarray, k: int) -> np.ndarray:
        return np.zeros(graph.num_nodes, dtype=int)
