"""Algorithm abstract base class plus a constant-prediction stub for smoke tests."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.graph import Graph
from core.registry import register_algorithm


class Algorithm(ABC):
    """A graph-partitioning algorithm.

    Concrete subclasses register themselves via ``@register_algorithm("key")``
    and accept their hyperparameters through ``__init__``.
    """

    name: str = ""

    @abstractmethod
    def fit_predict(self, graph: Graph, k: int) -> np.ndarray:
        """Return predicted cluster labels of shape (graph.num_nodes,)."""


@register_algorithm("all_zeros")
class _AllZerosAlgorithm(Algorithm):
    """Assign every node to cluster 0. Trivial baseline, used only for smoke tests."""

    def __init__(self) -> None:
        pass

    def fit_predict(self, graph: Graph, k: int) -> np.ndarray:
        return np.zeros(graph.num_nodes, dtype=int)
