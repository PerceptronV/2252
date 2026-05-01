"""Dataset abstract base class plus a tiny trivial dataset for smoke tests."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

from core.graph import Graph
from core.registry import register_dataset


class Dataset(ABC):
    """A (graph, target_partition) source.

    Concrete subclasses register themselves via ``@register_dataset("key")`` and
    accept their hyperparameters through ``__init__`` so they can be constructed
    from a YAML config.
    """

    name: str = ""

    @abstractmethod
    def load(self) -> tuple[Graph, np.ndarray]:
        """Return (graph, target_partition); target_partition has shape (n,)."""

    @property
    @abstractmethod
    def num_clusters(self) -> int:
        """Ground-truth cluster count, forwarded to algorithms as `k`."""


@register_dataset("trivial")
class _TrivialDataset(Dataset):
    """Two disjoint edges over 4 nodes; ground truth is the two pairs.

    Used only for smoke-testing the runner end-to-end. Real datasets live in
    sibling modules registered alongside this one.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def load(self) -> tuple[Graph, np.ndarray]:
        rows = np.array([0, 1, 2, 3])
        cols = np.array([1, 0, 3, 2])
        data = np.ones(4, dtype=float)
        A = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))
        target = np.array([0, 0, 1, 1])
        return Graph(adjacency=A, num_nodes=4, name="trivial"), target

    @property
    def num_clusters(self) -> int:
        return 2
