"""Canonical graph representation passed between datasets, algorithms, and evals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Graph:
    adjacency: sp.csr_matrix
    num_nodes: int
    node_features: np.ndarray | None = None
    name: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        A = self.adjacency
        if not sp.issparse(A) or A.format != "csr":
            object.__setattr__(self, "adjacency", sp.csr_matrix(A))
            A = self.adjacency
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"adjacency must be square; got shape {A.shape}")
        if A.shape[0] != self.num_nodes:
            raise ValueError(
                f"num_nodes={self.num_nodes} but adjacency is {A.shape[0]}x{A.shape[1]}"
            )
        if self.node_features is not None and self.node_features.shape[0] != self.num_nodes:
            raise ValueError(
                f"node_features has {self.node_features.shape[0]} rows, "
                f"expected {self.num_nodes}"
            )
