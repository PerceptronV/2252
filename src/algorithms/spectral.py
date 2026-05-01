"""Spectral clustering algorithm with pluggable post-embedding baselines."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from algorithms._spectral import spectral_embedding
from algorithms.base import Algorithm
import baselines  # noqa: F401  -- side-effect: register all baselines
from core.graph import Graph
from core.registry import BASELINES, register_algorithm, resolve


def _apply_embedding_normalization(
    embedding: np.ndarray,
    graph: Graph,
    normalization: str,
) -> np.ndarray:
    if normalization == "none":
        return embedding

    if normalization == "unit_row":
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embedding / norms

    if normalization == "degree_sqrt":
        degrees = np.asarray(graph.adjacency.sum(axis=1)).ravel()
        scales = np.sqrt(degrees.astype(float, copy=False))
        normalized = np.zeros_like(embedding, dtype=float)
        mask = scales > 0
        normalized[mask] = embedding[mask] / scales[mask, None]
        return normalized

    raise ValueError(
        "normalization must be one of {'auto', 'degree_sqrt', 'unit_row', 'none'}; "
        f"got {normalization!r}"
    )


def _resolve_normalization(normalization: str, baseline: str) -> str:
    if normalization != "auto":
        return normalization
    if baseline == "peng_kmeans":
        return "degree_sqrt"
    return "unit_row"


@register_algorithm("spectral")
class SpectralClusteringAlgorithm(Algorithm):
    """Bottom-k eigenspace followed by a configurable baseline partitioner."""

    def __init__(
        self,
        embedding_dim: int | None = None,
        normalization: str = "auto",
        tol: float = 0.0,
        max_iter: int | None = None,
        seed: int | None = None,
        baseline: str = "kmeans",
        baseline_params: dict[str, Any] | None = None,
    ) -> None:
        if embedding_dim is not None and embedding_dim < 1:
            raise ValueError(f"embedding_dim must be positive; got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.normalization = normalization
        self.tol = float(tol)
        self.max_iter = max_iter
        self.seed = seed
        self.baseline = baseline
        self.baseline_params = dict(baseline_params or {})

    def fit_predict(self, graph: Graph, k: int) -> np.ndarray:
        embedding_dim = self.embedding_dim if self.embedding_dim is not None else k
        embedding = spectral_embedding(
            graph.adjacency,
            k=embedding_dim,
            row_normalize=False,
            tol=self.tol,
            max_iter=self.max_iter,
            seed=self.seed,
        ).embedding
        normalization = _resolve_normalization(self.normalization, self.baseline)
        embedding = _apply_embedding_normalization(
            embedding,
            graph,
            normalization,
        )
        baseline = self._make_baseline()
        return baseline.fit_predict(graph, embedding, k)

    def _make_baseline(self) -> Any:
        baseline_cls = resolve(BASELINES, self.baseline)
        kwargs = dict(self.baseline_params)

        sig = inspect.signature(baseline_cls.__init__)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )
        if (
            self.seed is not None
            and "seed" not in kwargs
            and ("seed" in sig.parameters or accepts_kwargs)
        ):
            kwargs["seed"] = self.seed

        return baseline_cls(**kwargs)
