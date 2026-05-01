"""Pure-NumPy k-means baseline for spectral embeddings."""

from __future__ import annotations

import numpy as np

from baselines.base import Baseline
from core.graph import Graph
from core.registry import register_baseline


def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


def _graph_degrees(graph: Graph) -> np.ndarray:
    return np.asarray(graph.adjacency.sum(axis=1)).ravel().astype(float, copy=False)


class _BaseKMeansBaseline(Baseline):
    """Shared Lloyd-style k-means implementation."""

    def __init__(
        self,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        seed: int | None = None,
    ) -> None:
        if n_init < 1:
            raise ValueError(f"n_init must be positive; got {n_init}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be positive; got {max_iter}")
        if tol < 0:
            raise ValueError(f"tol must be non-negative; got {tol}")
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = seed

    def _point_weights(self, graph: Graph) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(self, graph: Graph, embedding: np.ndarray, k: int) -> np.ndarray:
        X = np.asarray(embedding, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"embedding must have shape (n, d); got {X.shape}")
        n, _ = X.shape
        if n != graph.num_nodes:
            raise ValueError(
                f"embedding has {n} rows but graph has {graph.num_nodes} nodes"
            )
        if not (1 <= k <= n):
            raise ValueError(f"k must satisfy 1 <= k <= n; got k={k}, n={n}")

        weights = self._point_weights(graph)
        rng = np.random.default_rng(self.seed)
        best_labels: np.ndarray | None = None
        best_inertia = np.inf

        for _ in range(self.n_init):
            centers = self._init_centers(X, weights, k, rng)
            labels = np.zeros(n, dtype=int)

            for _ in range(self.max_iter):
                distances = _squared_distances(X, centers)
                new_labels = np.argmin(distances, axis=1)
                new_centers = self._recompute_centers(
                    X,
                    weights,
                    new_labels,
                    k,
                    distances,
                )

                center_shift = np.linalg.norm(new_centers - centers)
                if np.array_equal(new_labels, labels) or center_shift <= self.tol:
                    labels = new_labels
                    centers = new_centers
                    break

                labels = new_labels
                centers = new_centers

            inertia = float(np.sum(weights * np.sum((X - centers[labels]) ** 2, axis=1)))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()

        assert best_labels is not None
        return best_labels

    def _init_centers(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n, d = X.shape
        centers = np.empty((k, d), dtype=float)
        weight_sum = float(np.sum(weights))
        if weight_sum > 0:
            first_idx = int(rng.choice(n, p=weights / weight_sum))
        else:
            first_idx = int(rng.integers(n))
        centers[0] = X[first_idx]

        closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
        for idx in range(1, k):
            weighted_potential = weights * closest_sq
            total = float(np.sum(weighted_potential))
            if total <= 0:
                centers[idx:] = X[rng.choice(n, size=k - idx, replace=False)]
                break
            probs = weighted_potential / total
            next_idx = int(rng.choice(n, p=probs))
            centers[idx] = X[next_idx]
            dist_to_new = np.sum((X - centers[idx]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, dist_to_new)

        return centers

    def _recompute_centers(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        labels: np.ndarray,
        k: int,
        distances: np.ndarray,
    ) -> np.ndarray:
        centers = np.empty((k, X.shape[1]), dtype=float)
        counts = np.bincount(labels, minlength=k)

        # Re-seed empty clusters with the points farthest from their assigned center.
        fallback_order = np.argsort(np.min(distances, axis=1))[::-1]
        fallback_ptr = 0

        for cluster in range(k):
            if counts[cluster] == 0:
                while fallback_ptr < len(fallback_order):
                    candidate = fallback_order[fallback_ptr]
                    fallback_ptr += 1
                    if candidate >= 0:
                        centers[cluster] = X[candidate]
                        break
                else:
                    centers[cluster] = X[0]
                continue
            centers[cluster] = X[labels == cluster].mean(axis=0)

        return centers


@register_baseline("kmeans")
class KMeansBaseline(_BaseKMeansBaseline):
    """Naive unweighted k-means on embedding rows."""

    def _point_weights(self, graph: Graph) -> np.ndarray:
        return np.ones(graph.num_nodes, dtype=float)


@register_baseline("peng_kmeans")
class PengKMeansBaseline(_BaseKMeansBaseline):
    """Degree-weighted k-means matching Peng, Sun, and Zanetti."""

    def _point_weights(self, graph: Graph) -> np.ndarray:
        return _graph_degrees(graph)

    def _recompute_centers(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        labels: np.ndarray,
        k: int,
        distances: np.ndarray,
    ) -> np.ndarray:
        centers = np.empty((k, X.shape[1]), dtype=float)
        counts = np.bincount(labels, minlength=k)

        # Re-seed empty clusters with the points farthest from their assigned center.
        fallback_order = np.argsort(np.min(distances, axis=1))[::-1]
        fallback_ptr = 0

        for cluster in range(k):
            if counts[cluster] == 0:
                while fallback_ptr < len(fallback_order):
                    candidate = fallback_order[fallback_ptr]
                    fallback_ptr += 1
                    if candidate >= 0:
                        centers[cluster] = X[candidate]
                        break
                else:
                    centers[cluster] = X[0]
                continue
            mask = labels == cluster
            cluster_weights = weights[mask]
            if float(np.sum(cluster_weights)) <= 0:
                centers[cluster] = X[mask].mean(axis=0)
            else:
                centers[cluster] = np.average(X[mask], axis=0, weights=cluster_weights)

        return centers
