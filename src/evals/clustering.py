"""Clustering metrics for graph partitioning experiments."""

from __future__ import annotations

import math

import numpy as np

from core.graph import Graph
from core.registry import register_eval
from evals.base import Eval


def _comb2(n: int) -> int:
    return n * (n - 1) // 2


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute the adjusted Rand index without external dependencies."""

    labels_true = np.asarray(labels_true, dtype=int)
    labels_pred = np.asarray(labels_pred, dtype=int)
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            f"label arrays must have the same shape; got {labels_true.shape} vs {labels_pred.shape}"
        )
    n = int(labels_true.size)
    if n < 2:
        return 1.0

    true_ids, true_inv = np.unique(labels_true, return_inverse=True)
    pred_ids, pred_inv = np.unique(labels_pred, return_inverse=True)
    contingency = np.zeros((true_ids.size, pred_ids.size), dtype=np.int64)
    np.add.at(contingency, (true_inv, pred_inv), 1)

    sum_comb = sum(_comb2(int(x)) for x in contingency.ravel())
    sum_true = sum(_comb2(int(x)) for x in contingency.sum(axis=1))
    sum_pred = sum(_comb2(int(x)) for x in contingency.sum(axis=0))
    total_comb = _comb2(n)
    if total_comb == 0:
        return 1.0

    expected = (sum_true * sum_pred) / total_comb
    max_index = 0.5 * (sum_true + sum_pred)
    denom = max_index - expected
    if math.isclose(denom, 0.0):
        return 1.0
    return float((sum_comb - expected) / denom)


def normalized_mutual_information(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute NMI using the arithmetic mean of label entropies."""

    labels_true = np.asarray(labels_true, dtype=int)
    labels_pred = np.asarray(labels_pred, dtype=int)
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            f"label arrays must have the same shape; got {labels_true.shape} vs {labels_pred.shape}"
        )
    n = int(labels_true.size)
    if n == 0:
        return 1.0

    _, true_inv = np.unique(labels_true, return_inverse=True)
    _, pred_inv = np.unique(labels_pred, return_inverse=True)
    contingency = np.zeros((true_inv.max() + 1, pred_inv.max() + 1), dtype=np.float64)
    np.add.at(contingency, (true_inv, pred_inv), 1.0)

    contingency /= float(n)
    true_probs = contingency.sum(axis=1)
    pred_probs = contingency.sum(axis=0)

    nz = contingency > 0
    expected = true_probs[:, None] * pred_probs[None, :]
    mutual_info = float(np.sum(contingency[nz] * np.log(contingency[nz] / expected[nz])))

    true_entropy = -float(np.sum(true_probs[true_probs > 0] * np.log(true_probs[true_probs > 0])))
    pred_entropy = -float(np.sum(pred_probs[pred_probs > 0] * np.log(pred_probs[pred_probs > 0])))
    denom = true_entropy + pred_entropy
    if math.isclose(denom, 0.0):
        return 1.0
    return float(2.0 * mutual_info / denom)


def cluster_conductances(graph: Graph, labels: np.ndarray) -> list[float]:
    """Return per-cluster conductance phi(S, V\\S) for all nontrivial clusters."""

    adjacency = graph.adjacency.tocsr()
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(float, copy=False)
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (graph.num_nodes,):
        raise ValueError(
            f"labels must have shape ({graph.num_nodes},); got {labels.shape}"
        )

    total_volume = float(np.sum(degrees))
    conductances: list[float] = []
    for cluster in np.unique(labels):
        mask = labels == cluster
        size = int(np.sum(mask))
        if size == 0 or size == graph.num_nodes:
            continue
        volume = float(np.sum(degrees[mask]))
        complement_volume = total_volume - volume
        denom = min(volume, complement_volume)
        if denom <= 0:
            continue

        cut_weight = float(adjacency[mask][:, ~mask].sum())
        conductances.append(cut_weight / denom)
    return conductances


def modularity(graph: Graph, labels: np.ndarray) -> float:
    """Return Newman-Girvan modularity for an undirected weighted graph."""

    adjacency = graph.adjacency.tocsr()
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (graph.num_nodes,):
        raise ValueError(
            f"labels must have shape ({graph.num_nodes},); got {labels.shape}"
        )

    m2 = float(adjacency.sum())
    if m2 <= 0:
        return 0.0

    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(float, copy=False)
    score = 0.0
    for cluster in np.unique(labels):
        mask = labels == cluster
        internal_weight = float(adjacency[mask][:, mask].sum())
        degree_sum = float(np.sum(degrees[mask]))
        score += internal_weight / m2 - (degree_sum / m2) ** 2
    return float(score)


@register_eval("ari")
class AdjustedRandIndexEval(Eval):
    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        return adjusted_rand_index(target, predicted)


@register_eval("nmi")
class NormalizedMutualInformationEval(Eval):
    """Normalized mutual information against target labels."""

    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        return normalized_mutual_information(target, predicted)


@register_eval("conductance")
class MeanConductanceEval(Eval):
    """Mean one-vs-rest conductance across predicted clusters."""

    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        values = cluster_conductances(graph, predicted)
        if not values:
            return float("inf")
        return float(np.mean(values))


@register_eval("min_conductance")
class MinConductanceEval(Eval):
    """Minimum one-vs-rest conductance across predicted clusters."""

    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        values = cluster_conductances(graph, predicted)
        if not values:
            return float("inf")
        return float(np.min(values))


@register_eval("max_conductance")
class MaxConductanceEval(Eval):
    """Maximum one-vs-rest conductance across predicted clusters."""

    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        values = cluster_conductances(graph, predicted)
        if not values:
            return float("inf")
        return float(np.max(values))


@register_eval("modularity")
class ModularityEval(Eval):
    """Newman-Girvan modularity of the predicted partition."""

    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        return modularity(graph, predicted)


@register_eval("returned_num_clusters")
class ReturnedNumClustersEval(Eval):
    """Number of clusters returned by the algorithm."""

    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        predicted = np.asarray(predicted, dtype=int)
        if predicted.shape != (graph.num_nodes,):
            raise ValueError(
                f"labels must have shape ({graph.num_nodes},); got {predicted.shape}"
            )
        return float(np.unique(predicted).size)
