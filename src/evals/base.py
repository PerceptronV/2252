"""Eval abstract base class plus a label-accuracy stub for smoke tests."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.graph import Graph
from core.registry import register_eval


class Eval(ABC):
    """A scalar evaluator over (graph, predicted, target).

    Some evals only use a subset of the inputs (e.g. conductance ignores
    ``target``, ARI ignores ``graph``). The uniform signature keeps the runner's
    call site identical for every eval. Concrete subclasses register themselves
    via ``@register_eval("key")``.
    """

    name: str = ""

    @abstractmethod
    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float: ...


@register_eval("label_accuracy")
class _LabelAccuracyEval(Eval):
    """Fraction of nodes whose predicted label equals the target label.

    Not a meaningful clustering metric (cluster ids are not identified across
    predicted vs. target); used only for smoke-testing the runner.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        return float(np.mean(predicted == target))
