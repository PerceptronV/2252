"""Evals package. Importing it side-effect-registers every concrete eval."""

from evals import base  # noqa: F401  -- registers _LabelAccuracyEval
from evals import clustering  # noqa: F401  -- registers clustering evals
