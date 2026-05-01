"""Baselines package. Importing it side-effect-registers every concrete baseline."""

from baselines import base  # noqa: F401  -- registers _AllZerosBaseline
from baselines import kmeans  # noqa: F401  -- registers KMeansBaseline
