"""Algorithms package. Importing it side-effect-registers every concrete algorithm."""

from algorithms import base  # noqa: F401  -- registers _AllZerosAlgorithm
from algorithms import spectral  # noqa: F401  -- registers SpectralClusteringAlgorithm
