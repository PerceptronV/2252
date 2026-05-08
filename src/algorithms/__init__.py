"""Algorithms package. Importing it side-effect-registers every concrete algorithm."""

from algorithms import base  # noqa: F401  -- registers _AllZerosAlgorithm
from algorithms import baseline  # noqa: F401  -- registers _AllZerosBaseline
from algorithms import kmeans  # noqa: F401  -- registers KMeansBaseline
from algorithms import fiedler  # noqa: F401  -- registers FiedlerSweepAlgorithm
from algorithms import louvain  # noqa: F401  -- registers LouvainAlgorithm
from algorithms import shi_malik  # noqa: F401  -- registers ShiMalikNormalizedCutAlgorithm
from algorithms import markov  # noqa: F401  -- registers MarkovClusteringAlgorithm
from algorithms import spectral  # noqa: F401  -- registers SpectralClusteringAlgorithm
