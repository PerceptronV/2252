"""Datasets package. Importing it side-effect-registers every concrete dataset."""

from data import base  # noqa: F401  -- registers _TrivialDataset
from data import serialized  # noqa: F401  -- registers SerializedGraphDataset
